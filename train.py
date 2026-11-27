"""Обучение модели с нуля.

Примеры:
  python train.py                          # маленькая модель, процессор
  python train.py --preset medium --device cuda   # побольше, на видеокарте (Colab/Kaggle)
  python train.py --resume                 # продолжить с checkpoints/full.pt
  python train.py --init_from model/model.pt --mix wiki=0.7,qa=0.3 --lr 6e-4
                                           # дообучить готовую модель на Википедии + диалогах

Сохраняет:
  checkpoints/full.pt   — каждые eval_every шагов: полное состояние для продолжения
                          (папка checkpoints/ не коммитится)
  model/model.pt        — в конце обучения: финальные веса для чата (fp16, коммитится)
"""
import argparse
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from model import GPT, GPTConfig, export_state, load_model
from tokenizer_utils import TOKENIZER_PATH, ChatFormat, iter_dialogs, load_tokenizer, train_tokenizer

PRESETS = {
    # ~ параметров без эмбеддингов: tiny ≈ 3M, small ≈ 11M, medium ≈ 42M
    "tiny":   dict(n_layer=4, n_head=4, n_embd=256, block_size=256),
    "small":  dict(n_layer=6, n_head=6, n_embd=384, block_size=256),
    "medium": dict(n_layer=8, n_head=8, n_embd=640, block_size=512),
}

p = argparse.ArgumentParser()
p.add_argument("--preset", default="small", choices=PRESETS)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--max_iters", type=int, default=3500)
p.add_argument("--max_minutes", type=float, default=0, help="остановиться через N минут (0 = без лимита)")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--min_lr", type=float, default=1e-4)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--eval_every", type=int, default=250)
p.add_argument("--resume", action="store_true", help="продолжить прерванное обучение (тот же график lr)")
p.add_argument("--init_from", help="начать новый цикл обучения с готовых весов, например model/model.pt")
p.add_argument("--mix", default="qa=1", help="доли источников данных, например wiki=0.7,qa=0.3")
p.add_argument("--no_bf16", action="store_true", help="не использовать bf16 (если процессор/GPU его не умеет)")
p.add_argument("--no_compile", action="store_true", help="не использовать torch.compile")
p.add_argument("--out", default="model/model.pt", help="куда сохранить финальные веса")
p.add_argument("--ckpt_dir", default="checkpoints", help="папка для промежуточных сохранений")
args = p.parse_args()

torch.manual_seed(1337)
torch.set_num_threads(os.cpu_count() or 4)

# ---------- токенизатор и данные ----------
if not os.path.exists(TOKENIZER_PATH):
    print("обучаю токенизатор ...")
    train_tokenizer()
tok = load_tokenizer()
fmt = ChatFormat(tok)

SOURCES = {  # имя -> (файл с диалогами, префикс файлов с токенами)
    "qa": ("data/qa_ru.jsonl.gz", ""),               # python prepare_data.py
    "wiki": ("data/wiki_qa_ru.jsonl.gz", "wiki_"),   # python prepare_wiki.py
}
mix = {k: float(v) for k, v in (x.split("=") for x in args.mix.split(","))}
assert set(mix) <= set(SOURCES), f"неизвестный источник в --mix: {set(mix) - set(SOURCES)}"


def encode_dialogs(dialogs, out_path, chunk=5000):
    """Переводит диалоги в токены пачками (быстро и без лишней памяти)."""
    with open(out_path, "wb") as f:
        for i in range(0, len(dialogs), chunk):
            part = dialogs[i:i + chunk]
            enc = tok.encode_batch([m["content"] for d in part for m in d])
            ids, k = [], 0
            for d in part:
                for m in d:
                    ids += [fmt.user if m["role"] == "user" else fmt.bot] + enc[k].ids + [fmt.end]
                    k += 1
            np.array(ids, dtype=np.uint16).tofile(f)


os.makedirs(args.ckpt_dir, exist_ok=True)
data = {}
for name in mix:
    path, prefix = SOURCES[name]
    train_bin = os.path.join(args.ckpt_dir, prefix + "train.bin")
    val_bin = os.path.join(args.ckpt_dir, prefix + "val.bin")
    if not os.path.exists(train_bin):
        print(f"перевожу {path} в токены ...")
        dialogs = list(iter_dialogs(path))
        n_val = max(500, len(dialogs) // 50)
        encode_dialogs(dialogs[:n_val], val_bin)
        encode_dialogs(dialogs[n_val:], train_bin)
    data[name] = {"train": np.fromfile(train_bin, dtype=np.uint16), "val": np.fromfile(val_bin, dtype=np.uint16)}
    print(f"{name}: доля {mix[name]}, токенов: обучение {len(data[name]['train']):,}, "
          f"проверка {len(data[name]['val']):,}")
mix_names = list(mix)
mix_p = np.array([mix[n] for n in mix_names]) / sum(mix.values())

# ---------- модель ----------
# полное состояние (с оптимизатором) для продолжения обучения
full_ckpt = os.path.join(args.ckpt_dir, "full.pt")
start_iter, best_val = 0, float("inf")
if args.resume:
    src = full_ckpt if os.path.exists(full_ckpt) else args.out
    model, ckpt = load_model(src)
    cfg = model.cfg
    start_iter = ckpt.get("iter", 0)
    best_val = ckpt.get("val_loss", float("inf"))
    print(f"продолжаю с {src}, шаг {start_iter}")
elif args.init_from:
    model, init = load_model(args.init_from)
    cfg = model.cfg
    ckpt = None
    print(f"начинаю новый цикл обучения с весов {args.init_from} (их шаг {init.get('iter')})")
else:
    cfg = GPTConfig(vocab_size=tok.get_vocab_size(), dropout=args.dropout, **PRESETS[args.preset])
    model = GPT(cfg)
    ckpt = None
model.to(args.device)
print(f"модель: {model.num_params() / 1e6:.1f}M параметров, {asdict(cfg)}")

decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95),
)
if ckpt is not None and "optimizer" in ckpt:
    opt.load_state_dict(ckpt["optimizer"])

# bf16 ускоряет обучение в 2-3 раза на видеокартах и процессорах с AMX/AVX512-BF16
def cpu_has_bf16():
    try:
        flags = open("/proc/cpuinfo").read()
        return "amx_bf16" in flags or "avx512_bf16" in flags
    except OSError:
        return False


dev_type = "cuda" if args.device.startswith("cuda") else "cpu"
amp_dtype = None
if not args.no_bf16:
    if dev_type == "cuda":
        # старые видеокарты (T4 в Colab) не умеют bf16 -> используем fp16
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    elif cpu_has_bf16():
        amp_dtype = torch.bfloat16
amp_ctx = torch.autocast(dev_type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None)
scaler = torch.amp.GradScaler(dev_type, enabled=amp_dtype == torch.float16)
fwd = model if args.no_compile else torch.compile(model)
print(f"точность: {amp_dtype or 'fp32'}, compile: {not args.no_compile}")


def get_batch(split, source=None):
    """Каждая строка батча берётся из источника, выбранного по долям --mix."""
    srcs = [source] * args.batch_size if source else np.random.choice(mix_names, args.batch_size, p=mix_p)
    xs, ys = [], []
    for s in srcs:
        arr = data[s][split]
        i = np.random.randint(0, len(arr) - cfg.block_size - 1)
        xs.append(arr[i:i + cfg.block_size])
        ys.append(arr[i + 1:i + 1 + cfg.block_size])
    x = torch.from_numpy(np.stack(xs).astype(np.int64))
    y = torch.from_numpy(np.stack(ys).astype(np.int64))
    return x.to(args.device), y.to(args.device)


@torch.no_grad()
def evaluate(iters=20):
    """Ошибка по каждому источнику и общая (взвешенная по долям)."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        total = 0.0
        for name, w in zip(mix_names, mix_p):
            losses = []
            for _ in range(max(4, int(iters * w))):
                with amp_ctx:
                    losses.append(fwd(*get_batch(split, name))[1].item())
            out[f"{split}_{name}"] = sum(losses) / len(losses)
            total += w * out[f"{split}_{name}"]
        out[split] = float(total)
    model.train()
    return out


def lr_at(it):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    frac = min(1.0, (it - args.warmup) / max(1, args.max_iters - args.warmup))
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * frac)) * (args.lr - args.min_lr)


def save(it, val_loss, final=False):
    meta = {"config": asdict(cfg), "iter": int(it), "val_loss": float(val_loss)}
    torch.save({**meta, "model": export_state(model, torch.float32), "optimizer": opt.state_dict()}, full_ckpt)
    if final:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({**meta, "model": export_state(model)}, args.out)
        print(f"финальные веса сохранены в {args.out}")


@torch.no_grad()
def sample(question="Что такое Солнце?"):
    model.eval()
    ids = fmt.encode_message("user", question) + [fmt.bot]
    out = model.generate(torch.tensor([ids], device=args.device), 80, stop_token=fmt.end)
    model.train()
    return fmt.decode(out[0, len(ids):].tolist())


# ---------- цикл обучения ----------
t0 = time.time()
tokens_per_step = args.batch_size * cfg.block_size
it = start_iter
for it in range(start_iter, args.max_iters + 1):
    if it % args.eval_every == 0 or it == args.max_iters:
        losses = evaluate()
        el = (time.time() - t0) / 60
        parts = ", ".join(f"{n} {losses['val_' + n]:.3f}" for n in mix_names) if len(mix_names) > 1 else ""
        print(f"шаг {it}: train {losses['train']:.3f}, val {losses['val']:.3f}"
              f"{' (' + parts + ')' if parts else ''}  ({el:.1f} мин)", flush=True)
        if it > start_iter:
            if losses["val"] < best_val:
                best_val = losses["val"]
            save(it, losses["val"], final=(it == args.max_iters))
            for q in ("Что такое Солнце?", "Как приготовить блины?"):
                print(f"  пример ответа на «{q}»:", sample(q).replace("\n", " ")[:200], flush=True)
    if it == args.max_iters:
        break
    if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
        print("лимит времени — сохраняю и выхожу")
        save(it, evaluate()["val"], final=True)
        break

    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    x, y = get_batch("train")
    with amp_ctx:
        _, loss = fwd(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    if it % 50 == 0:
        dt = time.time() - t0
        done = it - start_iter + 1
        print(f"  шаг {it} loss {loss.item():.3f} | {done * tokens_per_step / dt:,.0f} ток/с", flush=True)

print(f"готово, лучшая val loss {best_val:.3f}, модель в {args.out}")
