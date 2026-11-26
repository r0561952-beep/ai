"""Обучение модели с нуля.

Примеры:
  python train.py                          # маленькая модель, процессор
  python train.py --preset medium --device cuda   # побольше, на видеокарте (Colab/Kaggle)
  python train.py --resume                 # продолжить с model/model.pt

Сохраняет:
  model/model.pt        — веса для чата (fp16, коммитится в репозиторий)
  checkpoints/full.pt   — полное состояние для продолжения (локально, не коммитится)
"""
import argparse
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from model import GPT, GPTConfig
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
p.add_argument("--resume", action="store_true")
p.add_argument("--no_bf16", action="store_true", help="не использовать bf16 (если процессор/GPU его не умеет)")
p.add_argument("--no_compile", action="store_true", help="не использовать torch.compile")
p.add_argument("--out", default="model/model.pt")
args = p.parse_args()

torch.manual_seed(1337)
torch.set_num_threads(os.cpu_count() or 4)

# ---------- токенизатор и данные ----------
if not os.path.exists(TOKENIZER_PATH):
    print("обучаю токенизатор ...")
    train_tokenizer()
tok = load_tokenizer()
fmt = ChatFormat(tok)

os.makedirs("checkpoints", exist_ok=True)
train_bin, val_bin = "checkpoints/train.bin", "checkpoints/val.bin"
if not os.path.exists(train_bin):
    print("перевожу диалоги в токены ...")
    dialogs = list(iter_dialogs())
    n_val = max(500, len(dialogs) // 50)
    for path, part in ((val_bin, dialogs[:n_val]), (train_bin, dialogs[n_val:])):
        ids = [t for d in part for t in fmt.encode_dialog(d)]
        np.array(ids, dtype=np.uint16).tofile(path)
train_data = np.fromfile(train_bin, dtype=np.uint16)
val_data = np.fromfile(val_bin, dtype=np.uint16)
print(f"токенов: обучение {len(train_data):,}, проверка {len(val_data):,}")

# ---------- модель ----------
# полное состояние (с оптимизатором) для продолжения обучения
full_ckpt = "checkpoints/full.pt" if args.out == "model/model.pt" else args.out + ".full"
start_iter, best_val = 0, float("inf")
if args.resume:
    src = full_ckpt if os.path.exists(full_ckpt) else args.out
    ckpt = torch.load(src, map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})
    start_iter = ckpt.get("iter", 0)
    best_val = ckpt.get("val_loss", float("inf"))
    print(f"продолжаю с {src}, шаг {start_iter}")
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


def get_batch(data):
    ix = np.random.randint(0, len(data) - cfg.block_size - 1, args.batch_size)
    x = torch.from_numpy(np.stack([data[i:i + cfg.block_size] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + cfg.block_size] for i in ix]).astype(np.int64))
    return x.to(args.device), y.to(args.device)


@torch.no_grad()
def evaluate(iters=20):
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(iters):
            with amp_ctx:
                losses.append(fwd(*get_batch(data))[1].item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def lr_at(it):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    frac = min(1.0, (it - args.warmup) / max(1, args.max_iters - args.warmup))
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * frac)) * (args.lr - args.min_lr)


def save(it, val_loss):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    meta = {"config": asdict(cfg), "iter": it, "val_loss": val_loss}
    torch.save({**meta, "model": {k: v.half() for k, v in state.items()}}, args.out)
    torch.save({**meta, "model": state, "optimizer": opt.state_dict()}, full_ckpt)


@torch.no_grad()
def sample(question="Что такое солнце?"):
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
        print(f"шаг {it}: train {losses['train']:.3f}, val {losses['val']:.3f}  ({el:.1f} мин)", flush=True)
        if it > start_iter:
            if losses["val"] < best_val:
                best_val = losses["val"]
            save(it, losses["val"])
            print("  пример ответа на «Что такое солнце?»:", sample().replace("\n", " ")[:200], flush=True)
    if it == args.max_iters:
        break
    if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
        print("лимит времени — сохраняю и выхожу")
        save(it, evaluate()["val"])
        break

    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    x, y = get_batch(train_data)
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
