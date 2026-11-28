"""Дообучение готовой модели Qwen (base) на наших ролевых данных (LoRA).

Прогресс сохраняется в --out: если обучение прервалось, запусти ту же команду
снова — оно продолжится с места остановки (так же и с --run_minutes кусками).

  python prepare_rp.py
  python qwen_train.py --out checkpoints/qwen35_rp                                # Qwen3.5-2B (по умолчанию)
  python qwen_train.py --base Qwen/Qwen2.5-1.5B --out /content/drive/MyDrive/ai_qwen   # прошлая версия

Результат: <out>/adapter — веса LoRA, их подключает qwen_chat.py.
Модель учится только на репликах персонажа (реплики собеседника и карточка — контекст).
--fetish_share: какая доля 18+ кусков берётся из сцен с БДСМ/фетишами. Сцены с
несовершеннолетними (уже вычищены в prepare_rp.py), зоофилией, некрофилией и
изнасилованием в обучение не попадают.
"""
import argparse
import gzip
import json
import math
import os
import random
import re
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--base", default="Qwen/Qwen3.5-2B-Base", help="готовая модель с Hugging Face (base, без «воспитания»)")
p.add_argument("--out", default="qwen_out")
p.add_argument("--n_rp18", type=int, default=10000, help="сколько 18+ ролевых кусков взять")
p.add_argument("--n_rp", type=int, default=6000, help="сколько обычных ролевых кусков взять")
p.add_argument("--fetish_share", type=float, default=0.0, help="доля 18+ кусков из сцен с БДСМ/фетишами (0 — как попадётся)")
p.add_argument("--n_val", type=int, default=0, help="отложить N кусков для проверки «до/после» (в обучение не идут)")
p.add_argument("--max_len", type=int, default=768)
p.add_argument("--batch_size", type=int, default=4)
p.add_argument("--grad_accum", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--lora_r", type=int, default=16)
p.add_argument("--max_minutes", type=float, default=0, help="остановиться через N минут суммарно (0 = до конца)")
p.add_argument("--run_minutes", type=float, default=0, help="остановить ЭТОТ запуск через N минут (продолжить — запуском снова)")
p.add_argument("--save_every", type=int, default=50, help="сохранять прогресс каждые N шагов")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--no_grad_ckpt", action="store_true",
               help="не пересчитывать активации (быстрее на ~30%, но нужно больше памяти)")
args = p.parse_args()

SYSTEM_DEFAULT = "Ты — живой собеседник в ролевой игре. Отвечай от лица своего персонажа."
FETISH = re.compile(r"бдсм|bdsm|связа[нл]|связыва|верёвк|веревк|наручник|ошейник|поводок|плеть|плётк|плетк|кнут|кляп|"
                    r"повязк\w* на глаз|госпож|хозяйк|рабын|доминир|подчин|сабмис|фетиш|кинк|латекс|чулк|ступн|"
                    r"пальчик\w* ног|шлёп|шлеп|порк|выпорот|игрушк|вибратор|страпон", re.I)
EXCLUDE = re.compile(r"зоофил|скотолож|bestial|некрофил|necrophil|изнасил|без (?:её|его) согласия|против (?:её|его) воли", re.I)
THINK = ""  # у Qwen3/3.5 — пустой блок «рассуждений»: модель сразу отвечает (как в её шаблоне чата)


def chatml(role, text):
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def encode_dialog(tok, messages):
    """ChatML: карточка персонажа -> system, дальше user/assistant.
    labels = -100 везде, кроме ответов персонажа (их модель и учится писать)."""
    ids, labels = [], []
    msgs = list(messages)
    system = SYSTEM_DEFAULT
    if msgs and msgs[0]["role"] == "user" and msgs[0]["content"].startswith("[Ролевая игра]"):
        system = msgs.pop(0)["content"]
    for role, text in [("system", system)] + [(m["role"], m["content"]) for m in msgs]:
        if role == "assistant":
            head = tok(f"<|im_start|>{role}\n{THINK}", add_special_tokens=False).input_ids
            body = tok(f"{text}<|im_end|>\n", add_special_tokens=False).input_ids
            ids += head + body
            labels += [-100] * len(head) + body
        else:
            piece = tok(chatml(role, text), add_special_tokens=False).input_ids
            ids += piece
            labels += [-100] * len(piece)
    return ids[:args.max_len], labels[:args.max_len]


def load_examples(tok):
    """Возвращает (обучающие, проверочные) примеры. Выбор всегда одинаковый (seed) —
    продолжение после обрыва идёт по тем же данным."""
    rng = random.Random(0)
    examples, val = [], []
    for path, n, fetish in (("data/rp18_ru.jsonl.gz", args.n_rp18, args.fetish_share),
                            ("data/rp_ru.jsonl.gz", args.n_rp, 0.0)):
        if n <= 0 or not os.path.exists(path):
            continue
        rows = [json.loads(l)["messages"] for l in gzip.open(path, "rt", encoding="utf-8")]
        rng.shuffle(rows)
        n_val = round(args.n_val * n / max(1, args.n_rp18 + args.n_rp))
        text = lambda msgs: "\n".join(m["content"] for m in msgs)
        rows = [r for r in rows if not EXCLUDE.search(text(r))]
        val_rows, rows = rows[:n_val], rows[n_val:]
        if fetish > 0:
            hot = [r for r in rows if FETISH.search(text(r))]
            k = min(len(hot), round(n * fetish))
            rest = [r for r in rows if not FETISH.search(text(r))]
            picked = hot[:k] + rest[:n - k]
            print(f"{path}: сцен с БДСМ/фетишами {k} из {len(hot)}")
        else:
            picked = rows[:n]
        for dst, part in ((examples, picked), (val, val_rows)):
            for msgs in part:
                ids, labels = encode_dialog(tok, msgs)
                if any(l != -100 for l in labels):
                    dst.append((ids, labels))
        print(f"{path}: взято {len(picked)} кусков (+{len(val_rows)} для проверки)")
    rng.shuffle(examples)
    return examples, val


def batches(examples, pad_id):
    """Пачки из примеров похожей длины: меньше «пустых» токенов-заполнителей — обучение
    в ~2 раза быстрее. Порядок пачек перемешан, но всегда одинаков (продолжение после обрыва)."""
    chunk = args.batch_size * 64
    order = []
    for i in range(0, len(examples), chunk):
        block = sorted(examples[i:i + chunk], key=lambda x: len(x[0]))
        order += [block[j:j + args.batch_size] for j in range(0, len(block), args.batch_size)]
    random.Random(1).shuffle(order)
    for part in order:
        L = max(len(x[0]) for x in part)
        ids = torch.full((len(part), L), pad_id)
        lab = torch.full((len(part), L), -100)
        att = torch.zeros((len(part), L), dtype=torch.long)
        for j, (x, y) in enumerate(part):
            ids[j, :len(x)], lab[j, :len(y)], att[j, :len(x)] = torch.tensor(x), torch.tensor(y), 1
        yield ids, lab, att


def main():
    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, "state.pt")
    adapter_dir = os.path.join(args.out, "adapter")
    cuda = args.device.startswith("cuda")
    torch.set_num_threads(os.cpu_count() or 4)
    # видеокарта T4: fp16; процессор с AMX/AVX512-BF16: bf16 (в 2-3 раза быстрее fp32); иначе fp32
    try:
        cpu_bf16 = not cuda and any(f in open("/proc/cpuinfo").read() for f in ("amx_bf16", "avx512_bf16"))
    except OSError:
        cpu_bf16 = False
    amp = torch.float16 if cuda else (torch.bfloat16 if cpu_bf16 else None)
    print("точность:", amp or "fp32")
    if cuda:
        print("видеокарта:", torch.cuda.get_device_name(0))

    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
    global THINK
    THINK = "<think>\n\n</think>\n\n" if "<think>" in tok.get_vocab() else ""
    examples, val = load_examples(tok)
    n_tok = sum(len(x[0]) for x in examples)
    steps_total = math.ceil(len(examples) / (args.batch_size * args.grad_accum))
    print(f"примеров: {len(examples)}, токенов: {n_tok:,}, шагов: {steps_total}")

    # fp16 на T4 (bf16 там нет); сама модель заморожена, учатся только веса LoRA (в fp32)
    dtype = amp or torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(args.device)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    if os.path.exists(adapter_dir):
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
        print("продолжаю с сохранённого адаптера")
    else:
        # все линейные слои языковой модели (у Qwen3.5 есть ещё слои линейного внимания in_proj_*/out_proj)
        cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                         target_modules="all-linear")
        model = get_peft_model(model, cfg)
    # dropout в LoRA выключаем: на процессоре он съедал ~половину времени, а пользы почти не даёт
    for module in model.modules():
        if hasattr(module, "lora_dropout"):
            for key in list(module.lora_dropout.keys()):
                module.lora_dropout[key] = torch.nn.Identity()
    for prm in model.parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()
    model.print_trainable_parameters()

    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    step, minutes_before, val_before = 0, 0.0, None
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        step, minutes_before, val_before = st["step"], st["minutes"], st.get("val_before")
        print(f"продолжаю с шага {step}/{steps_total}, уже обучалась {minutes_before:.0f} мин")
    if step >= steps_total:
        print("обучение уже завершено")
        return

    def loss_of(ids, lab):
        """Ошибка только на репликах персонажа. Логиты (словарь ~150–250 тыс. слов!) считаем
        лишь там, где модель учится, — на ~15% быстрее, результат тот же."""
        keep = torch.zeros_like(lab, dtype=torch.bool)
        keep[:, :-1] = lab[:, 1:] != -100
        with torch.autocast("cuda" if cuda else "cpu", dtype=amp or torch.float32, enabled=amp is not None):
            # маску внимания не передаём: заполнитель стоит в конце, а модель смотрит только назад —
            # настоящие токены его не видят. Без маски внимание считается быстрым путём.
            logits = model(input_ids=ids, logits_to_keep=keep).logits
        return F.cross_entropy(logits.float(), lab[:, 1:][keep[:, :-1]])

    def evaluate():
        """Средняя ошибка на отложенных кусках (только на репликах персонажа)."""
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for ids, lab in val:
                loss = loss_of(torch.tensor([ids], device=args.device), torch.tensor([lab], device=args.device))
                k = sum(l != -100 for l in lab[1:])
                tot, n = tot + loss.item() * k, n + k
        model.train()
        return tot / max(1, n)

    if val and step == 0 and val_before is None:
        val_before = evaluate()
        print(f"проверка ДО обучения: ошибка {val_before:.3f} на {len(val)} новых сценах", flush=True)

    def lr_at(s):
        warm = max(1, steps_total // 20)
        if s < warm:
            return args.lr * (s + 1) / warm
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm))))

    def save(done):
        """done — сколько шагов уже выполнено (с него продолжим)."""
        model.save_pretrained(adapter_dir)
        torch.save({"optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "step": done, "val_before": val_before,
                    "minutes": minutes_before + (time.time() - t0) / 60}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    model.train()
    t0 = time.time()
    all_batches = list(batches(examples, pad_id))
    groups = [all_batches[i:i + args.grad_accum] for i in range(0, len(all_batches), args.grad_accum)]
    loss_avg, seen, start = None, 0, step
    for step in range(start, steps_total):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        total = 0.0
        for ids, lab, att in groups[step]:
            loss = loss_of(ids.to(args.device), lab.to(args.device)) / len(groups[step])
            scaler.scale(loss).backward()
            total += loss.item()
            seen += int(att.sum())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if math.isfinite(total):  # редкий NaN в fp16 — шаг пропускает GradScaler, в среднее не берём
            loss_avg = total if loss_avg is None else 0.95 * loss_avg + 0.05 * total
        el = (time.time() - t0) / 60
        done = step + 1
        if step % 10 == 0 or done == steps_total:
            left = el / (done - start) * (steps_total - done)
            print(f"шаг {done}/{steps_total}: loss {loss_avg or float('nan'):.3f} | {seen / max(1e-6, el * 60):,.0f} ток/с | "
                  f"{el:.1f} мин, осталось ~{left:.0f} мин", flush=True)
        if done % args.save_every == 0 or done == steps_total:
            save(done)
        if done < steps_total and ((args.max_minutes and minutes_before + el > args.max_minutes)
                                   or (args.run_minutes and el > args.run_minutes)):
            save(done)
            print("лимит времени — прогресс сохранён, запусти снова, чтобы продолжить")
            return
    if val:
        after = evaluate()
        print(f"проверка ПОСЛЕ обучения: ошибка {after:.3f} (до: {val_before if val_before is not None else float('nan'):.3f})")
    print(f"✅ готово! адаптер: {adapter_dir}")


if __name__ == "__main__":
    main()
