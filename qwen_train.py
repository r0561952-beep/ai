"""Дообучение готовой модели Qwen2.5 (base) на наших ролевых данных (LoRA).

Нужна видеокарта (Colab/Kaggle T4). Прогресс сохраняется в --out: если Colab
отключился, запусти ту же команду снова — обучение продолжится с места остановки.

  python prepare_rp.py
  python qwen_train.py --out /content/drive/MyDrive/ai_qwen

Результат: <out>/adapter — веса LoRA (~70 МБ), их подключает qwen_chat.py.
Модель учится только на репликах персонажа (реплики собеседника и карточка — контекст).
"""
import argparse
import gzip
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--base", default="Qwen/Qwen2.5-1.5B", help="готовая модель с Hugging Face (base, без «воспитания»)")
p.add_argument("--out", default="qwen_out")
p.add_argument("--n_rp18", type=int, default=10000, help="сколько 18+ ролевых кусков взять")
p.add_argument("--n_rp", type=int, default=6000, help="сколько обычных ролевых кусков взять")
p.add_argument("--max_len", type=int, default=768)
p.add_argument("--batch_size", type=int, default=4)
p.add_argument("--grad_accum", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--lora_r", type=int, default=16)
p.add_argument("--max_minutes", type=float, default=0, help="остановиться через N минут суммарно (0 = до конца)")
p.add_argument("--run_minutes", type=float, default=0, help="остановить ЭТОТ запуск через N минут (продолжить — запуском снова)")
p.add_argument("--save_every", type=int, default=50, help="сохранять прогресс каждые N шагов")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

SYSTEM_DEFAULT = "Ты — живой собеседник в ролевой игре. Отвечай от лица своего персонажа."


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
    pieces = [("system", system, False)] + [(m["role"], m["content"], m["role"] == "assistant") for m in msgs]
    for role, text, train in pieces:
        if train:
            head = tok(f"<|im_start|>{role}\n", add_special_tokens=False).input_ids
            body = tok(f"{text}<|im_end|>\n", add_special_tokens=False).input_ids
            ids += head + body
            labels += [-100] * len(head) + body
        else:
            piece = tok(chatml(role, text), add_special_tokens=False).input_ids
            ids += piece
            labels += [-100] * len(piece)
    return ids[:args.max_len], labels[:args.max_len]


def load_examples(tok):
    rng = random.Random(0)
    examples = []
    for path, n in (("data/rp18_ru.jsonl.gz", args.n_rp18), ("data/rp_ru.jsonl.gz", args.n_rp)):
        if n <= 0 or not os.path.exists(path):
            continue
        rows = [json.loads(l)["messages"] for l in gzip.open(path, "rt", encoding="utf-8")]
        rng.shuffle(rows)
        for msgs in rows[:n]:
            ids, labels = encode_dialog(tok, msgs)
            if any(l != -100 for l in labels):
                examples.append((ids, labels))
        print(f"{path}: взято {min(n, len(rows))} кусков")
    rng.shuffle(examples)
    return examples


def batches(examples, pad_id):
    for i in range(0, len(examples), args.batch_size):
        part = examples[i:i + args.batch_size]
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
    examples = load_examples(tok)
    n_tok = sum(len(x[0]) for x in examples)
    steps_total = math.ceil(len(examples) / (args.batch_size * args.grad_accum))
    print(f"примеров: {len(examples)}, токенов: {n_tok:,}, шагов: {steps_total}")

    # fp16 на T4 (bf16 там нет); сама модель заморожена, учатся только веса LoRA (в fp32)
    dtype = amp or torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(args.device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    if os.path.exists(adapter_dir):
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
        print("продолжаю с сохранённого адаптера")
    else:
        cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
    for prm in model.parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()
    model.print_trainable_parameters()

    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    step, minutes_before = 0, 0.0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        step, minutes_before = st["step"], st["minutes"]
        print(f"продолжаю с шага {step}/{steps_total}, уже обучалась {minutes_before:.0f} мин")
    if step >= steps_total:
        print("обучение уже завершено")
        return

    def lr_at(s):
        warm = max(1, steps_total // 20)
        if s < warm:
            return args.lr * (s + 1) / warm
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm))))

    def save(done):
        """done — сколько шагов уже выполнено (с него продолжим)."""
        model.save_pretrained(adapter_dir)
        torch.save({"optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "step": done,
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
            with torch.autocast("cuda" if cuda else "cpu", dtype=amp or torch.float32, enabled=amp is not None):
                out = model(input_ids=ids.to(args.device), attention_mask=att.to(args.device),
                            labels=lab.to(args.device))
            loss = out.loss / len(groups[step])
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
    print(f"✅ готово! адаптер: {adapter_dir}")


if __name__ == "__main__":
    main()
