"""Чат с обученной моделью. Модель помнит предыдущие реплики диалога
(сколько влезает в её окно контекста).

  python chat.py                       # интерактивный чат
  python chat.py -q "Что такое луна?"  # один вопрос
Команды в чате: /new — забыть диалог, /exit — выйти.
"""
import argparse

import torch

from model import load_model
from tokenizer_utils import ChatFormat, load_tokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="model/model.pt", help="или checkpoints/full.pt во время обучения")
p.add_argument("-q", "--question")
p.add_argument("--temperature", type=float, default=0.7)
p.add_argument("--top_k", type=int, default=40)
p.add_argument("--max_new_tokens", type=int, default=150)
args = p.parse_args()

model, ckpt = load_model(args.model)
model.eval()
fmt = ChatFormat(load_tokenizer())
budget = model.cfg.block_size - args.max_new_tokens  # место под историю

history = []  # список закодированных реплик — «память» диалога


def answer(question):
    history.append(fmt.encode_message("user", question))
    # убираем самые старые реплики, если не влезают в окно контекста
    while len(history) > 1 and sum(map(len, history)) + 1 > budget:
        history.pop(0)
    ids = [t for turn in history for t in turn][-budget + 1:] + [fmt.bot]
    out = model.generate(torch.tensor([ids]), args.max_new_tokens, temperature=args.temperature,
                         top_k=args.top_k, stop_token=fmt.end)
    reply_ids = out[0, len(ids):].tolist()
    if not reply_ids or reply_ids[-1] != fmt.end:
        reply_ids.append(fmt.end)
    history.append([fmt.bot] + reply_ids)
    return fmt.decode(reply_ids).strip()


if args.question:
    print(answer(args.question))
else:
    print(f"Модель: {ckpt['config']}, шаг обучения {ckpt.get('iter')}. /new — новый диалог, /exit — выход")
    while True:
        try:
            q = input("\nТы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q == "/exit":
            break
        if q == "/new":
            history.clear()
            print("(диалог забыт)")
            continue
        if q:
            print("Бот:", answer(q))
