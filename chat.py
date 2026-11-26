"""Чат с обученной моделью. Модель помнит предыдущие реплики диалога
(сколько влезает в её окно контекста).

  python chat.py                       # интерактивный чат
  python chat.py -q "Что такое луна?"  # один вопрос
Команды в чате: /new — забыть диалог, /exit — выйти.

Из Python / Colab:
  from chat import ChatBot
  bot = ChatBot()
  print(bot.answer("Привет!"))
"""
import argparse

import torch

from model import load_model
from tokenizer_utils import ChatFormat, load_tokenizer


class ChatBot:
    def __init__(self, model_path="model/model.pt", temperature=0.7, top_k=40, max_new_tokens=150):
        self.model, self.ckpt = load_model(model_path)
        self.model.eval()
        self.fmt = ChatFormat(load_tokenizer())
        self.temperature, self.top_k, self.max_new_tokens = temperature, top_k, max_new_tokens
        self.budget = self.model.cfg.block_size - max_new_tokens  # место под историю
        self.history = []  # закодированные реплики — «память» диалога

    def reset(self):
        self.history.clear()

    def answer(self, question):
        fmt = self.fmt
        self.history.append(fmt.encode_message("user", question))
        # убираем самые старые реплики, если не влезают в окно контекста
        while len(self.history) > 1 and sum(map(len, self.history)) + 1 > self.budget:
            self.history.pop(0)
        ids = [t for turn in self.history for t in turn][-self.budget + 1:] + [fmt.bot]
        out = self.model.generate(torch.tensor([ids]), self.max_new_tokens, temperature=self.temperature,
                                  top_k=self.top_k, stop_token=fmt.end)
        reply_ids = out[0, len(ids):].tolist()
        if not reply_ids or reply_ids[-1] != fmt.end:
            reply_ids.append(fmt.end)
        self.history.append([fmt.bot] + reply_ids)
        return fmt.decode(reply_ids).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model/model.pt", help="или checkpoints/full.pt во время обучения")
    p.add_argument("-q", "--question")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--max_new_tokens", type=int, default=150)
    args = p.parse_args()

    bot = ChatBot(args.model, args.temperature, args.top_k, args.max_new_tokens)
    if args.question:
        print(bot.answer(args.question))
        return
    print(f"Модель: {bot.ckpt['config']}, шаг обучения {bot.ckpt.get('iter')}. /new — новый диалог, /exit — выход")
    while True:
        try:
            q = input("\nТы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q == "/exit":
            break
        if q == "/new":
            bot.reset()
            print("(диалог забыт)")
            continue
        if q:
            print("Бот:", bot.answer(q))


if __name__ == "__main__":
    main()
