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
        self.history = []  # реплики диалога — «память» (для answer)

    def reset(self):
        self.history.clear()

    def build_prompt(self, messages):
        """messages: [{"role": "user"/"assistant", "content": "..."}], последняя — вопрос.
        Самые старые реплики выкидываются, если не влезают в окно контекста."""
        turns = [self.fmt.encode_message("user" if m["role"] == "user" else "bot", m["content"])
                 for m in messages]
        while len(turns) > 1 and sum(map(len, turns)) + 1 > self.budget:
            turns.pop(0)
        return [t for turn in turns for t in turn][-self.budget + 1:] + [self.fmt.bot]

    def stream(self, messages):
        """Выдаёт растущий текст ответа по мере генерации."""
        ids = self.build_prompt(messages)
        reply = []
        for t in self.model.generate_stream(torch.tensor([ids]), self.max_new_tokens,
                                            temperature=self.temperature, top_k=self.top_k,
                                            stop_token=self.fmt.end):
            if t == self.fmt.end:
                break
            reply.append(t)
            # недособранная буква (байты UTF-8) декодируется как «�» — не показываем её
            yield self.fmt.decode(reply).rstrip("�").strip()

    def answer(self, question):
        self.history.append({"role": "user", "content": question})
        text = ""
        for text in self.stream(self.history):
            pass
        self.history.append({"role": "assistant", "content": text})
        return text


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
