"""Чат с обученной моделью. Модель помнит предыдущие реплики диалога
(сколько влезает в её окно контекста).

  python chat.py                       # интерактивный чат
  python chat.py -q "Что такое луна?"  # один вопрос
  python chat.py --name "Лира" --persona "Насмешливая эльфийка-следопыт" --scenario "Таверна, вечер"
                                       # ролевая игра с персонажем
Команды в чате: /new — забыть диалог, /exit — выйти.

Из Python / Colab:
  from chat import ChatBot
  bot = ChatBot()
  print(bot.answer("Привет!"))
"""
import argparse

import torch

from model import load_model
from rp_card import make_card
from safety import REFUSAL, blocked
from tokenizer_utils import ChatFormat, load_tokenizer


class ChatBot:
    def __init__(self, model_path="model/model.pt", temperature=0.7, top_k=40, max_new_tokens=150):
        self.model, self.ckpt = load_model(model_path)
        self.model.eval()
        self.fmt = ChatFormat(load_tokenizer())
        self.temperature, self.top_k, self.max_new_tokens = temperature, top_k, max_new_tokens
        self.budget = self.model.cfg.block_size - max_new_tokens  # место под историю
        self.history = []  # реплики диалога — «память» (для answer)
        n = self.model.num_params() / 1e6
        self.label = f"своя модель {model_path.rsplit('/', 1)[-1]} ({n:.0f} млн параметров)"

    def configure(self, temperature, max_new_tokens):
        self.temperature, self.max_new_tokens = temperature, int(max_new_tokens)
        self.budget = self.model.cfg.block_size - self.max_new_tokens

    def reset(self):
        self.history.clear()

    def build_prompt(self, messages, card=None):
        """messages: [{"role": "user"/"assistant", "content": "..."}], последняя — вопрос.
        card — карточка персонажа: всегда остаётся в начале, а самые старые реплики
        выкидываются, если не влезают в окно контекста."""
        pinned = self.fmt.encode_message("user", card) if card else []
        turns = [self.fmt.encode_message("user" if m["role"] == "user" else "bot", m["content"])
                 for m in messages]
        while len(turns) > 1 and len(pinned) + sum(map(len, turns)) + 1 > self.budget:
            turns.pop(0)
        body = [t for turn in turns for t in turn][-max(1, self.budget - len(pinned) - 1):]
        return pinned + body + [self.fmt.bot]

    def stream(self, messages, card=None):
        """Выдаёт растущий текст ответа по мере генерации."""
        context = "\n".join([card or ""] + [m["content"] for m in messages])
        if blocked(context):
            yield REFUSAL
            return
        ids = self.build_prompt(messages, card)
        reply = []
        for t in self.model.generate_stream(torch.tensor([ids]), self.max_new_tokens,
                                            temperature=self.temperature, top_k=self.top_k,
                                            stop_token=self.fmt.end):
            if t == self.fmt.end:
                break
            reply.append(t)
            # недособранная буква (байты UTF-8) декодируется как «�» — не показываем её
            text = self.fmt.decode(reply).rstrip("�").strip()
            if len(reply) % 8 == 0 and blocked(context, text):
                yield REFUSAL
                return
            yield text

    def answer(self, question, card=None):
        self.history.append({"role": "user", "content": question})
        text = ""
        for text in self.stream(self.history, card):
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
    p.add_argument("--name", help="ролевая игра: имя персонажа")
    p.add_argument("--persona", help="ролевая игра: описание персонажа")
    p.add_argument("--scenario", help="ролевая игра: сценарий / место действия")
    args = p.parse_args()
    card = make_card(args.name or "", args.persona or "", args.scenario or "") \
        if (args.name or args.persona or args.scenario) else None

    bot = ChatBot(args.model, args.temperature, args.top_k, args.max_new_tokens)
    if args.question:
        print(bot.answer(args.question, card))
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
            print("Бот:", bot.answer(q, card))


if __name__ == "__main__":
    main()
