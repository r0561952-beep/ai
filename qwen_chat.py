"""Чат с Qwen2.5, дообученной на ролевых данных (адаптер LoRA из qwen_train.py).

Нужна видеокарта (на процессоре 1,5 млрд параметров отвечают очень медленно).

  python qwen_chat.py --adapter /content/drive/MyDrive/ai_qwen/adapter
  python web_chat.py --qwen /content/drive/MyDrive/ai_qwen/adapter --share   # веб-чат

Из Python:
  from qwen_chat import QwenBot
  bot = QwenBot("/content/drive/MyDrive/ai_qwen/adapter")
"""
import argparse
import threading

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList,
                          TextIteratorStreamer)

from rp_card import finish, make_card
from safety import REFUSAL, blocked

SYSTEM_DEFAULT = "Ты — живой собеседник в ролевой игре. Отвечай от лица своего персонажа."


class _StopFlag(StoppingCriteria):
    def __init__(self):
        self.stop = False

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop


class QwenBot:
    def __init__(self, adapter=None, base="Qwen/Qwen2.5-1.5B", temperature=0.6, top_k=40,
                 max_new_tokens=200, max_context=2048):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(base)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(self.device)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()  # вшиваем LoRA — быстрее
        self.model = model.eval()
        self.temperature, self.top_k, self.max_new_tokens = temperature, top_k, max_new_tokens
        self.max_context = max_context
        self.stop_ids = [self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        self.history = []
        self.ckpt = {"iter": "Qwen" + (" + ролевой адаптер" if adapter else "")}
        self.label = base.rsplit("/", 1)[-1] + (" + наш ролевой адаптер" if adapter else " (БЕЗ ролевого дообучения)")

    def configure(self, temperature, max_new_tokens):
        self.temperature, self.max_new_tokens = temperature, int(max_new_tokens)

    def reset(self):
        self.history.clear()

    def build_prompt(self, messages, card=None):
        """ChatML как при обучении: карточка -> system. Старые реплики выкидываются,
        если не влезают в окно; карточка остаётся всегда."""
        head = f"<|im_start|>system\n{card or SYSTEM_DEFAULT}<|im_end|>\n"
        turns = [f"<|im_start|>{'user' if m['role'] == 'user' else 'assistant'}\n{m['content']}<|im_end|>\n"
                 for m in messages]
        budget = self.max_context - self.max_new_tokens - len(self.tok(head).input_ids)
        while len(turns) > 1 and len(self.tok("".join(turns)).input_ids) > budget:
            turns.pop(0)
        return head + "".join(turns) + "<|im_start|>assistant\n"

    def stream(self, messages, card=None):
        context = "\n".join([card or ""] + [m["content"] for m in messages])
        if blocked(context):
            yield REFUSAL
            return
        inputs = self.tok(self.build_prompt(messages, card), return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        flag = _StopFlag()
        kwargs = dict(**inputs, streamer=streamer, max_new_tokens=self.max_new_tokens, do_sample=True,
                      temperature=max(self.temperature, 1e-3), top_k=self.top_k, top_p=0.95,
                      repetition_penalty=1.1, eos_token_id=self.stop_ids, pad_token_id=self.stop_ids[-1],
                      stopping_criteria=StoppingCriteriaList([flag]))
        result = {}
        worker = threading.Thread(target=lambda: result.update(out=self.model.generate(**kwargs)), daemon=True)
        worker.start()
        text, pieces = "", 0
        for piece in streamer:
            text += piece
            pieces += 1
            if pieces % 8 == 0 and blocked(context, text):
                flag.stop = True
                yield REFUSAL
                return
            yield text.strip()
        worker.join()
        out = result.get("out")
        if out is not None and out.shape[1] - inputs["input_ids"].shape[1] >= self.max_new_tokens \
                and out[0, -1].item() not in self.stop_ids:
            yield finish(text)  # упёрлись в лимит длины — не обрываем ответ на полуслове

    def answer(self, question, card=None):
        self.history.append({"role": "user", "content": question})
        text = ""
        for text in self.stream(self.history, card):
            pass
        self.history.append({"role": "assistant", "content": text})
        return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", help="папка с адаптером LoRA (из qwen_train.py)")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("-q", "--question")
    p.add_argument("--name")
    p.add_argument("--persona")
    p.add_argument("--scenario")
    a = p.parse_args()
    card = make_card(a.name or "", a.persona or "", a.scenario or "") if (a.name or a.persona or a.scenario) else None
    bot = QwenBot(a.adapter, a.base)
    if a.question:
        print(bot.answer(a.question, card))
        return
    while True:
        try:
            q = input("\nТы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q == "/exit":
            break
        if q:
            print("Бот:", bot.answer(q, card))


if __name__ == "__main__":
    main()
