"""Скачивает открытые русские датасеты с вопросами/ответами и собирает
их в один файл data/qa_ru.jsonl.gz.

Формат одной строки: {"messages": [{"role": "user", "content": "..."},
                                   {"role": "bot",  "content": "..."}, ...]}

Источники (лицензия CC-BY-4.0, автор — Илья Гусев):
  - IlyaGusev/ru_turbo_alpaca  — ~30 тыс. пар «вопрос/задание → ответ»
  - IlyaGusev/ru_turbo_saiga   — ~37 тыс. многоходовых диалогов

Запуск:  python prepare_data.py
"""
import gzip
import io
import json
import os
import random
import urllib.request

import zstandard

RAW_DIR = "data/raw"
OUT = "data/qa_ru.jsonl.gz"
SOURCES = {
    "ru_turbo_alpaca": "https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca/resolve/main/ru_turbo_alpaca.jsonl.zst",
    "ru_turbo_saiga": "https://huggingface.co/datasets/IlyaGusev/ru_turbo_saiga/resolve/main/ru_turbo_saiga.jsonl.zst",
}


def download(name, url):
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, name + ".jsonl.zst")
    if not os.path.exists(path):
        print(f"скачиваю {name} ...")
        urllib.request.urlretrieve(url, path)
    return path


def read_zst_jsonl(path):
    with open(path, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as r:
        for line in io.TextIOWrapper(r, encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)


def clean(text):
    return (text or "").replace("\r\n", "\n").strip()


def from_alpaca(path):
    for row in read_zst_jsonl(path):
        question = clean(row["instruction"])
        if clean(row.get("input")):
            question += "\n\n" + clean(row["input"])
        answer = clean(row["output"])
        # разметчики пометили ответ как плохой -> берём альтернативный
        if row.get("label") == "bad_output":
            answer = clean(row.get("alternative_output"))
        if question and answer:
            yield [{"role": "user", "content": question}, {"role": "bot", "content": answer}]


def from_saiga(path):
    for row in read_zst_jsonl(path):
        msgs = [
            {"role": m["role"], "content": clean(m["content"])}
            for m in row.get("messages", [])
            if m.get("role") in ("user", "bot") and clean(m.get("content"))
        ]
        # диалог должен начинаться с пользователя и чередоваться
        ok = len(msgs) >= 2 and all(
            m["role"] == ("user" if i % 2 == 0 else "bot") for i, m in enumerate(msgs)
        )
        if ok:
            if msgs[-1]["role"] == "user":
                msgs = msgs[:-1]
            yield msgs


def main():
    dialogs = []
    dialogs += list(from_alpaca(download("ru_turbo_alpaca", SOURCES["ru_turbo_alpaca"])))
    n_alpaca = len(dialogs)
    dialogs += list(from_saiga(download("ru_turbo_saiga", SOURCES["ru_turbo_saiga"])))
    print(f"alpaca: {n_alpaca}, saiga: {len(dialogs) - n_alpaca}, всего: {len(dialogs)}")
    random.Random(0).shuffle(dialogs)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for d in dialogs:
            f.write(json.dumps({"messages": d}, ensure_ascii=False) + "\n")
    print(f"сохранено в {OUT} ({os.path.getsize(OUT) / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
