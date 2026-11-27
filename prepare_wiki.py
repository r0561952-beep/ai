"""Превращает статьи русской Википедии в пары «вопрос → ответ» для обучения.

  python prepare_wiki.py            # 2 файла (~140 тыс. статей)
  python prepare_wiki.py --files 4  # больше статей

Результат: data/wiki_qa_ru.jsonl.gz (тот же формат, что data/qa_ru.jsonl.gz).
Источник: wikimedia/wikipedia (20231101.ru), лицензия CC BY-SA 4.0.
"""
import argparse
import gzip
import json
import os
import random
import re
import urllib.request

import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ru/train-{:05d}-of-00021.parquet"
RAW = "data/raw/wiki_ru_{}.parquet"
OUT = "data/wiki_qa_ru.jsonl.gz"

SHORT_Q = ["Что такое {t}?", "{t} — что это?", "Объясни, что такое {t}.", "Расскажи кратко: {t}."]
PERSON_Q = ["Кто это — {t}?", "Расскажи кратко об этом человеке: {t}.", "Что известно о человеке по имени {t}?"]
LONG_Q = ["Расскажи подробно про тему «{t}».", "Расскажи подробнее: {t}.", "Что ты знаешь о теме «{t}»?"]
SKIP_SECTIONS = {"Примечания", "Литература", "Ссылки", "См. также", "Источники", "Комментарии",
                 "Галерея", "Библиография", "Сочинения", "Фильмография", "Дискография", "Награды"}

STRESS = re.compile("[̀́]")
EMPTY_PARENS = re.compile(r"\(\s*[;,]?\s*\)")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z0-9«])")
PERSON = re.compile(r"^([А-ЯЁ][а-яё\-]+), ([А-ЯЁ][а-яё\-]+(?: [А-ЯЁ][а-яё\-]+)?)$")


def clean_title(title):
    t = re.sub(r"\s*\(.*?\)\s*", " ", title).strip()
    m = PERSON.match(t)
    return (f"{m.group(2)} {m.group(1)}", True) if m else (t, False)


def good_sentences(paragraph):
    """Убираем ударения, пустые скобки и предложения с пропущенными числами («Площадь —  км²»)."""
    p = EMPTY_PARENS.sub("", STRESS.sub("", paragraph))
    p = re.sub(r"\(\s*[;,]\s*", "(", p)
    out = []
    for s in SENT_SPLIT.split(p):
        if re.search(r"\S {2,}\S", s) or re.search(r"—\s+[,.;)]|\s[,;]\s*[,;.]|[:(]\s*[,;]", s):
            continue  # в предложении «дырка» на месте числа/шаблона
        s = re.sub(r"\s+([,.;:)])", r"\1", re.sub(r"\s{2,}", " ", s)).strip()
        if len(s) > 15:
            out.append(s)
    return out


def split_sections(text):
    """Возвращает [(заголовок, [абзацы])]; первый — вводная часть статьи."""
    sections, title, paras = [], "", []
    for block in (b.strip() for b in text.split("\n\n")):
        if not block:
            continue
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) == 1 and len(lines[0]) < 60 and not lines[0].endswith((".", "!", "?", ":")):
            sections.append((title, paras))
            title, paras = lines[0], []
        else:
            paras += [l for l in lines if len(l) > 40]  # короткие строки — обычно пункты списков
    sections.append((title, paras))
    return sections


def take(sentences, limit):
    text = ""
    for s in sentences:
        if len(text) + len(s) > limit:
            break
        text = f"{text} {s}".strip()
    return text


def article_to_dialogs(title, text, rng):
    t, is_person = clean_title(title)
    sections = split_sections(text)
    lead = [s for p in sections[0][1] for s in good_sentences(p)]
    short = take(lead, 900)
    # первое предложение должно быть про саму тему (иначе вводная часть «поломана»)
    keys = [w[:5].lower() for w in re.findall(r"\w{4,}", t)]
    if len(short) < 120 or (keys and not any(k in lead[0].lower() for k in keys)):
        return []
    q = rng.choice(PERSON_Q if is_person else SHORT_Q).format(t=t)
    dialogs = [[{"role": "user", "content": q}, {"role": "assistant", "content": short}]]
    body = lead + [s for name, ps in sections[1:] if name not in SKIP_SECTIONS for p in ps for s in good_sentences(p)]
    long = take(body, 2500)
    if len(long) > len(short) + 600:
        q = rng.choice(LONG_Q).format(t=t)
        dialogs.append([{"role": "user", "content": q}, {"role": "assistant", "content": long}])
    return dialogs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=2)
    ap.add_argument("--min_chars", type=int, default=1500, help="пропускать короткие статьи-заготовки")
    a = ap.parse_args()
    os.makedirs("data/raw", exist_ok=True)
    rng = random.Random(0)
    dialogs, n_articles = [], 0
    for i in range(a.files):
        path = RAW.format(i)
        if not os.path.exists(path):
            print(f"скачиваю часть {i} ...")
            urllib.request.urlretrieve(URL.format(i), path)
        table = pq.read_table(path, columns=["title", "text"])
        for title, text in zip(table.column("title").to_pylist(), table.column("text").to_pylist()):
            if len(text) < a.min_chars:
                continue
            d = article_to_dialogs(title, text, rng)
            if d:
                n_articles += 1
                dialogs += d
    rng.shuffle(dialogs)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for d in dialogs:
            f.write(json.dumps({"messages": d}, ensure_ascii=False) + "\n")
    print(f"статей: {n_articles}, пар вопрос-ответ: {len(dialogs)}, файл {OUT} "
          f"({os.path.getsize(OUT) / 1e6:.0f} МБ)")


if __name__ == "__main__":
    main()
