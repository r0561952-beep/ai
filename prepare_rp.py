"""Собирает русские ролевые диалоги для обучения модели свободной ролевой игре.

  python prepare_rp.py           # обычные (SFW) + 18+ данные
  python prepare_rp.py --no_18   # только обычные

Результат (в репозиторий НЕ коммитится):
  data/rp_ru.jsonl.gz    — обычные ролевые диалоги
  data/rp18_ru.jsonl.gz  — 18+ ролевые диалоги

Формат как у остальных данных: {"messages": [...]}. Первое сообщение — карточка
персонажа («[Ролевая игра] Персонаж: … Описание: …»), дальше реплики. Длинные сцены
режутся на куски, и карточка повторяется в начале каждого куска — модель всегда видит,
кого играет.

Безопасность: из 18+ данных выбрасываются целиком разделы с несовершеннолетними и
насилием, а также ЛЮБОЙ диалог с признаками несовершеннолетнего (см. safety.py).

Источники:
  IlyaGusev/gpt_roleplay_realm (CC-BY-4.0), cyttic/toloka-persona-chat-ru-calls,
  Arketov/hieunguyenminh_roleplay-deduped-ShareGPT_ru,
  18+: Arketov/ru_roleplay_conversation, Arketov/ru_roleplay_conversation_chub
"""
import argparse
import gzip
import json
import os
import random
import re
import urllib.request

import pyarrow.parquet as pq

from rp_card import cut, make_card
from safety import mentions_minor

RAW = "data/raw/rp"
HF = "https://huggingface.co/datasets/"
FILES = {
    "realm_ru.parquet": HF + "IlyaGusev/gpt_roleplay_realm/resolve/main/data/ru-00000-of-00001-34293b78fa99b4f0.parquet",
    "toloka_train.jsonl": HF + "cyttic/toloka-persona-chat-ru-calls/resolve/main/train.jsonl",
    "sharegpt_rp_ru.parquet": HF + "Arketov/hieunguyenminh_roleplay-deduped-ShareGPT_ru/resolve/main/data/train-00000-of-00001.parquet",
}
FILES_18 = {
    "ru_rp_conv.jsonl": HF + "Arketov/ru_roleplay_conversation/resolve/main/dataset_ru_rp.jsonl",
    "chub_0.parquet": HF + "Arketov/ru_roleplay_conversation_chub/resolve/main/data/train-00000-of-00002.parquet",
    "chub_1.parquet": HF + "Arketov/ru_roleplay_conversation_chub/resolve/main/data/train-00001-of-00002.parquet",
}
# разделы форумов, которые выбрасываются целиком: несовершеннолетние, пытки, насилие без согласия
BAD_TAGS = {"lolicit", "allthefallen-roleplaying", "allthefallen", "taboo-original-roleplays", "redlightponyville",
            "torture-chamber", "extreme-solos", "noncon-exotic-freeform-solos", "noncon-human-freeform-solos"}

SEGMENT_CHARS = 800   # сколько символов реплик в одном куске (карточка + куск ≈ 512 токенов)
TURN_CHARS = 700      # длинные реплики обрезаются по границе предложения


def fetch(files):
    os.makedirs(RAW, exist_ok=True)
    for name, url in files.items():
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            print(f"скачиваю {name} ...")
            urllib.request.urlretrieve(url, path)


def clean(text, char="", user=""):
    text = (text or "").replace("\r", "")
    text = re.sub(r"\{\{\s*char\s*\}\}|<BOT>", char or "Персонаж", text, flags=re.I)
    text = re.sub(r"\{\{\s*user\s*\}\}|<USER>", user or "Собеседник", text, flags=re.I)
    text = re.sub(r"<START>|<\|[^>]*\|>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def segments(card, turns):
    """Режет диалог на куски, каждый начинается с карточки персонажа."""
    out, cur, size = [], [], 0
    for role, text in turns:
        text = cut(text, TURN_CHARS)
        if not text:
            continue
        if cur and size + len(text) > SEGMENT_CHARS:
            out.append(cur)
            cur, size = [], 0
        cur.append({"role": role, "content": text})
        size += len(text)
    if cur:
        out.append(cur)
    # кусок без единой реплики персонажа бесполезен
    return [[{"role": "user", "content": card}] + s for s in out if any(m["role"] == "assistant" for m in s)]


def from_realm():
    t = pq.read_table(os.path.join(RAW, "realm_ru.parquet"), columns=["name", "context", "greeting", "dialogues"])
    for r in t.to_pylist():
        card = make_card(r["name"], clean(r["context"]))
        for d in r["dialogues"]:
            turns = [("assistant", clean(r["greeting"]))] if r.get("greeting") else []
            turns += [("user" if m["role"] == "user" else "assistant", clean(m["content"])) for m in d["chat"]]
            yield from segments(card, turns)


def from_toloka():
    for line in open(os.path.join(RAW, "toloka_train.jsonl"), encoding="utf-8"):
        msgs = json.loads(line)["messages"]
        persona = msgs[0]["content"].split("О тебе:", 1)[-1].strip() if msgs[0]["role"] == "system" else ""
        card = make_card("обычный человек, болтаешь по телефону", "Живой человек, не ассистент. " + persona)
        turns = [(m["role"], clean(m["content"])) for m in msgs if m["role"] in ("user", "assistant")]
        yield from segments(card, turns)


def from_sharegpt():
    for r in pq.read_table(os.path.join(RAW, "sharegpt_rp_ru.parquet")).to_pylist():
        conv = r["conversation"]
        system = [clean(m["content"]) for m in conv if m["role"] == "system"]
        card = make_card("", system[0] if system else "", system[1] if len(system) > 1 else "")
        turns = [(m["role"], clean(m["content"])) for m in conv if m["role"] in ("user", "assistant")]
        yield from segments(card, turns)


def safe_segments(card, turns, names, stats):
    """Режет на куски и ещё раз проверяет УЖЕ очищенный текст (вместе с именами):
    при любом признаке несовершеннолетнего выбрасывается весь диалог."""
    segs = segments(card, turns)
    text = "\n".join([names] + [m["content"] for seg in segs for m in seg])
    if mentions_minor(text, strict=True):
        stats["признак несовершеннолетнего"] += 1
        return []
    stats["взято"] += 1
    return segs


def from_rp18(stats):
    for line in open(os.path.join(RAW, "ru_rp_conv.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if set(r.get("tags") or []) & BAD_TAGS:
            stats["раздел исключён"] += 1
            continue
        everything = "\n".join([r.get("bot") or "", r.get("user") or "", r.get("bot_persona") or "",
                                r.get("user_persona") or "", r.get("scenario") or ""] + [m["content"] for m in r["conv"]])
        if mentions_minor(everything, strict=True):
            stats["признак несовершеннолетнего"] += 1
            continue
        char, user = r.get("bot") or "", r.get("user") or ""
        persona = re.sub(r"^Профиль персонажа:\s*", "", clean(r.get("bot_persona"), char, user))
        card = make_card(char, persona, clean(r.get("scenario"), char, user),
                         re.sub(r"^Профиль персонажа:\s*", "", clean(r.get("user_persona"), char, user)))
        turns = [("assistant" if m["role"] == "bot" else "user", clean(m["content"], char, user)) for m in r["conv"]]
        yield from safe_segments(card, turns, f"{char}\n{user}", stats)
    for f in ("chub_0.parquet", "chub_1.parquet"):
        for r in pq.read_table(os.path.join(RAW, f)).to_pylist():
            char, user = r.get("bot_name") or "", r.get("user_name") or ""
            fields = [char, user] + [r.get(k) or "" for k in ("personality_ru", "scenario_ru", "bot_description_ru",
                                                "user_description_ru", "example_dialogs_ru", "tavern_personality_ru")]
            conv = r.get("conversation_ru") or []
            if mentions_minor("\n".join(fields + [m["content"] for m in conv]), strict=True):
                stats["признак несовершеннолетнего"] += 1
                continue
            persona = clean(r.get("personality_ru") or r.get("bot_description_ru"), char, user)
            card = make_card(char, persona, clean(r.get("scenario_ru"), char, user),
                             clean(r.get("user_description_ru"), char, user))
            turns = [("assistant" if m["is_bot"] else "user", clean(m["content"], char, user)) for m in conv]
            yield from safe_segments(card, turns, f"{char}\n{user}", stats)


def save(path, dialogs):
    random.Random(0).shuffle(dialogs)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for d in dialogs:
            f.write(json.dumps({"messages": d}, ensure_ascii=False) + "\n")
    chars = sum(len(m["content"]) for d in dialogs for m in d)
    print(f"  {path}: {len(dialogs)} кусков, {chars / 1e6:.0f} млн символов, {os.path.getsize(path) / 1e6:.0f} МБ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_18", action="store_true", help="не скачивать и не использовать 18+ данные")
    a = ap.parse_args()
    fetch(FILES)
    sfw = list(from_realm()) + list(from_toloka()) + list(from_sharegpt())
    print("обычные ролевые диалоги:")
    save("data/rp_ru.jsonl.gz", sfw)
    if not a.no_18:
        fetch(FILES_18)
        stats = {"взято": 0, "раздел исключён": 0, "признак несовершеннолетнего": 0}
        rp18 = list(from_rp18(stats))
        print(f"18+ диалоги: {stats}")
        save("data/rp18_ru.jsonl.gz", rp18)


if __name__ == "__main__":
    main()
