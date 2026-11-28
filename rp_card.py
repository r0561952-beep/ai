"""Карточка персонажа для ролевой игры — одинаковая при обучении и в чате."""

CARD_LIMITS = {"persona": 400, "scenario": 150, "user": 120}


def cut(text, limit):
    """Обрезает по границе предложения, не длиннее limit символов."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    part = text[:limit]
    m = max(part.rfind(". "), part.rfind("! "), part.rfind("? "), part.rfind("\n"), part.rfind("… "))
    return (part[:m + 1] if m > limit * 0.4 else part.rsplit(" ", 1)[0] + "…").strip()


def make_card(name="", persona="", scenario="", user=""):
    lines = ["[Ролевая игра]"]
    if name:
        lines.append(f"Персонаж: {name.strip()}")
    if persona:
        lines.append("Описание: " + cut(persona, CARD_LIMITS["persona"]))
    if scenario:
        lines.append("Сценарий: " + cut(scenario, CARD_LIMITS["scenario"]))
    if user:
        lines.append("Собеседник: " + cut(user, CARD_LIMITS["user"]))
    return "\n".join(lines)


_END = ".!?…»\")"


def finish(text):
    """Ответ упёрся в лимит длины — убираем недописанное предложение в конце."""
    t = (text or "").rstrip()
    best, stars = -1, 0
    for i, ch in enumerate(t):
        stars += ch == "*"
        if ch in _END or (ch == "*" and stars % 2 == 0):  # «*» — только закрывающая звёздочка действия
            best = i
    if best == len(t) - 1:
        return t
    t = t[:best + 1] if best >= len(t) * 0.4 else t + "…"
    # закрываем то, что осталось открытым: *действие*, "реплика", «реплика»
    if t.count("*") % 2:
        t += "*"
    if t.count('"') % 2:
        t += '"'
    if t.count("«") > t.count("»"):
        t += "»"
    return t
