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
