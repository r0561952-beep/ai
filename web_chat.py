"""Веб-чат в стиле ChatGPT (Gradio).

  python web_chat.py           # открыть http://localhost:7860
  python web_chat.py --share   # получить публичную ссылку (например, из Colab на телефон)

Из Python / Colab:
  from web_chat import build_ui, launch
  launch(share=True)
"""
import argparse

import gradio as gr

from chat import ChatBot
from rp_card import make_card

TITLE = "Наша нейросеть"
DESCRIPTION = (
    "Русскоязычная модель, обученная с нуля. Для ролевой игры задай персонажа в «🎭 Персонаж» "
    "(ниже под чатом), действия пиши в звёздочках: *улыбается*. Оставь поля пустыми — будет обычный чат."
)
EXAMPLES = ["*стряхивает капли дождя с плаща* Привет! Можно присесть рядом?", "Расскажи о себе.",
            "*протягивает кружку эля* За знакомство!", "Как приготовить блины?"]
DEFAULT_CHARACTER = {
    "name": "Лира",
    "persona": "Насмешливая и смелая эльфийка-следопыт из Северного леса. Метко стреляет из лука, "
               "обожает загадки и не доверяет незнакомцам.",
    "scenario": "Таверна на перекрёстке дорог, вечер, за окном гроза.",
}
CSS = "footer {display: none !important}"


def _text(content):
    """Gradio может хранить сообщение строкой или списком частей — достаём текст."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    if isinstance(content, (list, tuple)):
        return " ".join(_text(c) for c in content)
    return str(content or "")


def build_ui(bot=None):
    bot = bot or ChatBot()

    def respond(message, history, name, persona, scenario, temperature, max_tokens):
        bot.configure(temperature, max_tokens)
        card = make_card(name, persona, scenario) if (name or persona or scenario).strip() else None
        messages = [{"role": m["role"], "content": _text(m["content"])} for m in history]
        messages.append({"role": "user", "content": _text(message)})
        text = ""
        for text in bot.stream(messages, card):
            yield text
        if not text:
            yield "…"

    return gr.ChatInterface(
        respond,
        title=TITLE,
        description=f"**🧠 Модель: {getattr(bot, 'label', '?')}**\n\n" + DESCRIPTION,
        examples=[[e] for e in EXAMPLES],
        chatbot=gr.Chatbot(placeholder="### Начни игру 🎭", height="65vh", show_label=False),
        textbox=gr.Textbox(placeholder="Напиши реплику… *действия — в звёздочках*", submit_btn=True, stop_btn=True),
        additional_inputs=[
            gr.Textbox(value=DEFAULT_CHARACTER["name"], label="Персонаж (имя)"),
            gr.Textbox(value=DEFAULT_CHARACTER["persona"], label="Описание: характер, внешность, манера речи", lines=3),
            gr.Textbox(value=DEFAULT_CHARACTER["scenario"], label="Сценарий: где и когда происходит действие", lines=2),
            gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Температура (больше — фантазийнее)"),
            gr.Slider(20, 250, value=150, step=10, label="Максимальная длина ответа (токенов)"),
        ],
        additional_inputs_accordion=gr.Accordion("🎭 Персонаж и настройки", open=False),
        save_history=True,   # список прошлых чатов, как в ChatGPT (хранится в браузере)
        fill_height=True,
        autofocus=False,
    )


def launch(share=False, port=None, bot=None, model="model/model.pt", qwen_adapter=None):
    """port=None — Gradio сам найдёт свободный порт (7860, 7861, ...).
    qwen_adapter — путь к адаптеру Qwen (тогда вместо нашей модели — Qwen, нужна видеокарта)."""
    gr.close_all()  # если чат уже запускали в этом же Colab — закрываем старый
    if bot is None and qwen_adapter:
        from qwen_chat import QwenBot
        bot = QwenBot(qwen_adapter)
    demo = build_ui(bot or ChatBot(model))
    demo.launch(share=share, server_port=port, theme=gr.themes.Soft(), css=CSS)
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="публичная ссылка *.gradio.live")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--model", default="model/model.pt", help="например model/model_rp.pt — ролевая модель")
    p.add_argument("--qwen", help="папка адаптера Qwen (из qwen_train.py) — чат с Qwen вместо нашей модели")
    a = p.parse_args()
    launch(share=a.share, port=a.port, model=a.model, qwen_adapter=a.qwen)
