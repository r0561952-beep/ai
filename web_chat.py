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

TITLE = "Наша нейросеть"
DESCRIPTION = (
    "Маленькая русскоязычная модель, обученная с нуля. "
    "Пишет по-русски, но часто отвечает невпопад — ей ещё учиться и учиться 🙂"
)
EXAMPLES = ["Как приготовить блины?", "Привет! Как дела?", "Дай совет, как лучше учиться.", "Что такое солнце?"]
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

    def respond(message, history, temperature, max_tokens):
        bot.temperature, bot.max_new_tokens = temperature, int(max_tokens)
        bot.budget = bot.model.cfg.block_size - bot.max_new_tokens
        messages = [{"role": m["role"], "content": _text(m["content"])} for m in history]
        messages.append({"role": "user", "content": _text(message)})
        text = ""
        for text in bot.stream(messages):
            yield text
        if not text:
            yield "…"

    return gr.ChatInterface(
        respond,
        title=TITLE,
        description=DESCRIPTION,
        examples=[[e] for e in EXAMPLES],
        chatbot=gr.Chatbot(placeholder="### Спроси что-нибудь 👋", height="70vh", show_label=False),
        textbox=gr.Textbox(placeholder="Напиши сообщение…", submit_btn=True, stop_btn=True),
        additional_inputs=[
            gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Температура (больше — фантазийнее)"),
            gr.Slider(20, 200, value=150, step=10, label="Максимальная длина ответа (токенов)"),
        ],
        additional_inputs_accordion=gr.Accordion("⚙️ Настройки", open=False),
        save_history=True,   # список прошлых чатов, как в ChatGPT (хранится в браузере)
        fill_height=True,
        autofocus=False,
    )


def launch(share=False, port=None, bot=None):
    """port=None — Gradio сам найдёт свободный порт (7860, 7861, ...)."""
    gr.close_all()  # если чат уже запускали в этом же Colab — закрываем старый
    demo = build_ui(bot)
    demo.launch(share=share, server_port=port, theme=gr.themes.Soft(), css=CSS)
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="публичная ссылка *.gradio.live")
    p.add_argument("--port", type=int, default=None)
    a = p.parse_args()
    launch(share=a.share, port=a.port)
