from pathlib import Path
import sys
import gradio as gr
import logging
from typing import AsyncGenerator, List, Tuple

# Настройка пути и импорта
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from src.graph_model import graph

# 🎨 CSS стили
css = """
body {
    background: linear-gradient(to right, #f7f8fa, #eef1f5);
    font-family: 'Segoe UI', sans-serif;
    margin: 0;
    color: #111;
}

h1 {
    text-align: center;
    font-size: 22px !important;
    color: #FFFACD;
    font-weight: 600;
    margin-top: 12px;
    margin-bottom: 16px;
    letter-spacing: 0.4px;
    text-shadow: 1px 1px 1px rgba(0, 0, 0, 0.05);
}

/* Окно чата */
.chatbot {
    font-size: 15px !important;
    background-color: #ffffff;
    border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    padding: 8px;
}

/* Поле ввода */
.big-textbox textarea {
    font-size: 15px !important;
    min-height: 90px !important;
    border-radius: 8px !important;
    border: 1px solid #ccc !important;
    padding: 10px !important;
    background-color: #ffffff !important;
    color: #111 !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
}

/* Кнопки */
.large-button {
    font-size: 15px !important;
    height: 38px !important;
    min-width: 110px !important;
    border-radius: 8px !important;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}
.large-button:hover {
    background-color: #3b82f6 !important;
    color: #fff !important;
    transform: scale(1.02);
}
"""

# Логирование
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Генерация ответа
async def generate_response(user_input: str, thread_id: str = "gradio_chat") -> AsyncGenerator[str, None]:
    messages = [{"role": "user", "content": user_input}]
    try:
        async for step in graph.astream(
            {"messages": messages},
            stream_mode="values",
            config={"configurable": {"thread_id": thread_id}},
        ):
            last_message = step.get("messages", [])[-1]
            if last_message:
                yield last_message.content
    except Exception as e:
        logging.exception("Ошибка при генерации ответа:")
        yield "Произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже."

# Обработка диалога
async def process_chat(user_input: str, chat_history: List[Tuple[str, str]]):
    chat_history.append((user_input, ""))
    full_response = ""
    async for chunk in generate_response(user_input):
        full_response += chunk
        chat_history[-1] = (user_input, full_response)
        yield chat_history

# Интерфейс
with gr.Blocks(title="Чат-помощник по ГК РФ", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("<h1>🧑⚖️ Онлайн-помощник по Гражданскому кодексу РФ</h1>")

    with gr.Row():
        chatbot = gr.Chatbot(
            label="Диалог",
            height=380,  # 📉 Снижено с 600 → 460 → 380
            bubble_full_width=False,
            show_label=True,
            elem_classes="chatbot"
        )

    with gr.Row():
        msg = gr.Textbox(
            label="Ваш вопрос",
            placeholder="Например: Что регулирует статья 1196 ГК РФ?",
            lines=4,
            elem_classes="big-textbox"
        )

    with gr.Row():
        submit_btn = gr.Button("📨 Отправить", variant="primary", elem_classes="large-button")
        clear_btn = gr.Button("🧹 Очистить", elem_classes="large-button")

    msg.submit(process_chat, [msg, chatbot], chatbot)
    submit_btn.click(process_chat, [msg, chatbot], chatbot)
    clear_btn.click(lambda: None, None, chatbot, queue=False)


def run_web():
    demo.launch()

    
if __name__ == "__main__":
    demo.launch()
