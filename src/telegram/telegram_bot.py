import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from src.graph_model import graph

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот для работы с Гражданским кодексом РФ.\n"
        "Задай мне любой вопрос — и я постараюсь найти на него ответ в законе."
    )

user_states = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_input = update.message.text

    try:
        thread_id = str(user_id)
        messages = [{"role": "user", "content": user_input}]
        response_text = None

        async for step in graph.astream(
            {"messages": messages},
            stream_mode="values",
            config={"configurable": {"thread_id": thread_id}},
        ):
            last_message = step.get("messages", [])[-1]
            if last_message:
                response_text = last_message.content

        if response_text:
            await update.message.reply_text(response_text)
        else:
            await update.message.reply_text("Извините, не удалось получить ответ. Попробуйте ещё раз.")

    except Exception as e:
        logging.exception("Ошибка при обработке сообщения:")
        await update.message.reply_text("Произошла ошибка. Попробуй позже.")


def run_telegram_bot():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("Бот запущен.")
    application.run_polling()

def main():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("Бот запущен.")
    application.run_polling()

if __name__ == "__main__":
    main()
