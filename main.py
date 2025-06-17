import os
from dotenv import load_dotenv
import logging

from src.vector_database_process import build_or_load_vector_store
from src.telegram.telegram_bot import run_telegram_bot
from src.frontend.web_app import run_web

def main():
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)
    logging.info("🚀 Загрузка проекта...")


    # === Запуск Telegram-бота ===
    logging.info("🤖 Запуск Telegram-бота...")
    run_telegram_bot()
    run_web()


if __name__ == "__main__":
    main()
