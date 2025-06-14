
![Logo](./src/images/BBB.png) 

# 🏛️ Bureaucracy Buster Bot

AI-помощник, способный отвечать на вопросы, связанные с Гражданским кодексом РФ. Проект использует векторный поиск по PDF-документу и Telegram-интерфейс для общения.

## 🚀 Возможности

- 📄 Парсит и индексирует PDF-документы (Гражданский кодекс РФ).
- 🔍 Находит релевантные статьи через FAISS и OpenAI embeddings.
- 🧠 Использует GPT-4 для генерации ответов на основе найденного контекста.
- 🤖 Работает через Telegram-бота.

## 🛠 Установка
git clone https://github.com/edvardvet/the-bureaucracy-buster-bot.git
poetry install

## env

TELEGRAM_TOKEN=
OPENAI_API_KEY=
PROXY_URL=
LANGCHAIN_API_KEY=

## ⚙️ Использование 
Запусти Telegram-бота:

poetry run python main.py

## 🧠 Используемые технологии
1.LangChain, LangGraph, FAISS
2.PyMuPDF (для извлечения текста из PDF)
3.OpenAI GPT-4 + text-embedding-3-small
4.Telegram Bot API (python-telegram-bot)
5.Poetry — управление зависимостями

## 📌 Цели проекта
Автоматизация юридических справок по законодательству РФ.

Практика в vector search, RAG, LangGraph, LLM tool usage.

Портфолио-проект с реальной ценностью.