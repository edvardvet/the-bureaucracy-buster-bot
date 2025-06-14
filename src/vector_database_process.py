import os
import re
import fitz
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()


PDF_PATH = "data/civil_code.pdf"
VECTOR_STORE_PATH = "src/vector_store.faiss"


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("PROXY_URL"),
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    full_text = []
    for page in doc:
        full_text.append(page.get_text())
    return "\n".join(full_text)


def split_into_articles(text: str) -> list[dict]:
    pattern = r"(Статья \d+\..*?)(?=Статья \d+\.|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    articles = []
    for i, match in enumerate(matches, start=1):
        title_line = match.split("\n")[0]
        articles.append({
            "title": title_line.strip(),
            "content": match.strip(),
            "article_id": i
        })
    return articles


def create_documents(articles: list[dict]) -> list[Document]:
    documents = []
    for article in articles:
        chunks = text_splitter.split_text(article["content"])
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": "pdf",
                    "title": article["title"],
                    "article_id": article["article_id"],
                    "chunk_id": i + 1,
                }
            )
            documents.append(doc)
    return documents


def build_or_load_vector_store(docs: list[Document], embeddings, path: str):
    if os.path.exists(path):
        print("Загрузка векторной базы данных из файла...")
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Создание новой векторной базы данных...")
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(path)
        return vector_store

full_text = extract_text_from_pdf(PDF_PATH)

articles = split_into_articles(full_text)

documents = create_documents(articles)

vector_store = build_or_load_vector_store(documents, embeddings, VECTOR_STORE_PATH)


