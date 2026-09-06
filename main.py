import os
import fitz  # PyMuPDF
import httpx
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

app = FastAPI(title="AI Professor 24/7 Cloud Bot", version="2.0.0")

INDEX_DIR = "faiss_data"
DEFAULT_INDEX = "." if os.path.exists("index.faiss") else "faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

class QueryPayload(BaseModel):
    user_id: str
    question: str


# --- TELEGRAM SENDER ---
async def send_tg_message(chat_id: int | str, text: str):
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(url, json={"chat_id": chat_id, "text": text})


def process_and_index_pdf(user_id: str, content: bytes):
    doc = fitz.open(stream=content, filetype="pdf")
    pages_to_read = min(len(doc), 40)
    text_list = []
    for i in range(pages_to_read):
        page_text = doc[i].get_text().strip()
        if page_text:
            text_list.append(page_text)

    full_text = "\n\n".join(text_list)
    if not full_text:
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.create_documents([full_text])

    user_index_path = os.path.join(INDEX_DIR, str(user_id))
    if os.path.exists(user_index_path):
        db = FAISS.load_local(user_index_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(user_index_path)
    return len(chunks)


def query_rag(user_id: str, question: str) -> str:
    user_index_path = os.path.join(INDEX_DIR, str(user_id))
    if os.path.exists(user_index_path):
        db = FAISS.load_local(user_index_path, embeddings, allow_dangerous_deserialization=True)
    elif os.path.exists(DEFAULT_INDEX):
        db = FAISS.load_local(DEFAULT_INDEX, embeddings, allow_dangerous_deserialization=True)
    else:
        return "I could not find any notes. Please upload a PDF first!"

    docs = db.similarity_search(question, k=4)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an expert AI professor.

Use ONLY the provided context.

Do not copy the notes directly.

Explain concepts:
- Simple and in easy explainable manner
- Step by Step
- With real life examples
- With analogies
- Explain in detail
- Also give the definations
Context:
{context}

Question:
{question}

If answer is not found,
say:
'I could not find this topic in the uploaded notes.'
"""
    response = llm.invoke(prompt)
    return response.content


# --- API ENDPOINTS ---
@app.get("/")
def read_root():
    return {"status": "AI Professor 24/7 Cloud Bot is Active"}


# --- 24/7 TELEGRAM WEBHOOK ---
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    try:
        update = await request.json()
    except Exception:
        return {"ok": True}

    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    if not chat_id:
        return {"ok": True}

    text = (message.get("text") or message.get("caption") or "").strip()
    document = message.get("document")

    # 1. /start
    if text.startswith("/start"):
        welcome_text = (
            "👋 Welcome to the 24/7 AI Professor Bot!\n\n"
            "1️⃣ Send or forward me any PDF.\n"
            "2️⃣ Wait a few seconds for indexing.\n"
            "3️⃣ Ask any questions about your notes!\n\n"
            "⚡ Running 24/7 permanently in the cloud."
        )
        await send_tg_message(chat_id, welcome_text)
        return {"ok": True}

    # 2. PDF Document Upload
    if document:
        file_id = document.get("file_id")
        file_name = document.get("file_name", "document.pdf")

        if not file_name.lower().endswith(".pdf"):
            await send_tg_message(chat_id, "⚠️ Please upload a PDF file.")
            return {"ok": True}

        await send_tg_message(chat_id, "⏳ Reading and indexing your document in the cloud...")

        file_info_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(file_info_url)
            file_path = resp.json().get("result", {}).get("file_path")
            if not file_path:
                await send_tg_message(chat_id, "❌ Failed to download file from Telegram.")
                return {"ok": True}

            download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
            file_resp = await client.get(download_url)
            content = file_resp.content

        chunks = process_and_index_pdf(str(chat_id), content)
        if chunks > 0:
            await send_tg_message(chat_id, f"📄 Successfully indexed {chunks} chunks! You can now ask questions about your document.")
        else:
            await send_tg_message(chat_id, "⚠️ Could not extract text from this PDF.")
        return {"ok": True}

    # 3. Questions
    if text:
        answer = query_rag(str(chat_id), text)
        await send_tg_message(chat_id, answer)
        return {"ok": True}

    return {"ok": True}
