import os
import tempfile
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

app = FastAPI(title="AI Professor Telegram RAG", version="1.0.0")

INDEX_DIR = "faiss_data"
DEFAULT_INDEX = "." if os.path.exists("index.faiss") else "faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

class QueryPayload(BaseModel):
    user_id: str
    question: str


@app.get("/")
def read_root():
    return {"status": "AI Professor Backend is Running"}


@app.post("/upload")
async def upload_pdf(user_id: str = Form(...), file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        os.remove(tmp_path)

        if not documents:
            raise HTTPException(status_code=400, detail="No readable text found in PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        user_index_path = os.path.join(INDEX_DIR, user_id)
        if os.path.exists(user_index_path):
            db = FAISS.load_local(user_index_path, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(chunks)
        else:
            db = FAISS.from_documents(chunks, embeddings)

        db.save_local(user_index_path)
        return {"status": "success", "chunks_indexed": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(payload: QueryPayload):
    user_index_path = os.path.join(INDEX_DIR, payload.user_id)

    if os.path.exists(user_index_path):
        db = FAISS.load_local(user_index_path, embeddings, allow_dangerous_deserialization=True)
    elif os.path.exists(DEFAULT_INDEX):
        db = FAISS.load_local(DEFAULT_INDEX, embeddings, allow_dangerous_deserialization=True)
    else:
        return {"answer": "I could not find any notes. Please upload a PDF first!"}

    try:
        docs = db.similarity_search(payload.question, k=4)
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
{payload.question}

If answer is not found,
say:
'I could not find this topic in the uploaded notes.'
"""

        response = llm.invoke(prompt)
        return {"answer": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
