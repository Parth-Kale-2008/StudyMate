import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

documents = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        print(f"Loading: {file}")

        loader = PyPDFLoader(
            os.path.join("data", file)
        )

        documents.extend(loader.load())

print(f"\nTotal Pages Loaded: {len(documents)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks Created: {len(chunks)}")

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(
    chunks,
    embeddings
)


db.save_local("faiss_index")

print("\n FAISS Database Created Successfully!")
print(" Saved in: faiss_index/")