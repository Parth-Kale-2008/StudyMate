from dotenv import load_dotenv

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

load_dotenv()

embeddings = OpenAIEmbeddings()

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(
    temperature=0.5
)

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    docs = db.similarity_search(
        question,
        k=4
    )

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

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

    print("\n")
    print(response.content)
    