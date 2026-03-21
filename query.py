import os

from dotenv import load_dotenv
from openai import OpenAI

import chromadb

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_DIR = "docs"

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_DIR)


def get_embedding(text: str):
    resp = client_openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


def retrieve(query: str, k: int = 3):
    """
    Retrieves k nearest neighbors embedding vector representing the input text
    :param query:
    :param k:
    :return:
    """
    query_embedding = get_embedding(query)

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

def answer(question: str):
    results = retrieve(question, k=3)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_blocks = []
    for doc, meta in zip(docs, metas):
        context_blocks.append(f"[Sursa: {meta['source']}, chunk:  {meta['chunk']}]\n{doc}\n\n\n\n\n")
        #print(f"[new: {context_blocks}")
        #print("\n")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
Response STRICTLY based on the following context.
If the answer does not exist in the context, just ay "coudn't find the answer."

Context:
{context}

Question:
{question}
"""
    response = client_openai.responses.create(
        input=prompt,
        model= "gpt-5"
    )

    print("\n=== CONTEXT ===\n")
    print(context)
    print("\n=== Response ===\n")
    print(response.output_text)

if __name__ == "__main__":
    while True:
        q = input("Give me a question (0 for exit): ")

        if q == "0":
            break

        answer(q)




