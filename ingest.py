import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

import chromadb

load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "docs"

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def chunk_text(text: str, chunk_size: int = 700, overlap: int =100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(text: str):
    resp = client_openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def ingest():
    """
    Reads all the documents in director and ingests them in the collection
    :return:
    """
    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for file in DATA_DIR.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for i,chunk in enumerate(chunks):
            chunk_id = f"{file.stem}_{i}"
            embedding = get_embedding(chunk)

            ids.append(chunk_id)
            documents.append(chunk)
            embeddings.append(embedding)
            metadatas.append({"source": file.name, "chunk": i})

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"ingested {len(ids)} chunks")
    else:
        print(f"no .txt files found")

if __name__ == "__main__":
    ingest()
