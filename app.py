from fastapi import FastAPI

from rag import rag_answer
from vector_store import add_document

app = FastAPI()

@app.post("/ingest")
def ingest(data:dict):

    add_document(
        data["text"],
        data["id"]
    )

    return {"status":"stored"}

@app.post("/ask")
def ask(data:dict):

    return rag_answer(
        data["question"]
    )