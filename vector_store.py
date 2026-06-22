import chromadb

from sentence_transformers import SentenceTransformer

from config import *


client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    COLLECTION_NAME
)

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def add_document(text,id):

    embedding = embedder.encode(text).tolist()

    collection.add(
        ids=[id],
        documents=[text],
        embeddings=[embedding]
    )


def search(query,k=3):

    q_embedding = embedder.encode(query).tolist()

    result = collection.query(
        query_embeddings=[q_embedding],
        n_results=k
    )

    return result["documents"][0]
