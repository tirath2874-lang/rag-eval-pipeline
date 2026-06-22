from vector_store import search
from ollama_client import generate
from vector_store import collection

def rag_answer(question):

    docs = search(question)

    context = "\n".join(docs)

    prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Answer only using context.
    """

    answer = generate(prompt)

    return {
        "question":question,
        "context":context,
        "answer":answer
    }
print(collection.count())