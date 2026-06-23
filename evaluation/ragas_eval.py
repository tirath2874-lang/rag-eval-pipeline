import json
from datasets import Dataset

from rag import rag_answer

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
)

with open("evaluation/test_dataset.json", "r", encoding="utf-8") as f:
    tests = json.load(f)

questions = []
answers = []
contexts = []
ground_truths = []

for row in tests:
    result = rag_answer(row["question"])

    questions.append(row["question"])
    answers.append(result["answer"])

    contexts.append([result["context"]])

    ground_truths.append(row["ground_truth"])

dataset = Dataset.from_dict(
    {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    }
)

result = evaluate(
    dataset,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
    ],
)

print(result)