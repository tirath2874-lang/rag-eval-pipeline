import json

from datasets import Dataset

from rag import rag_answer

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)

with open("evaluation/test_dataset.json") as f:
    tests = json.load(f)

questions=[]
answers=[]
contexts=[]
ground_truths=[]

for row in tests:

    result = rag_answer(
        row["question"]
    )

    questions.append(
        row["question"]
    )

    answers.append(
        result["answer"]
    )

    contexts.append(
        [result["context"]]
    )

    ground_truths.append(
        row["ground_truth"]
    )

dataset = Dataset.from_dict(
{
    "question":questions,
    "answer":answers,
    "contexts":contexts,
    "ground_truth":ground_truths
})

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ]
)

print(result)