from deepeval import assert_test

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric
)

from deepeval.test_case import LLMTestCase

from rag import rag_answer

question = "What is FastAPI?"

result = rag_answer(question)

test_case = LLMTestCase(
    input=question,
    actual_output=result["answer"],
    retrieval_context=[result["context"]]
)

faithfulness = FaithfulnessMetric(
    threshold=0.7
)

relevancy = AnswerRelevancyMetric(
    threshold=0.7
)

assert_test(
    test_case,
    [faithfulness,relevancy]
)