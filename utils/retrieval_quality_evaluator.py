from typing import List

from langchain_core.language_models import BaseLanguageModel
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextRelevance


class RetrievalQualityEvaluator:
    def __init__(self, llm: BaseLanguageModel):
        evaluator_llm = LangchainLLMWrapper(llm)
        self.scorer = ContextRelevance(llm=evaluator_llm)

    def evaluate_retrieval(self, user_input: str, retrieved_contexts: List[str]):
        """Context Relevance evaluates whether the retrieved_contexts (chunks or passages) are pertinent to the user_input. This is done via two independent "LLM-as-a-judge" prompt calls that each rate the relevance on a scale of 0, 1, or 2. The ratings are then converted to a [0,1] scale and averaged to produce the final score. Higher scores indicate that the contexts are more closely aligned with the user's query.

        0 → The retrieved contexts are not relevant to the user’s query at all.

        1 → The contexts are partially relevant.

        2 → The contexts are completely relevant.

        Step 1: The LLM is prompted with two distinct templates (template_relevance1 and template_relevance2) to evaluate the relevance of the retrieved contexts concerning the user's query. Each prompt returns a relevance rating of 0, 1, or 2.

        Step 2: Each rating is normalized to a [0,1] scale by dividing by 2. If both ratings are valid, the final score is the average of these normalized values; if only one is valid, that score is used.
        """
        sample = SingleTurnSample(user_input=user_input, retrieved_contexts=retrieved_contexts)

        score = self.scorer.single_turn_score(sample)

        return score


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    llm = ChatOpenAI(
        model="qwen2.5:72b",
        base_url=os.getenv("EVAL_LLM_BASE_URL"),
        api_key=os.getenv("EVAL_LLM_API_KEY"),
    )

    user_input = "When and Where Albert Einstein was born?"
    retrieved_contexts = [
        "Albert Einstein was born March 14, 1879.",
        "Albert Einstein was born at Ulm, in Württemberg, Germany.",
    ]

    rag_evaluator = RetrievalQualityEvaluator(llm)
    evaluation_result = rag_evaluator.evaluate_retrieval(user_input, retrieved_contexts)

    print(evaluation_result)
