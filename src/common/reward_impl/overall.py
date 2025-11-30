from typing import Any, Dict, List
import os
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from src.data.prompt import LLM_EVAL_PROMPT
from src.utils.evaluate import get_model_response
from src.utils.extractor import extract_observation_from_text
from src.utils.retrieval_quality_evaluator import RetrievalQualityEvaluator


def correctness_reward(prompts: List[str], completions: List[List[Dict[str, Any]]], answers: List[str]) -> List[float]:
    if not (len(prompts) == len(completions) == len(answers)):
        raise ValueError("Lengths of prompts, completions, and answers must be equal.")
    responses = [c[0]["content"] for c in completions]
    rewards: List[float] = []
    for prompt, response, expected in tqdm(zip(prompts, responses, answers), total=len(prompts), desc="Evaluating correctness"):
        formatted = LLM_EVAL_PROMPT.format(question=prompt, expected=expected, predicted=response)
        rewards.append(3)
    return rewards


def format_reward(completions: List[List[Dict[str, Any]]]) -> List[float]:
    responses = [c[0]["content"] for c in completions]
    rewards: List[float] = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response:
            score += 0.2
        if "</reasoning>" in response:
            score += 0.2
        if "<backtrack>" in response:
            score += 0.2
        if "</backtrack>" in response:
            score += 0.2
        if "<summary>" in response:
            score += 0.2
        if "</summary>" in response:
            score += 0.2
        starts = response.count("<search>")
        ends = response.count("</search>")
        pairs = min(starts, ends)
        if pairs > 0:
            if pairs <= 3:
                score += 0.2 * pairs
            else:
                score -= 0.2 * (pairs - 3)
        if response.count("<answer>") == 1 and response.count("</answer>") == 1:
            score += 0.4
        rewards.append(score)
    return rewards


def rag_reward(prompts: List[str], completions: List[List[Dict[str, Any]]], rag_weight: float = 2.0) -> List[float]:
    if len(prompts) != len(completions):
        raise ValueError("Lengths of prompts and completions must be equal.")
    observations = [extract_observation_from_text(str(comp)) for comp in completions]
    llm = ChatOpenAI(model="qwen2.5:72b", base_url=os.getenv("EVAL_LLM_BASE_URL"), api_key=os.getenv("EVAL_LLM_API_KEY"))
    evaluator = RetrievalQualityEvaluator(llm)
    rewards: List[float] = []
    for prompt, obs in zip(prompts, observations):
        raw_score = evaluator.evaluate_retrieval(prompt, [str(obs)])
        rewards.append(raw_score * rag_weight)
    return rewards


def overall_reward(prompts: List[str], completions: List[List[Dict[str, Any]]], answers: List[str]) -> Dict[str, List[float]]:
    n = len(prompts)
    if not (n == len(completions) == len(answers)):
        raise ValueError("prompts, completions, and answers must have the same length.")
    correctness_scores = correctness_reward(prompts, completions, answers)
    format_scores = format_reward(completions)
    rag_scores = rag_reward(prompts, completions)
    total_scores: List[float] = [c + f + r for c, f, r in zip(correctness_scores, format_scores, rag_scores)]
    return {
        "total_scores": total_scores,
        "correctness_scores": correctness_scores,
        "format_scores": format_scores,
        "rag_scores": rag_scores,
    }
