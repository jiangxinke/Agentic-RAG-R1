from typing import Any, Dict, List
import os
import re

from langchain_openai import ChatOpenAI
from tqdm import tqdm

from src.data.prompt import LLM_EVAL_PROMPT
from src.utils.evaluate import get_model_response
from src.utils.extractor import extract_observation_from_text
from src.utils.retrieval_quality_evaluator import RetrievalQualityEvaluator


def correctness_reward(prompts: List[str], completions: List[List[Dict[str, Any]]], answers: List[str]) -> List[float]:
    """
    Assigns a reward based on the correctness of the model's answers.

    For each prompt, compares the model's first completion to the expected answer
    using an LLM evaluation prompt. Returns 3.0 for a "yes" judgment, 0.0 otherwise.

    Args:
        prompts: List of prompt strings to evaluate.
        completions: Nested list of completion dicts from the model; we use the first element's "content".
        answers: List of expected answer strings.

    Returns:
        A list of floats, one per prompt, where each value is either 3.0 (correct) or 0.0 (incorrect).

    Raises:
        ValueError: If the lengths of `prompts`, `completions`, and `answers` do not match.
        IndexError: If a completion list is empty or lacks a "content" field.
    """
    if not (len(prompts) == len(completions) == len(answers)):
        raise ValueError("Lengths of prompts, completions, and answers must be equal.")

    responses = [c[0]["content"] for c in completions]
    rewards: List[float] = []

    for prompt, response, expected in tqdm(zip(prompts, responses, answers), total=len(prompts), desc="Evaluating correctness"):
        formatted = LLM_EVAL_PROMPT.format(question=prompt, expected=expected, predicted=response)
        llm_resp = get_model_response(formatted)
        rewards.append(3.0 if "yes" in llm_resp.lower() else 0.0)

    return rewards


def format_reward(completions: List[List[Dict[str, Any]]]) -> List[float]:
    """
    Computes a formatting reward based on the presence and counts of specific XML-like tags
    in each model response.

    Tag scoring:
      - <reasoning> and </reasoning>: 0.2 each (max 0.4)
      - <backtrack> and </backtrack>: 0.2 each (max 0.4)
      - <summary> and </summary>: 0.2 each (max 0.4)
      - <search> ... </search>: 0.2 per matching pair, up to 3 pairs (max 0.6); pairs beyond 3 subtract 0.2 each
      - <answer> ... </answer>: 0.4 if exactly one matching pair, otherwise 0.0

    Args:
        completions: Nested list of completion dicts from the model; we use each first element's "content".

    Returns:
        A list of floats, one per completion, representing the format score.

    Raises:
        IndexError: If a completion list is empty or lacks a "content" field.
    """
    responses = [c[0]["content"] for c in completions]
    rewards: List[float] = []

    for response in responses:
        score = 0.0

        # reasoning tags
        if "<reasoning>" in response:
            score += 0.2
        if "</reasoning>" in response:
            score += 0.2

        # backtrack tags
        if "<backtrack>" in response:
            score += 0.2
        if "</backtrack>" in response:
            score += 0.2
            
        # summary tags
        if "<summary>" in response:
            score += 0.2
        if "</summary>" in response:
            score += 0.2

        # search tag pairs
        starts = response.count("<search>")
        ends = response.count("</search>")
        pairs = min(starts, ends)
        if pairs > 0:
            if pairs <= 3:
                score += 0.2 * pairs
            else:
                score -= 0.2 * (pairs - 3)

        # answer tag pair
        if response.count("<answer>") == 1 and response.count("</answer>") == 1:
            score += 0.4

        rewards.append(score)

    return rewards


def rag_reward(prompts: List[str], completions: List[List[Dict[str, Any]]], rag_weight: float = 2.0) -> List[float]:
    """
    Computes a reward based on retrieval-augmented generation (RAG) quality.

    For each prompt, extracts observations from the completion text, then uses a
    RetrievalQualityEvaluator backed by a ChatOpenAI LLM to score how well the
    retrieved context matches the prompt. The raw score is multiplied by `rag_weight`.

    Args:
        prompts: List of prompt strings to evaluate.
        completions: Nested list of completion dicts from the model.
        rag_weight: Multiplier to apply to each raw RAG evaluation score.

    Returns:
        A list of floats, one per prompt, representing the weighted RAG score.

    Raises:
        ValueError: If the lengths of `prompts` and `completions` do not match.
        Exception: Propagates errors from `extract_observation_from_text` or the evaluator.
    """
    if len(prompts) != len(completions):
        raise ValueError("Lengths of prompts and completions must be equal.")

    observations = [extract_observation_from_text(str(comp)) for comp in completions]

    llm = ChatOpenAI(
        model="qwen2.5:72b",
        base_url=os.getenv("EVAL_LLM_BASE_URL"),
        api_key=os.getenv("EVAL_LLM_API_KEY"),
    )
    evaluator = RetrievalQualityEvaluator(llm)

    rewards: List[float] = []
    for prompt, obs in zip(prompts, observations):
        raw_score = evaluator.evaluate_retrieval(prompt, [str(obs)])
        rewards.append(raw_score * rag_weight)

    return rewards


def overall_reward(prompts: List[str], completions: List[List[Dict[str, Any]]], answers: List[str]) -> Dict[str, List[float]]:
    """
    Combines correctness, format, and RAG rewards into a comprehensive score set.

    Args:
        prompts: List of prompt strings.
        completions: Nested list of completion dicts from the model.
        answers: List of expected answer strings.

    Returns:
        A dict with keys:
          - 'total_scores': Combined scores (correctness + format + RAG).
          - 'correctness_scores': Individual correctness rewards.
          - 'format_scores': Individual format rewards.
          - 'rag_scores': Individual RAG rewards.

    Raises:
        ValueError: If the lengths of inputs do not align.
    """
    # Validation
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


def token_level_advanced_scores(prompts: List[str], completions: List[List[Dict[str, Any]]], answers: List[str]) -> List[List[float]]:
    """
    对每个token进行分析，计算奖励：(1-mask)*r + act*总分
    mask: 在<search></search>内为1，否则0
    act: 在<action></action>内为1，否则0
    r: 总分/该回答token数
    """
    all_scores = overall_reward(prompts, completions, answers)
    total_scores = all_scores["total_scores"]
    token_rewards = []

    for idx, completion in enumerate(completions):
        text = completion[0]["content"]
        tokens = text.split()  # 可替换为更细致的分词方式
        n_tokens = len(tokens)
        if n_tokens == 0:
            token_rewards.append([])
            continue
        r = total_scores[idx] / n_tokens

        # 记录每个token的mask和act
        mask_flags = [0] * n_tokens
        act_flags = [0] * n_tokens

        # 计算每个token的起止字符位置
        token_spans = []
        pos = 0
        for t in tokens:
            start = text.find(t, pos)
            end = start + len(t)
            token_spans.append((start, end))
            pos = end

        # 找出所有<search>...</search>和<action>...</action>区间
        search_spans = [(m.start(), m.end()) for m in re.finditer(r"<search>(.*?)</search>", text, re.DOTALL)]
        action_spans = [(m.start(), m.end()) for m in re.finditer(r"<action>(.*?)</action>", text, re.DOTALL)]

        # 判断每个token是否在这些区间内
        for i, (start, end) in enumerate(token_spans):
            for s_start, s_end in search_spans:
                if start >= s_start and end <= s_end:
                    mask_flags[i] = 1
                    break
            for a_start, a_end in action_spans:
                if start >= a_start and end <= a_end:
                    act_flags[i] = 1
                    break

        # 计算奖励
        rewards = [(1 - mask_flags[i]) * r + act_flags[i] * total_scores[idx] for i in range(n_tokens)]
        token_rewards.append(rewards)

    return token_rewards

