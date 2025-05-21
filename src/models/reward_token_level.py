import os
import pdb
import re
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

import torch
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from src.data.prompt import LLM_EVAL_PROMPT
from src.utils.evaluate import get_model_response
from src.utils.extractor import extract_observation_from_text
from src.utils.retrieval_quality_evaluator import RetrievalQualityEvaluator


def correctness_reward(
    prompts: List[str],
    completions: List[List[Dict[str, Any]]],
    answers: List[str],
) -> List[float]:
    """
    Assigns a reward based on the correctness of the model's answers.

    For each prompt, compares the model's first completion to the expected answer
    using an LLM evaluation prompt. Returns 3.0 for a "yes" judgment, 0.0 otherwise.
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


def _collect_char_spans(pattern: str, text: str, flags: int = 0) -> List[Tuple[int, int]]:
    """Return (start, end) char-level spans for every regex match in text."""
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags)]


def _offset_mapping(text: str, tokenizer) -> List[Tuple[int, int]]:
    """Return a list of (start, end) char offsets for each token emitted by tokenizer."""
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return enc["offset_mapping"]


def _token_indices_from_spans(
    offsets: Sequence[Tuple[int, int]],
    spans: Sequence[Tuple[int, int]],
) -> Set[int]:
    """Given token offsets and char spans, return set of token indices whose range overlaps any span."""
    out: Set[int] = set()
    for s_start, s_end in spans:
        for idx, (t_start, t_end) in enumerate(offsets):
            if t_end <= s_start:
                continue
            if t_start >= s_end:
                break
            out.add(idx)
    return out


def overall_reward_token_level(
    prompts: List[str],
    completions: List[List[Dict[str, Any]]],
    answers: List[str],
    completion_ids: Union[List[List[int]], "Tensor"],  # type: ignore[valid-type]
    tokenizer: Any,
) -> Dict[str, Any]:
    """
    Compute token-wise rewards, mirroring the shape of completion_ids.
    """

    # Convert tensor → nested list for processing
    is_tensor = torch is not None and isinstance(completion_ids, torch.Tensor)
    ids_list: List[List[int]] = completion_ids.tolist() if is_tensor else completion_ids

    n = len(prompts)
    if not (n == len(completions) == len(answers) == len(ids_list)):
        raise ValueError("prompts, completions, answers, completion_ids must share the same length")

    correctness_scores = correctness_reward(prompts, completions, answers)
    format_scores = format_reward(completions)
    rag_scores = rag_reward(prompts, completions)
    total_scores = [c + f + r for c, f, r in zip(correctness_scores, format_scores, rag_scores)]

    token_rewards_rows: List[List[float]] = []

    search_block_re = re.compile(r"<search>[\s\S]*?</search>")
    single_tag_scores = {
        "<reasoning>": 0.2,
        "</reasoning>": 0.2,
        "<backtrack>": 0.2,
        "</backtrack>": 0.2,
        "<summary>": 0.2,
        "</summary>": 0.2,
    }

    for idx_sample in range(n):
        comp_text = completions[idx_sample][0]["content"]
        ids_row = ids_list[idx_sample]
        seq_len = len(ids_row)
        per_token = [0.0] * seq_len

        # --- correctness (uniform)
        if correctness_scores[idx_sample] != 0 and seq_len:
            # bonus = correctness_scores[idx_sample] / seq_len
            bonus = correctness_scores[idx_sample]
            per_token = [v + bonus for v in per_token]

        # Offsets (align with tokenizer)
        offsets = _offset_mapping(comp_text, tokenizer)

        # Ensure offsets length == seq_len (pad / truncate)
        if len(offsets) < seq_len:
            pad_n = seq_len - len(offsets)
            last_end = offsets[-1][1] if offsets else 0
            offsets.extend([(last_end, last_end)] * pad_n)
        elif len(offsets) > seq_len:
            offsets = offsets[:seq_len]

        # --- RAG block reward
        rag_spans = _collect_char_spans(search_block_re, comp_text)
        rag_token_idxs = _token_indices_from_spans(offsets, rag_spans)
        if rag_token_idxs and rag_scores[idx_sample] != 0:
            # bonus = rag_scores[idx_sample] / len(rag_token_idxs)
            bonus = rag_scores[idx_sample]
            for j in rag_token_idxs:
                per_token[j] += bonus

        # --- Format: single tags
        for tag, tag_score in single_tag_scores.items():
            for span in _collect_char_spans(re.escape(tag), comp_text):
                tag_token_idxs = _token_indices_from_spans(offsets, [span])
                if tag_token_idxs:
                    # bonus = tag_score / len(tag_token_idxs)
                    bonus = tag_score
                    for j in tag_token_idxs:
                        per_token[j] += bonus

        # --- Format: <search> tag pair scoring
        open_tags = _collect_char_spans(re.escape("<search>"), comp_text)
        close_tags = _collect_char_spans(re.escape("</search>"), comp_text)
        pair_cnt = min(len(open_tags), len(close_tags))
        if pair_cnt:
            pair_score = 0.2 * pair_cnt if pair_cnt <= 3 else -0.2 * (pair_cnt - 3)
            tag_spans = open_tags[:pair_cnt] + close_tags[:pair_cnt]
            idxs = _token_indices_from_spans(offsets, tag_spans)
            if idxs:
                # bonus = pair_score / len(idxs)
                bonus = pair_score
                for j in idxs:
                    per_token[j] += bonus

        # --- Format: <answer> single pair
        if comp_text.count("<answer>") == 1 and comp_text.count("</answer>") == 1:
            a_open = _collect_char_spans(re.escape("<answer>"), comp_text)[0]
            a_close = _collect_char_spans(re.escape("</answer>"), comp_text)[0]
            idxs = _token_indices_from_spans(offsets, [a_open, a_close])
            if idxs:
                # bonus = 0.4 / len(idxs)
                bonus = 0.4
                for j in idxs:
                    per_token[j] += bonus

        token_rewards_rows.append(per_token)

    # Convert back to tensor if needed
    if is_tensor:
        device = completion_ids.device
        tok_tensor = torch.zeros_like(completion_ids, dtype=torch.float32, device=device)
        for row_i, row in enumerate(token_rewards_rows):
            L = min(len(row), tok_tensor.size(1))
            tok_tensor[row_i, :L] = torch.tensor(row[:L], dtype=torch.float32, device=device)
        # pdb.set_trace()
        return {
            "token_rewards": tok_tensor,
            "total_scores": total_scores,
            "correctness_scores": correctness_scores,
            "format_scores": format_scores,
            "rag_scores": rag_scores,
        }
    else:
        return {
            "token_rewards": token_rewards_rows,
            "total_scores": total_scores,
            "correctness_scores": correctness_scores,
            "format_scores": format_scores,
            "rag_scores": rag_scores,
        }
