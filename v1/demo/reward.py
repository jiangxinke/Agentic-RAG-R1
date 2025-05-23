import os
from typing import Any, Dict, List

import numpy as np
import ray
from openai import OpenAI
from tqdm import tqdm

# Prompt for LLM judge
LLM_EVAL_PROMPT = """
你是一名严格、但能识别同义表达的阅卷老师。请阅读以下信息并判断学生的选择题作答是否正确：

1. 【题目】：
{question}

2. 【正确答案】：
{expected}

3. 【学生的作答】：
{predicted}

你的任务是：
- 首先判断学生的作答是否与正确答案一致（如果含义相同也视为一致）；
- 如果学生作答正确，请只输出：Yes
- 如果学生作答错误，请只输出：No

**重要要求**：
- 不要输出引号、标点、换行、额外文字、空格或其他任何字符。
- 只输出一个单词：Yes 或 No。
"""


def get_model_response(prompt: str, model="qwen2.5:72b", temperature=0.7):
    client = OpenAI(
        api_key=os.getenv("EVAL_LLM_API_KEY"),
        base_url=os.getenv("EVAL_LLM_BASE_URL"),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def llm_eval_accuracy(prompts: List[str], predictions: List[str], answers: List[str]) -> List[float]:
    """
    用 LLM 判分，每题输出 3.0 或 0.0。
    """
    rewards = []
    for question, predicted, expected in tqdm(zip(prompts, predictions, answers), total=len(prompts), desc="LLM Eval"):
        formatted_prompt = LLM_EVAL_PROMPT.format(question=question, expected=expected, predicted=predicted)
        llm_response = get_model_response(formatted_prompt)
        rewards.append(3.0 if "yes" in llm_response.lower() else 0.0)
    return rewards


def format_reward(responses: List[str]) -> List[float]:
    """
    Computes a formatting reward based on the presence and counts of specific XML-like tags
    in each model response.
    """
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
