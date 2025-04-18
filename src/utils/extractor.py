import random
import re

import numpy as np
import torch

# def extract_answer_from_model_output(text):
#     """
#     Extracts the value from the last <answer> tag in the text.
#     Returns None if no valid answer is found.
#     """
#     # Split on <answer> and take everything after the last occurrence
#     parts = text.split("<answer>")
#     if len(parts) < 2:  # No <answer> tag found
#         return None

#     last_part = parts[-1]

#     # Extract content up to </answer>
#     if "</answer>" not in last_part:
#         return None

#     answer = last_part.split("</answer>")[0].strip()
#     return None if answer == "..." else answer


def extract_answer_from_model_output(text):
    """
    从文本中提取第一个 <answer> 标签到第一个 </answer> 标签之间的内容。
    如果未找到有效答案，则返回 None。
    """
    start_index = text.find("<answer>")
    if start_index == -1:
        return None

    # 计算 <answer> 标签结束的位置
    start_index += len("<answer>")
    end_index = text.find("</answer>", start_index)
    if end_index == -1:
        return None
    answer = text[start_index:end_index].strip()
    return None if answer == "..." else answer


def extract_answer_from_dataset(text):
    """
    Extracts the answer from the dataset.
    The dataset separates the answer using the '####' delimiter.
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_last_number(text):
    """
    Extracts the last number from text if it's properly separated.

    Args:
        text (str): The text to extract a number from.

    Returns:
        float or None: The extracted number as a float, or None if no valid number is found.

    Explanation:
        1. First removes $ and % signs from the text.
        2. Uses regex to find numbers that are:
           - Preceded by space, equals sign, or start of string
           - Followed by end of string or space
        3. Returns the first matching number as a float, or None if no match is found.
    """
    import re

    # Remove $ and % signs
    text = text.replace("$", "").replace("%", "")

    # HERE
    # Look for numbers that are:
    # - preceded by space or = or start of string (via \b or ^)
    # - followed by end of string or space
    pattern = r"(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def extract_single_number(text):
    """
    Extracts a single number from text if exactly one exists.

    Args:
        text (str): The text to extract a number from.

    Returns:
        float or None: The extracted number as a float if exactly one number exists,
                      otherwise None.

    Explanation:
        1. Uses regex to find all numbers in the text.
        2. Returns the first number as a float if exactly one number is found.
        3. Returns None if zero or multiple numbers are found.
    """
    import re

    numbers = re.findall(r"-?\d*\.?\d+", text)
    return float(numbers[0]) if len(numbers) == 1 else None


def extract_observation_from_text(text):
    """
    Extracts the content between <observation> and </observation> tags.

    Args:
        content (str): The string content from which to extract the observation.

    Returns:
        str: The content between <observation> and </observation> tags.
    """
    # Use regular expression to extract the content between <observation>...</observation>
    match = re.search(r"<observation>(.*?)</observation>", text, re.DOTALL)
    if match:
        return match.group(1).strip()  # Return the extracted observation content
    return "No observation"  # If no observation tag is found, return an empty string


def analyze_completions(completions, reward_dict):
    import re

    processed_completions = []
    for comp in completions:
        if isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], dict) and "content" in comp[0]:
            processed_completions.append(comp[0]["content"])
        elif isinstance(comp, dict) and "content" in comp:
            processed_completions.append(comp["content"])
        elif isinstance(comp, str):
            processed_completions.append(comp)
        else:
            continue

    if not processed_completions:
        return {
            "avg_completion_length": 0,
            "avg_completion_length_noobservation": 0,
            "answer_format_accuracy": 0,
            "avg_search_pairs": 0,
            "avg_search_content_length": 0,
            "avg_reasoning_pairs": 0,
            "avg_reasoning_content_length": 0,
            "avg_backtrack_pairs": 0,
            "avg_backtrack_content_length": 0,
        }

    # 1. 完成长度
    completion_lengths = [len(comp) for comp in processed_completions]
    print(f"completion_lengths: {completion_lengths}")
    avg_completion_length = sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0

    # 1.1 计算去除observation标签内容后的长度
    completion_lengths_noobservation = []
    for comp in processed_completions:
        # 移除所有<observation>...</observation>内容
        cleaned_comp = re.sub(r"<observation>.*?</observation>", "", comp, flags=re.DOTALL)
        completion_lengths_noobservation.append(len(cleaned_comp))

    avg_completion_length_noobservation = (
        sum(completion_lengths_noobservation) / len(completion_lengths_noobservation) if completion_lengths_noobservation else 0
    )

    # 2. 答案格式准确率
    answer_format_count = sum(1 for comp in processed_completions if "<answer>" in comp and "</answer>" in comp)
    avg_answer_format_accuracy = answer_format_count / len(processed_completions) if processed_completions else 0

    # 3. 搜索相关指标
    search_pairs_count = 0
    search_content_lengths = []

    for comp in processed_completions:
        search_starts = [m.start() for m in re.finditer("<search>", comp)]
        search_ends = [m.start() for m in re.finditer("</search>", comp)]

        valid_pairs = 0
        for start_pos in search_starts:
            valid_end = next((end for end in search_ends if end > start_pos), None)
            if valid_end is not None:
                valid_pairs += 1
                content_length = len(comp[start_pos + len("<search>") : valid_end].strip().split())
                search_content_lengths.append(content_length)
                search_ends.remove(valid_end)

        search_pairs_count += valid_pairs

    # 4. 推理标志使用情况
    reasoning_pairs_count = 0
    reasoning_content_lengths = []

    for comp in processed_completions:
        if "<reasoning>" in comp and "</reasoning>" in comp:
            reasoning_pairs_count += 1
            reasoning_start = comp.find("<reasoning>") + len("<reasoning>")
            reasoning_end = comp.find("</reasoning>")
            if reasoning_start < reasoning_end:
                content_length = len(comp[reasoning_start:reasoning_end].strip().split())
                reasoning_content_lengths.append(content_length)

    # 5. 反思标签使用情况
    backtrack_pairs_count = 0
    backtrack_content_lengths = []

    for comp in processed_completions:
        if "<backtrack>" in comp and "</backtrack>" in comp:
            backtrack_pairs_count += 1
            backtrack_start = comp.find("<backtrack>") + len("<backtrack>")
            backtrack_end = comp.find("</backtrack>")
            if backtrack_start < backtrack_end:
                content_length = len(comp[backtrack_start:backtrack_end].strip().split())
                backtrack_content_lengths.append(content_length)

    total_completions = len(processed_completions)

    avg_search_pairs = search_pairs_count / total_completions if total_completions else 0
    avg_search_content_length = sum(search_content_lengths) / len(search_content_lengths) if search_content_lengths else 0

    avg_reasoning_pairs = reasoning_pairs_count / total_completions if total_completions else 0
    avg_reasoning_content_length = (
        sum(reasoning_content_lengths) / len(reasoning_content_lengths) if reasoning_content_lengths else 0
    )

    avg_backtrack_pairs = backtrack_pairs_count / total_completions if total_completions else 0
    avg_backtrack_content_length = (
        sum(backtrack_content_lengths) / len(backtrack_content_lengths) if backtrack_content_lengths else 0
    )
    # 汇总奖励和性能指标
    metrics = {
        "avg_completion_length": avg_completion_length,
        "avg_completion_length_noobservation": avg_completion_length_noobservation,
        "avg_answer_format_accuracy": avg_answer_format_accuracy,
        "avg_search_pairs": avg_search_pairs,
        "avg_search_content_length": avg_search_content_length,
        "avg_reasoning_pairs": avg_reasoning_pairs,
        "avg_reasoning_content_length": avg_reasoning_content_length,
        "avg_backtrack_pairs": avg_backtrack_pairs,
        "avg_backtrack_content_length": avg_backtrack_content_length,
    }

    # 添加奖励指标
    if "correctness_scores" in reward_dict:
        metrics["correctness_reward"] = torch.tensor(reward_dict["correctness_scores"]).mean().item()
    if "format_scores" in reward_dict:
        metrics["format_reward"] = torch.tensor(reward_dict["format_scores"]).mean().item()
    if "rag_scores" in reward_dict:
        metrics["rag_reward"] = torch.tensor(reward_dict["rag_scores"]).mean().item()

    return metrics
