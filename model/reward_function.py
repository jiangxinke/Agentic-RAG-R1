from tqdm import tqdm

from utils.answer_extractor import *
from utils.evaluate import get_model_response
from utils.retrieval_quality_evaluator import *


def correctness_reward(prompts, completions, answer, **kwargs):
    """
    Assigns a reward based on the correctness of the model's answer.

    Args:
        prompts (list[str]): List of prompt texts.
        completions (list[list[dict]]): List of completion dictionaries.
        answer (list[str]): List of expected answers.
        **kwargs: Additional keyword arguments.

    Returns:
        list[float]: Reward scores based on answer correctness.
    """
    # Extract the content from each completion's first element
    responses = [completion[0]["content"] for completion in completions]

    # Prepare the evaluation prompt
    eval_prompt = """
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

    rewards = []

    for prompt, response, ans in tqdm(zip(prompts, responses, answer), total=len(prompts), desc="Evaluating correctness"):
        formatted_prompt = eval_prompt.format(question=prompt, expected=ans, predicted=response)
        llm_response = get_model_response(formatted_prompt)

        if "yes" in llm_response.lower():
            rewards.append(3.0)  # Full points for correct answer
        else:
            rewards.append(0.0)  # No points for incorrect answer

    return rewards


# def correctness_reward(prompts, completions, answer, **kwargs):
#     """
#     Assigns a reward based on the correctness of the model's answer.

#     Args:
#         prompts (list[str]): List of prompt texts.
#         completions (list[list[dict]]): List of completion dictionaries.
#         answer (list[str]): List of expected answers.
#         **kwargs: Additional keyword arguments.

#     Returns:
#         list[float]: Reward scores based on answer correctness.

#     Explanation:
#         1. Extracts the text content from each completion.
#         2. Processes each response to extract the answer portion.
#         3. Compares extracted answers with expected answers using two methods:
#            - Exact string matching (2.0 points)
#            - Numeric equivalence check (1.5 points)
#         4. Returns a list of reward scores.
#     """
#     # Extract the content from each completion's first element
#     responses = [completion[0]["content"] for completion in completions]

#     # Extract answers from model outputs
#     extracted = [extract_answer_from_model_output(r) for r in responses]

#     rewards = []
#     for r, a in zip(extracted, answer):
#         if r == a:  # Exact match case
#             rewards.append(2.0)
#         else:
#             rewards.append(0.0)

#     # Log completion lengths
#     completion_lengths = [len(response.split()) for response in responses]
#     return rewards


def format_reward(completions, **kwargs):
    """
    计算格式奖励：
      1) <reasoning> 和 </reasoning>：
         - 只要检测到就分别给 0.2 分 (共 0.4)。
      2) <search> 和 </search>：
         - 最多允许出现 3 对；1～3 对时，每对 0.2 分 (最高 0.6)；
         - 若大于 3 对，直接得 0 分（表示扣分）。
      3) <answer> 和 </answer>：
         - 必须恰好出现 1 对才给 0.4 分，否则为 0 分。
      4) <backtrack> 和 </backtrack>：
         - 只要检测到就分别给 0.2 分 (共 0.4)。
      注：如果需要严格检查顺序，可以额外比较标签在字符串中的索引。
    """

    # 提取 completion 内容
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response in responses:
        score = 0.0

        # <reasoning> 检测，出现就加分
        if "<reasoning>" in response:
            score += 0.2
        if "</reasoning>" in response:
            score += 0.2

        if "<backtrack>" in response:
            score += 0.2
        if "</backtrack>" in response:
            score += 0.2

        # <search> 可多对，超过 3 对扣分
        search_start_count = response.count("<search>")
        search_end_count = response.count("</search>")
        search_pairs = min(search_start_count, search_end_count)

        if search_pairs == 0:
            pass
        elif search_pairs <= 3:
            # 1 到 3 对之间：每对 0.2 分
            # 上限可达 3 * 0.2 = 0.6
            score += 0.2 * search_pairs
        else:
            # 超过 3 对 => "最高 0.6，额外对数进行扣分"
            score -= 0.2 * (search_pairs - 3)

        # <answer> 只允许 1 对
        answer_start_count = response.count("<answer>")
        answer_end_count = response.count("</answer>")
        # 当 exactly one pair => +0.4
        if (answer_start_count == 1) and (answer_end_count == 1):
            score += 0.4

        # 记录该条的格式分
        rewards.append(score)

    return rewards


def rag_reward(prompts, completions, rag_weight=2):
    # list; list
    extract_observation = [extract_observation_from_text(str(item_completion)) for item_completion in completions]

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="qwen2.5:72b",
        base_url="http://8289.model.mingxingtech.com:10032/v1",
        api_key="8cefb70606f3472d8731bd65661ce409",
    )

    rag_reward = []
    for prompt, item_observation in zip(prompts, extract_observation):
        user_input = prompt
        retrieved_contexts = [
            str(item_observation),
        ]

        rag_evaluator = RetrievalQualityEvaluator(llm)
        evaluation_result = rag_evaluator.evaluate_retrieval(user_input, retrieved_contexts)
        # Apply weight to the RAG reward
        weighted_result = evaluation_result * rag_weight
        rag_reward.append(weighted_result)

    return rag_reward


def combined_reward(prompts, completions, answer):
    """
    Combines correctness and format rewards to provide a comprehensive evaluation.

    Args:
        prompts (list[str]): List of prompt texts.
        completions (list[list[dict]]): List of completion dictionaries.
        answer (list[str]): List of expected answers.

    Returns:
        dict: A dictionary containing:
            - 'total_scores': List of combined rewards (0.0 to 2.8)
            - 'correctness_scores': List of correctness rewards (0.0 to 2.0)
            - 'format_scores': List of format rewards (0.0 to 0.8)
            - 'rag_scores': List of RAG rewards
    """
    # Get individual rewards
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)
    rag_scores = rag_reward(prompts=prompts, completions=completions)

    # Combine rewards - correctness is weighted more heavily
    total_scores = []
    for c_score, f_score, rag_score in zip(correctness_scores, format_scores, rag_scores):
        # Correctness score range: 0.0 to 2.0
        # Format score range: 0.0 to 0.8
        # Total range: 0.0 to 2.8
        total_scores.append(c_score + f_score + rag_score)

    return {
        "total_scores": total_scores,
        "correctness_scores": correctness_scores,
        "format_scores": format_scores,
        "rag_scores": rag_scores,
    }
