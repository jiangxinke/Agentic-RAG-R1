from utils.answer_extractor import *
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

    Explanation:
        1. Extracts the text content from each completion.
        2. Processes each response to extract the answer portion.
        3. Compares extracted answers with expected answers using two methods:
           - Exact string matching (2.0 points)
           - Numeric equivalence check (1.5 points)
        4. Returns a list of reward scores.
    """
    # Extract the content from each completion's first element
    responses = [completion[0]["content"] for completion in completions]

    # Extract answers from model outputs
    extracted = [extract_answer_from_model_output(r) for r in responses]

    rewards = []
    for r, a in zip(extracted, answer):
        if r == a:  # Exact match case
            rewards.append(2.0)
        else:
            # Try numeric equivalence
            r_num = extract_single_number(str(r))
            a_num = extract_single_number(str(a))
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)

    # Log completion lengths
    completion_lengths = [len(response.split()) for response in responses]
    return rewards


def format_reward(completions, **kwargs):
    """
    Assigns a reward for adhering to the desired XML format.

    Args:
        completions (list[list[dict]]): List of completion dictionaries.
        **kwargs: Additional keyword arguments.

    Returns:
        list[float]: Reward scores based on format compliance.

    Explanation:
        1. Extracts the text content from each completion.
        2. Assigns points based on the presence of required XML tags:
           - 0.2 points for opening <reasoning> tag
           - 0.2 points for closing </reasoning> tag
           - 0.2 points for opening <answer> tag
           - 0.2 points for closing </answer> tag
        3. Returns a list of format compliance scores.
    """
    # Extract the content from each completion's first element
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    format_scores = []

    for response in responses:
        score = 0.0
        if "<reasoning>" in response:
            score += 0.2
        if "</reasoning>" in response:
            score += 0.2
        if "<search>" in response:
            score += 0.2
        if "</search>" in response:
            score += 0.2
        if "<answer>" in response:
            score += 0.2
        if "</answer>" in response:
            score += 0.2
        rewards.append(score)
        format_scores.append(score)

    return rewards


def rag_reward(prompts, completions):
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
        rag_reward.append(evaluation_result)

    return rag_reward


def combined_reward(prompts, completions, answer):
    """
    Combines correctness and format rewards to provide a comprehensive evaluation.

    Args:
        prompts (list[str]): List of prompt texts.
        completions (list[list[dict]]): List of completion dictionaries.
        answer (list[str]): List of expected answers.

    Returns:
        list[float]: Combined rewards for each prompt-completion pair.

    Explanation:
        1. Calculates individual reward components:
           - Correctness rewards (range: 0.0 to 2.0)
           - Format rewards (range: 0.0 to 0.8)
        2. Combines the rewards by adding them together.
        3. Returns the combined scores with total range of 0.0 to 2.8.
    """
    # Get individual rewards
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)
    rag_scores = rag_reward(prompts=prompts, completions=completions)

    # Combine rewards - correctness is weighted more heavily
    combined_rewards = []
    for c_score, f_score, rag_score in zip(correctness_scores, format_scores, rag_scores):
        # Correctness score range: 0.0 to 2.0
        # Format score range: 0.0 to 0.8
        # Total range: 0.0 to 2.8
        combined_rewards.append(c_score + f_score + rag_score)

    return combined_rewards
