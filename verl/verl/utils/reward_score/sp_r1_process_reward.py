# verl/utils/reward_score/sp_r1_process_reward.py
import re
import random
from verl.utils.reward_score.search_r1_like_qa_em import compute_score as compute_outcome_score
import sys
import os
import asyncio

from verl.spr1_special_tokens.recognition import TokenRecognizer
from verl.utils.reward_score.sp_r1.search_quality_score import AsyncLLMRelevanceScorer

def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    reflect_reward=1.0,
    **kwargs
):
    """
    Compute both outcome reward and process reward for a QA example.

    Args:
        data_source: str, data source name
        solution_str: str, model-generated text
        ground_truth: dict, ground truth answer info
        extra_info: dict, optional extra context
        reflect_reward: float, reward to give for <reflect> tags
        mock_tool_call: bool, if True, mock tool_call reward instead of calling real evaluator
        kwargs: additional args

    Returns:
        dict:
            {
                "outcome_score": float,
                "process_scores": list[dict]  # each dict: {"from_idx", "to_idx", "score_value", "token_type"}
            }
    """
    extra_info = extra_info or {}
    process_scores = []

    # -------------------------
    # Outcome reward (final answer)
    # -------------------------
    outcome_score = compute_outcome_score(solution_str, ground_truth, method="flexible")
    outcome_score = outcome_score * 5

    # -------------------------
    # Process reward: 
    # -------------------------
    tokenizer = kwargs.get('tokenizer')
    input_ids = kwargs.get('input_ids')

    tokenizer = None # FIXME jxk
    # if tokenizer is not None and input_ids is not None:
    #     try:
    #         recognizer = TokenRecognizer(tokenizer=tokenizer)

    #         # 0. Action Format
    #         answer_intervals = recognizer.find_intervals(input_ids, "<answer>", "</answer>", include_boundaries=True)
    #         for start, end in answer_intervals:
    #             # TokenRecognizer returns [start, end), we want inclusive [start, end-1]
    #             process_scores.append({
    #                 "from_idx": start,
    #                 "to_idx": end - 1,
    #                 "score_value": 0.9,    # format reward
    #                 "token_type": "format"
    #             })
    #         if len(answer_intervals) == 0:
    #             process_scores.append({
    #                 "from_idx": 0,
    #                 "to_idx": 1,
    #                 "score_value": 0.0,
    #                 "token_type": "none"
    #             })

    #         think_intervals = recognizer.find_intervals(input_ids, "<think>", "</think>", include_boundaries=True)
    #         for start, end in think_intervals:
    #             # TokenRecognizer returns [start, end), we want inclusive [start, end-1]
    #             process_scores.append({
    #                 "from_idx": start,
    #                 "to_idx": end - 1,
    #                 "score_value": 0.3,    # format reward
    #                 "token_type": "format"
    #             })
    #         if len(answer_intervals) == 0:
    #             process_scores.append({
    #                 "from_idx": 0,
    #                 "to_idx": 1,
    #                 "score_value": 0.0,
    #                 "token_type": "none"
    #             })
                
    #         # 1. Reflect
    #         reflect_intervals = recognizer.find_intervals(input_ids, "<reflect>", "</reflect>", include_boundaries=True)
    #         for start, end in reflect_intervals:
    #             # TokenRecognizer returns [start, end), we want inclusive [start, end-1]
    #             process_scores.append({
    #                 "from_idx": start,
    #                 "to_idx": end - 1,
    #                 "score_value": reflect_reward if reflect_reward is not None else 0.0, # gjr 0
    #                 "token_type": "reflect"
    #             })
    #                 # "score_value": 0, # gjr 0
    #         if len(reflect_intervals) == 0:
    #             process_scores.append({
    #                 "from_idx": 0,
    #                 "to_idx": 1,
    #                 "score_value": 0.0,
    #                 "token_type": "none"
    #             })
                
    #         # 2. Tool Call
    #         tool_intervals = recognizer.find_intervals(input_ids, "<tool_call>", "</tool_call>", include_boundaries=True)
    #         for start, end in tool_intervals:
    #             # content is between start and end (exclusive of tags)
    #             # interval is [start, end). start_token at start, end_token at end-1.
    #             # content: input_ids[start+1 : end-1]
                
    #             content_ids = input_ids[start+1 : end-1]
    #             if hasattr(content_ids, 'tolist'):
    #                 content_ids = content_ids.tolist()
    #             content = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
                
    #             import os
    #             from dotenv import load_dotenv
    #             load_dotenv()
    #             scorer = AsyncLLMRelevanceScorer(
    #                 model=os.getenv("EVAL_LLM_MODEL_NAME"),
    #                 api_key=os.getenv("EVAL_LLM_API_KEY"),
    #                 base_url=os.getenv("EVAL_LLM_BASE_URL"),
    #                 max_workers=2,
    #             )
    #             # 用户输入和候选内容列表
    #             user_input = extra_info.get("question", "")
    #             retrieved_contexts = [content]  # 可以是单条也可以是多条
    #             # 异步计算相关性分数
    #             async def compute_score():
    #                 score = await scorer.score_batch(user_input, retrieved_contexts)
    #                 return score

    #             try:
    #                 # raise ValueError("Mock tool_call reward not implemented.")
    #                 score_value = asyncio.run(compute_score())
    #             except Exception as e:
    #                 print(f"Error evaluating tool_call: {e}")
    #                 score_value = 0.0
                
    #             process_scores.append({
    #                 "from_idx": start,
    #                 "to_idx": end - 1,
    #                 "score_value": score_value-0.5, # gjr 0
    #                 "token_type": "tool_call"
    #             })
    #                 # "score_value": score_value if score_value is not None else 0.0, # gjr 0
    #     except Exception as e:
    #          print(f"Error in TokenRecognizer: {e}, falling back to string regex.")
    #          pass
    
    ## NOTE 这里不需要tokenizer
    if not (tokenizer is not None and input_ids is not None and TokenRecognizer is not None):

        # 0. Action Format
        matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
        if matches:
            per_score = 0.9 / len(matches)
            for match in matches:
                start, end = match.span()
                process_scores.append({
                    "from_idx": start,
                    "to_idx": end - 1,
                    "score_value": per_score,
                    "token_type": "format"
                })
        if not re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL):
            process_scores.append({
                "from_idx": 0,
                "to_idx": 1,
                "score_value": 0.0,
                "token_type": "none"
            })
        
        matches = list(re.finditer(r"<think>(.*?)</think>", solution_str, re.DOTALL))
        if matches:
            per_score = 0.3 / len(matches)
            for match in matches:
                start, end = match.span()
                process_scores.append({
                    "from_idx": start,
                    "to_idx": end - 1,
                    "score_value": per_score,
                    "token_type": "format"
                })

        if not re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL):
            process_scores.append({
                "from_idx": 0,
                "to_idx": 1,
                "score_value": 0.0,
                "token_type": "none"
            })

        # 2. Reflect
        matches = list(re.finditer(r"<reflect>(.*?)</reflect>", solution_str, re.DOTALL))
        if matches:
            per_score = 0.7 / len(matches)
            for match in matches:
                start, end = match.span()
                process_scores.append({
                    "from_idx": start,
                    "to_idx": end - 1,
                    "score_value": 0, # gjr 0
                    # "score_value": per_score, # gjr 0
                    "token_type": "reflect"
                })  
        if not re.search(r"<reflect>(.*?)</reflect>", solution_str, re.DOTALL):
            process_scores.append({
                "from_idx": 0,
                "to_idx": 1,
                "score_value": 0.0,
                "token_type": "none"
            })

        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", solution_str, re.DOTALL):
            start, end = match.span()
            content = match.group(1).strip()

            #### jr
            import os
            from dotenv import load_dotenv
            load_dotenv()
            scorer = AsyncLLMRelevanceScorer(
                model=os.getenv("EVAL_LLM_MODEL_NAME"),
                api_key=os.getenv("EVAL_LLM_API_KEY"),
                base_url=os.getenv("EVAL_LLM_BASE_URL"),
                max_workers=2,
            )
            # 用户输入和候选内容列表
            user_input = extra_info.get("question", "")
            retrieved_contexts = [content]  # 可以是单条也可以是多条
            # 异步计算相关性分数
            async def compute_score():
                print(f"user_input: {user_input}")
                print(f"retrieved_contexts: {retrieved_contexts}")

                score = await scorer.score_batch(user_input, retrieved_contexts)
                return score

            try:
                # raise ValueError("Mock tool_call reward not implemented.")
                score_value = asyncio.run(compute_score())
            except Exception as e:
                print(f"Error evaluating tool_call: {e}")
                score_value = 0.0

            process_scores.append({
                "from_idx": start,
                "to_idx": end - 1,
                # "score_value": 0,
                "score_value": score_value-0.5, # 由于返回的是0和1，需要变为负数
                "token_type": "tool_call"
            })

    if len(process_scores) == 0:
        process_scores.append({
            "from_idx": 0,
            "to_idx": 1,
            "score_value": 0.0,
            "token_type": "none"
        })

    return {
        "outcome_score": outcome_score,
        "process_scores": process_scores
    }