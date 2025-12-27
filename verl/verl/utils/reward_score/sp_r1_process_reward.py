# verl/utils/reward_score/sp_r1_process_reward.py
import re
import random
from verl.utils.reward_score.search_r1_like_qa_em import compute_score as compute_outcome_score
import sys
import os

try:
    from spr1_special_tokens.recognition import TokenRecognizer
except ImportError:
    # Attempt to add project root to path if import fails
    # This file is in verl/utils/reward_score/
    # We want to reach the parent of verl package which contains spr1_special_tokens
    # ../../../../
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    try:
        from spr1_special_tokens.recognition import TokenRecognizer
    except ImportError:
        print("Warning: Could not import TokenRecognizer from spr1_special_tokens")
        TokenRecognizer = None

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
    outcome_score = compute_outcome_score(solution_str, ground_truth)

    # -------------------------
    # Process reward: 
    # -------------------------
    tokenizer = kwargs.get('tokenizer')
    input_ids = kwargs.get('input_ids')

    if tokenizer is not None and input_ids is not None:
        try:
            recognizer = TokenRecognizer(tokenizer=tokenizer)
            
            # 1. Reflect
            reflect_intervals = recognizer.find_intervals(input_ids, "<reflect>", "</reflect>", include_boundaries=True)
            for start, end in reflect_intervals:
                # TokenRecognizer returns [start, end), we want inclusive [start, end-1]
                process_scores.append({
                    "from_idx": start,
                    "to_idx": end - 1,
                    "score_value": reflect_reward if reflect_reward is not None else 0.0,
                    "token_type": "reflect"
                })
                
            # 2. Tool Call
            tool_intervals = recognizer.find_intervals(input_ids, "<tool_call>", "</tool_call>", include_boundaries=True)
            for start, end in tool_intervals:
                # content is between start and end (exclusive of tags)
                # interval is [start, end). start_token at start, end_token at end-1.
                # content: input_ids[start+1 : end-1]
                
                content_ids = input_ids[start+1 : end-1]
                if hasattr(content_ids, 'tolist'):
                    content_ids = content_ids.tolist()
                content = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
                
                try:
                    from verl.utils.reward_score.sp_r1.search_quality_score import RetrievalQualityEvaluator
                    import os
                    from dotenv import load_dotenv
                    from langchain_openai import ChatOpenAI
                    load_dotenv()
                    llm = ChatOpenAI(
                        model=os.getenv("EVAL_LLM_MODEL_NAME"),
                        base_url=os.getenv("EVAL_LLM_BASE_URL"),
                        api_key=os.getenv("EVAL_LLM_API_KEY"),
                    )
                    user_input = extra_info.get("question", "")
                    retrieved_contexts = [content]
                    rag_evaluator = RetrievalQualityEvaluator(llm)
                    evaluation_result = rag_evaluator.evaluate_retrieval(user_input, retrieved_contexts)
                    score_value = 1.0 * evaluation_result
                except Exception as e:
                    print(f"Error evaluating tool_call: {e}")
                    score_value = 0.0
                
                process_scores.append({
                    "from_idx": start,
                    "to_idx": end - 1,
                    "score_value": score_value if score_value is not None else 0.0,
                    "token_type": "tool_call"
                })
        except Exception as e:
             print(f"Error in TokenRecognizer: {e}, falling back to string regex.")
             pass
    
    if not (tokenizer is not None and input_ids is not None and TokenRecognizer is not None):
        for match in re.finditer(r"<reflect>(.*?)</reflect>", solution_str, re.DOTALL):
            start, end = match.span()
            process_scores.append({
                "from_idx": start,
                "to_idx": end - 1,
                "score_value": reflect_reward if reflect_reward is not None else 0.0,
                "token_type": "reflect"
            })

        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", solution_str, re.DOTALL):
            start, end = match.span()
            content = match.group(1).strip()

            try:
                from verl.utils.reward_score.sp_r1.search_quality_score import RetrievalQualityEvaluator
                import os
                from dotenv import load_dotenv
                from langchain_openai import ChatOpenAI
                load_dotenv()
                llm = ChatOpenAI(
                    model=os.getenv("EVAL_LLM_MODEL_NAME"),
                    base_url=os.getenv("EVAL_LLM_BASE_URL"),
                    api_key=os.getenv("EVAL_LLM_API_KEY"),
                )
                user_input = extra_info.get("question", "")
                retrieved_contexts = [content]
                rag_evaluator = RetrievalQualityEvaluator(llm)
                evaluation_result = rag_evaluator.evaluate_retrieval(user_input, retrieved_contexts)
                score_value = 1.0 * evaluation_result
            except Exception as e:
                print(f"Error evaluating tool_call: {e}")
                score_value = 0.0

            process_scores.append({
                "from_idx": start,
                "to_idx": end - 1,
                "score_value": score_value if score_value is not None else 0.0,
                "token_type": "tool_call"
            })

    return {
        "outcome_score": outcome_score,
        "process_scores": process_scores
    }