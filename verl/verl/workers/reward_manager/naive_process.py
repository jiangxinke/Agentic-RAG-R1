from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase


@register("naive_process")
class NaiveProcessRewardManager(AbstractRewardManager):
    """Reward manager with process (slice-based) reward support."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        print("Calculate Process Reward")
        # exit()
        # 1. Directly return rm_scores if provided
        if "rm_scores" in data.batch:
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": {}}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            item = data[i]

            # -------- prompt / response extraction --------
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = item.batch["attention_mask"][:prompt_len].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            response_ids = item.batch["responses"]
            valid_resp_len = item.batch["attention_mask"][prompt_len:].sum()
            valid_resp_ids = response_ids[:valid_resp_len]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)

            # -------- meta info --------
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch[self.reward_fn_key]
            extra_info = item.non_tensor_batch.get("extra_info", {})

            # -------- compute score --------
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                tokenizer=self.tokenizer,
                total_len=int(valid_resp_len),
                input_ids=valid_resp_ids,
            )

            # -------- helper: assign slice reward --------
            def assign_slice(score_value, from_idx, to_idx):
                # clamp
                from_idx = max(0, min(from_idx, valid_resp_len - 1))
                to_idx = max(from_idx, min(to_idx, valid_resp_len - 1))

                span_len = to_idx - from_idx + 1
                reward_tensor[i, from_idx:to_idx + 1] += score_value / span_len

                reward_extra_info["from_idx"].append(from_idx)
                reward_extra_info["to_idx"].append(to_idx)
                reward_extra_info["score_value"].append(score_value)

            # -------- process reward formats --------
            if isinstance(score, list):
                for s in score:
                    score_value = s["score_value"]
                    from_idx = s["from_idx"]
                    to_idx = s["to_idx"]

                    # EOS must be single-token
                    if s.get("token_type") == "eos":
                        to_idx = from_idx

                    assign_slice(score_value, from_idx, to_idx)

            elif isinstance(score, dict) and "score_value" in score:
                score_value = score["score_value"]
                from_idx = score["from_idx"]
                to_idx = score["to_idx"]

                if score.get("token_type") == "eos":
                    to_idx = from_idx

                assign_slice(score_value, from_idx, to_idx)

            else:
                # fallback: scalar reward → last valid token
                pos = valid_resp_len - 1 if valid_resp_len > 0 else 0
                reward_tensor[i, pos] = float(score)
                reward_extra_info["score"].append(score)

            # -------- debug printing --------
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n[Sample {i}] ({data_source})")
                print("[Prompt]", prompt_str)
                print("[Response]", response_str)
                print("[GT]", ground_truth)
                print("[Score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
