# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.query_rewrite import replace_last_element_by_gt_if_low_score
from verl.workers.reward_manager import register


@register("naive_process")
class NaiveProcessRewardManager:
    """The reward manager with slice-based reward assignment support."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
                           Should return a list of dicts with keys: "score_value", "from_idx", "to_idx" (and optional "token_type").
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        self.new_ids_for_replacement = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode token IDs to text
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]  # Remove EOS token from response

            response_tokens = self.tokenizer.convert_ids_to_tokens(valid_response_ids)
            token_to_id = {token: id for token, id in zip(response_tokens, valid_response_ids)}

            # Get auxiliary data for reward calculation
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # Compute reward score (supports list of dicts with slice info)
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                tokenizer=self.tokenizer,
                response_id_mapping=token_to_id,
                total_len=valid_response_length.item(),
            )
                
            # Handle list of dicts (multiple slice regions for reward assignment)
            if isinstance(score, list) and all(
                isinstance(s, dict) and 
                "score_value" in s and 
                "from_idx" in s and 
                "to_idx" in s 
                for s in score
            ):  
                # print(len(score))
                for score_dict in score:
                    # Step 1: Filter abnormal EOS slice (EOS should be single token)
                    if "token_type" in score_dict and score_dict["token_type"] == "eos":
                        slice_len = score_dict["to_idx"] - score_dict["from_idx"]
                        if slice_len > 1:
                            print(f"[Warning] Sample {i} - EOS slice too long (len={slice_len}), force to single token")
                            # Force EOS slice to 1 token (keep from_idx, set to_idx=from_idx)
                            score_dict["to_idx"] = score_dict["from_idx"]

                    # Step 2: Extract and correct slice indices (ensure within valid range)
                    score_value = score_dict["score_value"]
                    from_idx = score_dict["from_idx"]
                    to_idx = score_dict["to_idx"]

                    # Correct invalid indices (prevent out of bounds)
                    if not (0 <= from_idx <= to_idx <= valid_response_length):
                        print(f"[Warning] Sample {i} - Invalid slice: from={from_idx}, to={to_idx}, valid_len={valid_response_length}, auto-correcting")
                        from_idx = max(0, min(from_idx, valid_response_length - 1))  # Clamp to [0, valid_len-1]
                        to_idx = max(from_idx, min(to_idx, valid_response_length - 1))  # Ensure to_idx >= from_idx
                        # Update corrected indices back to score_dict (for consistent logging)
                        score_dict["from_idx"] = from_idx
                        score_dict["to_idx"] = to_idx

                    # Step 3: Final validation (ensure no invalid slice after correction)
                    assert 0 <= from_idx <= to_idx <= valid_response_length, \
                        f"Sample {i} - Corrected slice still invalid: from={from_idx}, to={to_idx}, valid_len={valid_response_length}"

                    # Step 4: Assign distributed reward to the slice
                    span_len = to_idx - from_idx + 1
                    distributed_score = score_value / span_len
                    reward_tensor[i, from_idx:to_idx+1] += distributed_score

                    # Step 5: Record valid extra info (avoid abnormal data)
                    # reward_extra_info["score_value"].append(score_value)
                    reward_extra_info["from_idx"].append(from_idx)
                    reward_extra_info["to_idx"].append(to_idx)

            # Compatibility: handle single dict (one slice region)
            elif isinstance(score, dict):
                # Step 1: Filter abnormal EOS slice (same as list case)
                if "token_type" in score and score["token_type"] == "eos":
                    slice_len = score["to_idx"] - score["from_idx"]
                    if slice_len > 1:
                        print(f"[Warning] Sample {i} - EOS slice too long (len={slice_len}), force to single token")
                        score["to_idx"] = score["from_idx"]

                # Step 2: Extract and correct slice indices
                score_value = score["score_value"]
                from_idx = score["from_idx"]
                to_idx = score["to_idx"]

                # Correct invalid indices
                if not (0 <= from_idx <= to_idx <= valid_response_length):
                    print(f"[Warning] Sample {i} - Invalid slice: from={from_idx}, to={to_idx}, valid_len={valid_response_length}, auto-correcting")
                    from_idx = max(0, min(from_idx, valid_response_length - 1))
                    to_idx = max(from_idx, min(to_idx, valid_response_length - 1))
                    score["from_idx"] = from_idx
                    score["to_idx"] = to_idx

                # Step 3: Final validation
                assert 0 <= from_idx <= to_idx <= valid_response_length, \
                    f"Sample {i} - Corrected slice still invalid: from={from_idx}, to={to_idx}, valid_len={valid_response_length}"

                # Step 4: Assign distributed reward
                span_len = to_idx - from_idx + 1
                distributed_score = score_value / span_len
                reward_tensor[i, from_idx:to_idx+1] += distributed_score

                # Step 5: Record extra info
                # reward_extra_info["score_value"].append(score_value)
                reward_extra_info["from_idx"].append(from_idx)
                reward_extra_info["to_idx"].append(to_idx)
                if "token_type" in score:
                    reward_extra_info["token_type"].append(score["token_type"])

            # Compatibility: handle original scalar reward (assign to last valid token)
            else:
                reward = score
                # Ensure we assign to valid position (avoid index out of bounds)
                assign_pos = valid_response_length - 1 if valid_response_length > 0 else 0
                reward_tensor[i, assign_pos] = reward
                reward_extra_info["score"].append(reward)

            # Debug printing (limit by num_examine per data source)
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n[Sample {i}] [Data Source: {data_source}]")
                print(f"[Prompt] {prompt_str[:100]}..." if len(prompt_str) > 100 else f"[Prompt] {prompt_str}")
                print(f"[Response] {response_str[:100]}..." if len(response_str) > 100 else f"[Response] {response_str}")
                print(f"[Ground Truth] {ground_truth[:100]}..." if len(ground_truth) > 100 else f"[Ground Truth] {ground_truth}")
                
                if isinstance(score, list):
                    for idx, s in enumerate(score):
                        print(f"[Slice {idx}] score={s['score_value']}, range=({s['from_idx']}, {s['to_idx']})" + 
                            (f", type={s['token_type']}" if "token_type" in s else ""))
                elif isinstance(score, dict):
                    print(f"[Score] value={score['score_value']}, range=({score['from_idx']}, {score['to_idx']})" + 
                        (f", type={score['token_type']}" if "token_type" in score else ""))
                else:
                    print(f"[Score] {score} (assigned to position {assign_pos})")

        if return_dict:
            test_reward_extra_info = defaultdict(list)
            test_reward_extra_info["reward_tensor"]=reward_tensor
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": test_reward_extra_info, # FIXME
            }
        else:
            return reward_tensor