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

import inspect
from typing import Any, Dict, List

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import default_compute_score


@register("naive_process")
class NaiveProcessRewardManager(RewardLoopManagerBase):
    """
    RewardLoopManager with process / slice-based reward support.

    Supported compute_score return format:
    - dict: {"outcome_score": float, "process_scores": list[dict]}
    - float: fallback outcome/terminal reward

    reward_extra_info format:
    {
        "outcome_reward": float,
        "process_reward": list[dict],
    }
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> Dict[str, Any]:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        # --------------------
        # Extract valid response tokens
        # --------------------
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]

        # --------------------
        # Collect reward context
        # --------------------
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {}).copy()
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        # --------------------
        # Decode response (async-safe)
        # --------------------
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        # --------------------
        # Compute reward score (async or sync)
        # --------------------
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                tokenizer=self.tokenizer,
                total_len=valid_response_length,
                input_ids=valid_response_ids,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    tokenizer=self.tokenizer,
                    total_len=valid_response_length,
                    input_ids=valid_response_ids,
                    **extra_reward_kwargs,
                ),
            )

        # --------------------
        # Extract outcome and process reward
        # --------------------
        if isinstance(result, dict):
            outcome_reward = result.get("outcome_score", 0.0)
            process_reward = result.get("process_scores", [])
        else:
            # fallback: scalar reward
            outcome_reward = float(result)
            process_reward = []

        reward_extra_info = {
            "outcome_reward": outcome_reward,
            "process_reward": process_reward,
        }

        # --------------------
        # Return outcome_reward as reward_score (scalar) for compatibility
        # --------------------
        print(f"outcome_reward: {outcome_reward}")  
        print(f"process_reward: {process_reward}")
        return {
            "reward_score": outcome_reward,
            "reward_extra_info": reward_extra_info,
        }