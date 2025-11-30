from typing import Callable

from src.common.reward_impl.overall import overall_reward as _overall_reward
from src.common.reward_impl.token_level import overall_reward_token_level as _overall_token_reward


def get_reward(name: str) -> Callable:
    if name in ("token", "token_level", "overall_token"):
        return _overall_token_reward
    if name in ("overall", "sequence"):
        return _overall_reward
    raise ValueError(f"Unknown reward name: {name}")


class RewardStrategy:
    def compute(self, **kwargs):
        raise NotImplementedError


class OverallRewardStrategy(RewardStrategy):
    def compute(self, **kwargs):
        return _overall_reward(
            prompts=kwargs.get("prompts"),
            completions=kwargs.get("completions"),
            answers=kwargs.get("answers"),
        )


class TokenLevelRewardStrategy(RewardStrategy):
    def compute(self, **kwargs):
        return _overall_token_reward(
            prompts=kwargs.get("prompts"),
            completions=kwargs.get("completions"),
            answers=kwargs.get("answers"),
            completion_ids=kwargs.get("completion_ids"),
            tokenizer=kwargs.get("tokenizer"),
        )


def get_reward_strategy(name: str) -> RewardStrategy:
    if name in ("token", "token_level", "overall_token"):
        return TokenLevelRewardStrategy()
    if name in ("overall", "sequence"):
        return OverallRewardStrategy()
    raise ValueError(f"Unknown reward strategy: {name}")
