import os
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import AutoTokenizer

from src.common.generation import compute_log_probabilities, generate_completions
from src.common.config import GenerationConfig


def generate_rollout_data_ppo(
    policy_model: torch.nn.Module,
    critic_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    batch_samples: Dict[str, List[Any]],
    gen_cfg: GenerationConfig,
) -> Dict[str, Any]:
    prompts = batch_samples["prompt"]
    answers = batch_samples["answer"]
    with torch.no_grad():
        p_ids, p_mask, c_ids, c_mask = generate_completions(
            policy_model,
            tokenizer,
            prompts,
            gen_cfg,
        )
        input_ids = torch.cat([p_ids, c_ids], dim=1)
        attention_mask = torch.cat([p_mask, c_mask], dim=1)
        k = c_ids.size(1)
        old_log_probs = compute_log_probabilities(policy_model, input_ids, attention_mask, k)
        formatted = [[{"content": tokenizer.decode(ids, skip_special_tokens=True)}] for ids in c_ids]
        repeated_prompts = [p for p in prompts for _ in range(gen_cfg.num_generations.get("constant", 1) if isinstance(gen_cfg.num_generations, dict) else int(gen_cfg.num_generations))]
        repeated_answers = [a for a in answers for _ in range(gen_cfg.num_generations.get("constant", 1) if isinstance(gen_cfg.num_generations, dict) else int(gen_cfg.num_generations))]
        from src.common.reward.reward import overall as overall_reward
        rewards_dict = overall_reward(
            prompts=repeated_prompts,
            completions=formatted,
            answers=repeated_answers,
        )
        rewards = torch.tensor(rewards_dict["total_scores"], dtype=torch.float32, device=old_log_probs.device)
        values = critic_model(input_ids=input_ids, attention_mask=attention_mask).detach()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": c_mask,
        "old_log_probs": old_log_probs,
        "rewards": rewards,
        "values": values,
        "formatted_completions": formatted,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": k,
    }


def compute_advantages_ppo(rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    adv = rewards - values
    adv = adv.unsqueeze(1)
    return adv


def maximize_ppo_objective(
    policy_model: torch.nn.Module,
    rollout_data: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    epsilon: float,
    accelerator: Accelerator,
) -> float:
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    comp_mask = rollout_data["completion_mask"]
    old_lp = rollout_data["old_log_probs"]
    advantages = rollout_data["advantages"]
    k = rollout_data["logits_to_keep"]
    curr_lp = compute_log_probabilities(policy_model, input_ids, attention_mask, k)
    ratio = torch.exp(curr_lp - old_lp)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    loss = -((torch.min(surr1, surr2) * comp_mask).sum(dim=1) / comp_mask.sum(dim=1)).mean()
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    return float(loss)


def maximize_critic_objective(
    critic_model: torch.nn.Module,
    rollout_data: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
) -> float:
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    rewards = rollout_data["rewards"]
    values_pred = critic_model(input_ids=input_ids, attention_mask=attention_mask)
    rewards = rewards.to(values_pred.dtype)
    loss = nn.MSELoss()(values_pred, rewards)
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    return float(loss)


def train_with_ppo(
    config: Dict[str, Any],
    device: torch.device,
    policy_model: torch.nn.Module,
    critic_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    accelerator: Optional[Accelerator] = None,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    num_iterations: int = 1,
    steps_per_iteration: int = 500,
    learning_rate: float = 5e-6,
    critic_lr: float = 5e-6,
    epsilon: float = 0.2,
    checkpoint_dir: Optional[str] = None,
    current_step: int = 0,
    save_interval: int = 5,
    gen_cfg: GenerationConfig | None = None,
    metrics_path: str | None = None,
) -> None:
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=critic_lr)
    policy_model.train()
    critic_model.train()
    policy_model, policy_optimizer, dataloader = accelerator.prepare(policy_model, policy_optimizer, dataloader)
    critic_model, critic_optimizer = accelerator.prepare(critic_model, critic_optimizer)
    sum_steps = current_step
    for it in range(1, num_iterations + 1):
        torch.cuda.empty_cache()
        step = 0
        for batch in dataloader:
            rollout = generate_rollout_data_ppo(
                policy_model,
                critic_model,
                tokenizer,
                batch,
                gen_cfg,
            )
            rollout["advantages"] = compute_advantages_ppo(rollout["rewards"], rollout["values"])
            loss_policy = maximize_ppo_objective(policy_model, rollout, policy_optimizer, epsilon, accelerator)
            loss_critic = maximize_critic_objective(critic_model, rollout, critic_optimizer, accelerator)
            avg_reward = float(rollout["rewards"].mean())
            sum_steps += 1
            step += 1
            if metrics_path:
                try:
                    with open(metrics_path, "a") as mf:
                        mf.write(
                            f"{{\"iteration\":{it},\"step\":{step},\"policy_loss\":{loss_policy:.6f},\"critic_loss\":{loss_critic:.6f},\"avg_reward\":{avg_reward:.4f}}}\n"
                        )
                except Exception:
                    pass
            if sum_steps % save_interval == 0 and sum_steps > current_step:
                if accelerator.is_local_main_process:
                    ckpt = f"{checkpoint_dir}/step-{sum_steps:04d}"
                    os.makedirs(ckpt, exist_ok=True)
                    policy_model.save_pretrained(ckpt)
                    critic_model.module.save(ckpt + "_critic")
                    tokenizer.save_pretrained(ckpt)
            if step >= steps_per_iteration:
                break
            accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
