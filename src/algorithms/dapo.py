import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import deepspeed
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer

from src.common.generation import (
    build_agentic_rag_model,
    compute_log_probabilities,
    generate_completions,
    save_lora_only_in_zero2,
)
from src.common.config import GenerationConfig
from src.common.decouple_layer import (
    create_action_token_mask,
    create_args_content_mask,
    generate_grad_control_dicts,
)
from src.common.reward.reward import overall as overall_reward
from src.common.reward.token import token_level as overall_reward_token_level


def compute_dapo_loss(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    ref_log_probs: torch.Tensor,
    curr_log_probs: torch.Tensor,
    beta: float,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    kl = torch.exp(ref_log_probs - curr_log_probs) - (ref_log_probs - curr_log_probs) - 1
    per_token = pg_losses - beta * kl
    masked_loss = per_token * completion_mask
    if loss_agg_mode == "token-mean":
        return masked_loss.sum() / completion_mask.sum()
    if loss_agg_mode == "seq-mean-token-sum":
        return masked_loss.sum(dim=-1).mean()
    if loss_agg_mode == "seq-mean-token-mean":
        seq_losses = masked_loss.sum(dim=-1) / completion_mask.sum(dim=-1).clamp(min=1)
        return seq_losses.mean()
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def filter_groups_by_metric(
    rollout_data: Dict[str, Any],
    metric_fn: Callable[[List[Any]], List[float]],
    num_generations: int,
) -> Tuple[Dict[str, Any], int]:
    metrics = metric_fn(rollout_data["formatted_completions"])
    batch_size = len(metrics)
    num_prompts = batch_size // num_generations
    valid_prompt_indices: List[int] = []
    for prompt_idx in range(num_prompts):
        start_idx = prompt_idx * num_generations
        end_idx = start_idx + num_generations
        group_metrics = metrics[start_idx:end_idx]
        if len(set(group_metrics)) > 1:
            valid_prompt_indices.append(prompt_idx)
    if not valid_prompt_indices:
        return rollout_data, 0
    valid_indices: List[int] = []
    for prompt_idx in valid_prompt_indices:
        start_idx = prompt_idx * num_generations
        end_idx = start_idx + num_generations
        valid_indices.extend(range(start_idx, end_idx))
    filtered: Dict[str, Any] = {}
    for key, value in rollout_data.items():
        if isinstance(value, torch.Tensor):
            filtered[key] = value[valid_indices]
        elif isinstance(value, list):
            filtered[key] = [value[i] for i in valid_indices]
        else:
            filtered[key] = value
    filtered["batch_size"] = len(valid_prompt_indices)
    return filtered, len(valid_prompt_indices)


def maximize_dapo_objective(
    config: Dict[str, Any],
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    rollout_data: Dict[str, Any],
    tokenizer: AutoTokenizer,
    reward_strategy,
    optimizer: torch.optim.Optimizer,
    beta: float,
    clip_ratio_low: float,
    clip_ratio_high: float,
    accelerator: Accelerator,
    loss_agg_mode: str = "token-mean",
    enable_overlong_penalty: bool = True,
    max_response_length: int = 20480,
    overlong_buffer_len: int = 4096,
    overlong_penalty_factor: float = 1.0,
) -> Tuple[float, float, Dict[str, Any]]:
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    old_lp = rollout_data["old_log_probs"]
    ref_lp = rollout_data["ref_log_probs"]
    k = rollout_data["logits_to_keep"]
    curr_lp = compute_log_probabilities(model, input_ids, attention_mask, k)
    ratio = torch.exp(curr_lp - old_lp)
    rewards_dict = reward_strategy.compute(
        prompts=rollout_data["repeated_prompts"],
        completions=rollout_data["formatted_completions"],
        answers=rollout_data["repeated_answers"],
    )
    rewards = torch.tensor(rewards_dict["total_scores"], dtype=torch.float32, device=curr_lp.device)
    if enable_overlong_penalty:
        completion_lengths = completion_mask.sum(dim=-1)
        expected_len = max_response_length - overlong_buffer_len
        exceed_len = completion_lengths - expected_len
        penalties = torch.clamp(-exceed_len / overlong_buffer_len * overlong_penalty_factor, max=0.0)
        rewards = rewards + penalties
    avg_reward = float(rewards.mean())
    num_generations = rewards.numel() // (rewards.shape[0]) if rewards.ndim == 1 else 1
    groups = rewards.view(-1, num_generations)
    means = groups.mean(dim=1)
    stds = groups.std(dim=1)
    mins = groups.min(dim=1).values
    maxs = groups.max(dim=1).values
    degenerate = (means == mins) | (means == maxs)
    exp_means = means.repeat_interleave(num_generations)
    exp_stds = stds.repeat_interleave(num_generations)
    mask = degenerate.repeat_interleave(num_generations)
    advantages = (rewards - exp_means) / (exp_stds + 1e-4)
    advantages[mask] = (torch.randint(0, 2, rewards.shape, device=rewards.device) * 2 - 1).float()[mask]
    advantages = advantages.unsqueeze(1)
    loss = compute_dapo_loss(
        ratio=ratio,
        advantages=advantages,
        completion_mask=completion_mask,
        ref_log_probs=ref_lp,
        curr_log_probs=curr_lp,
        beta=beta,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        loss_agg_mode=loss_agg_mode,
    )
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    return float(loss), avg_reward, rewards_dict


def train_with_layered_optimization_dapo(
    config,
    policy_model,
    ref_model,
    rollout,
    tokenizer,
    reward_function,
    optimizer,
    beta,
    clip_ratio_low,
    clip_ratio_high,
    accelerator,
    loss_agg_mode,
    enable_overlong_penalty,
    max_response_length,
    overlong_buffer_len,
    overlong_penalty_factor,
):
    action_train_dict, args_train_dict = generate_grad_control_dicts(policy_model)
    completion_ids = rollout["completion_ids"]
    completion_mask_copy = rollout["completion_mask"].clone()
    total_loss = 0.0
    rollout["completion_mask"] = create_action_token_mask(completion_ids, tokenizer, completion_mask_copy)
    if rollout["completion_mask"].sum() > 0:
        loss_val, avg_r, rdict = maximize_dapo_objective(
            config=config,
            model=policy_model,
            ref_model=ref_model,
            rollout_data=rollout,
            tokenizer=tokenizer,
            reward_function=reward_function,
            optimizer=optimizer,
            beta=beta,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            accelerator=accelerator,
            loss_agg_mode=loss_agg_mode,
            enable_overlong_penalty=enable_overlong_penalty,
            max_response_length=max_response_length,
            overlong_buffer_len=overlong_buffer_len,
            overlong_penalty_factor=overlong_penalty_factor,
        )
        total_loss += loss_val
    rollout["completion_mask"] = create_args_content_mask(completion_ids, tokenizer, completion_mask_copy)
    loss_val, avg_r, rdict = maximize_dapo_objective(
        config=config,
        model=policy_model,
        ref_model=ref_model,
        rollout_data=rollout,
        tokenizer=tokenizer,
        reward_function=reward_function,
        optimizer=optimizer,
        beta=beta,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        accelerator=accelerator,
        loss_agg_mode=loss_agg_mode,
        enable_overlong_penalty=enable_overlong_penalty,
        max_response_length=max_response_length,
        overlong_buffer_len=overlong_buffer_len,
        overlong_penalty_factor=overlong_penalty_factor,
    )
    total_loss += loss_val
    rollout["completion_mask"] = completion_mask_copy
    return total_loss, avg_r, rdict


def train_with_dapo(
    config: Dict[str, Any],
    device: torch.device,
    policy_model: torch.nn.Module,
    ref_base_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    accelerator: Optional[Accelerator] = None,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    num_iterations: int = 1,
    steps_per_iteration: int = 500,
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    mu: int = 1,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
    loss_agg_mode: str = "token-mean",
    enable_dynamic_sampling: bool = True,
    gen_batch_size: Optional[int] = None,
    train_batch_size: Optional[int] = None,
    max_num_gen_batches: int = 10,
    filter_metric: str = "acc",
    enable_overlong_penalty: bool = False,
    max_response_length: int = 2048,
    overlong_buffer_len: int = 512,
    overlong_penalty_factor: float = 1.0,
    reward_strategy=None,
    checkpoint_dir: Optional[str] = None,
    current_step: int = 0,
    save_interval: int = 5,
    gen_cfg: GenerationConfig | None = None,
    metrics_path: str | None = None,
    use_decouple_layer: bool = False,
    use_SSRL: bool = False,
) -> None:
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    policy_model.train()
    policy_model, optimizer, dataloader = accelerator.prepare(policy_model, optimizer, dataloader)
    zero_stage = policy_model.config["zero_optimization"]["stage"]
    sum_steps = current_step
    all_steps = num_iterations * steps_per_iteration
    for it in range(1, num_iterations + 1):
        torch.cuda.empty_cache()
        ref_model = build_agentic_rag_model(config, device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        lora_params = [p for n, p in policy_model.named_parameters() if "lora" in n]
        with deepspeed.zero.GatheredParameters(lora_params, enabled=True):
            sd = policy_model.state_dict()
            lora_sd = {k: v for k, v in sd.items() if "lora" in k}
            ref_model.load_state_dict(lora_sd, strict=False)
            ref_model.to(accelerator.device)
        if zero_stage != 2:
            ref_model = accelerator.prepare(ref_model)
        step = 0
        for batch in dataloader:
            num_g_cfg = gen_cfg.num_generations if gen_cfg else {"mode": "constant", "constant": 1}
            if isinstance(num_g_cfg, dict):
                mode = num_g_cfg.get("mode", "constant")
                if mode == "constant":
                    num_g = int(num_g_cfg.get("constant", 1))
                elif mode == "range":
                    num_g = random.randint(num_g_cfg["range"][0], num_g_cfg["range"][1])
                elif mode == "function":
                    num_g = int(8 - (sum_steps / all_steps) * 4)
                else:
                    num_g = int(num_g_cfg.get("constant", 1))
            else:
                num_g = int(num_g_cfg)
            with torch.no_grad():
                rollout = generate_rollout_data(
                    policy_model,
                    ref_model,
                    tokenizer,
                    batch,
                    gen_cfg,
                    use_SSRL,
                )
            for _ in range(mu):
                if not use_decouple_layer:
                    loss_val, avg_r, rdict = maximize_dapo_objective(
                        config=config,
                        model=policy_model,
                        ref_model=ref_model,
                        rollout_data=rollout,
                        tokenizer=tokenizer,
                        reward_strategy=reward_strategy,
                        optimizer=optimizer,
                        beta=beta,
                        clip_ratio_low=clip_ratio_low,
                        clip_ratio_high=clip_ratio_high,
                        accelerator=accelerator,
                        loss_agg_mode=loss_agg_mode,
                        enable_overlong_penalty=enable_overlong_penalty,
                        max_response_length=max_response_length,
                        overlong_buffer_len=overlong_buffer_len,
                        overlong_penalty_factor=overlong_penalty_factor,
                    )
                else:
                    loss_val, avg_r, rdict = train_with_layered_optimization_dapo(
                        config=config,
                        policy_model=policy_model,
                        ref_model=ref_model,
                        rollout=rollout,
                        tokenizer=tokenizer,
                        reward_function=reward_strategy.compute,
                        optimizer=optimizer,
                        beta=beta,
                        clip_ratio_low=clip_ratio_low,
                        clip_ratio_high=clip_ratio_high,
                        accelerator=accelerator,
                        loss_agg_mode=loss_agg_mode,
                        enable_overlong_penalty=enable_overlong_penalty,
                        max_response_length=max_response_length,
                        overlong_buffer_len=overlong_buffer_len,
                        overlong_penalty_factor=overlong_penalty_factor,
                    )
            sum_steps += 1
            step += 1
            if metrics_path:
                try:
                    with open(metrics_path, "a") as mf:
                        mf.write(
                            f"{{\"iteration\":{it},\"step\":{step},\"loss\":{loss_val:.6f},\"avg_reward\":{avg_r:.4f}}}\n"
                        )
                except Exception:
                    pass
            if sum_steps % save_interval == 0 and sum_steps > current_step:
                if accelerator.is_local_main_process:
                    ckpt = f"{checkpoint_dir}/step-{sum_steps:04d}"
                    os.makedirs(ckpt, exist_ok=True)
                    zero_stage_cur = policy_model.config["zero_optimization"]["stage"] if hasattr(policy_model, "config") else 3
                    if zero_stage_cur == 2:
                        save_lora_only_in_zero2(policy_model, tokenizer, ckpt, accelerator)
                    else:
                        policy_model.save_pretrained(ckpt)
                        tokenizer.save_pretrained(ckpt)
            if step >= steps_per_iteration:
                break
            accelerator.wait_for_everyone()
        del ref_model
        torch.cuda.empty_cache()
