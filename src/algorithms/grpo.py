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


def generate_rollout_data(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    batch_samples: Dict[str, List[Any]],
    gen_cfg: GenerationConfig,
    use_SSRL: bool,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    prompts = batch_samples["prompt"]
    answers = batch_samples["answer"]
    with torch.no_grad():
        p_ids, p_mask, c_ids, c_mask = generate_completions(
            model,
            tokenizer,
            prompts,
            gen_cfg,
            use_SSRL,
        )
        input_ids = torch.cat([p_ids, c_ids], dim=1)
        attention_mask = torch.cat([p_mask, c_mask], dim=1)
        k = c_ids.size(1)
        old_log_probs = compute_log_probabilities(model, input_ids, attention_mask, k)
        ref_model.masked_spans_per_sample, ref_model.masked_parellel_spans_per_sample = (
            model.masked_spans_per_sample,
            model.masked_parellel_spans_per_sample,
        )
        ref_log_probs = compute_log_probabilities(ref_model, input_ids, attention_mask, k)
    formatted = [[{"content": tokenizer.decode(ids, skip_special_tokens=True)}] for ids in c_ids]
    repeated_prompts = [p for p in prompts for _ in range(gen_cfg.num_generations.get("constant", 1) if isinstance(gen_cfg.num_generations, dict) else int(gen_cfg.num_generations))]
    repeated_answers = [a for a in answers for _ in range(gen_cfg.num_generations.get("constant", 1) if isinstance(gen_cfg.num_generations, dict) else int(gen_cfg.num_generations))]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_ids": c_ids,
        "completion_mask": c_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": k,
        "batch_size": len(prompts),
    }


def compute_group_relative_advantages(
    config: Dict[str, Any],
    rewards: torch.Tensor,
) -> torch.Tensor:
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
    adv = (rewards - exp_means) / (exp_stds + 1e-4)
    rand = (torch.randint(0, 2, rewards.shape, device=rewards.device) * 2 - 1).float()
    adv[mask] = rand[mask]
    return adv.unsqueeze(1)


def maximize_grpo_objective(
    config: Dict[str, Any],
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    rollout_data: Dict[str, Any],
    tokenizer: AutoTokenizer,
    reward_strategy,
    optimizer: torch.optim.Optimizer,
    beta: float,
    epsilon: float,
    accelerator: Accelerator,
    grad_control_dict: Dict[str, bool] = None,
    no_backtrack: bool = False,
) -> Tuple[float, float, Dict[str, Any]]:
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_ids = rollout_data["completion_ids"]
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
    avg_reward = float(rewards.mean())
    adv = compute_group_relative_advantages(config, rewards)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + 1.5 * epsilon) * adv
    surr = torch.min(surr1, surr2)
    kl = torch.exp(ref_lp - curr_lp) - (ref_lp - curr_lp) - 1
    per_token = surr - beta * kl
    loss = -((per_token * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    if grad_control_dict is not None:
        original_grad_state = {name: p.requires_grad for name, p in model.named_parameters()}
        for name, param in model.named_parameters():
            param.requires_grad_(grad_control_dict.get(name, False))
    if not no_backtrack:
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    if grad_control_dict is not None:
        for name, param in model.named_parameters():
            param.requires_grad_(original_grad_state[name])
    return float(loss), avg_reward, rewards_dict


def train_with_layered_optimization(
    config,
    policy_model,
    ref_model,
    rollout,
    tokenizer,
    reward_function,
    optimizer,
    beta,
    epsilon,
    accelerator,
):
    action_train_dict, args_train_dict = generate_grad_control_dicts(policy_model)
    completion_ids = rollout["completion_ids"]
    completion_mask_copy = rollout["completion_mask"].clone()
    total_loss = 0.0
    rollout["completion_mask"] = create_action_token_mask(completion_ids, tokenizer, completion_mask_copy)
    if rollout["completion_mask"].sum() > 0:
        loss_val, avg_r, rdict = maximize_grpo_objective(
            config,
            policy_model,
            ref_model,
            rollout,
            tokenizer,
            reward_function,
            optimizer,
            beta,
            epsilon,
            accelerator,
            grad_control_dict=action_train_dict,
            no_backtrack=True,
        )
        total_loss += loss_val
    rollout["completion_mask"] = create_args_content_mask(completion_ids, tokenizer, completion_mask_copy)
    loss_val, avg_r, rdict = maximize_grpo_objective(
        config,
        policy_model,
        ref_model,
        rollout,
        tokenizer,
        reward_function,
        optimizer,
        beta,
        epsilon,
        accelerator,
        grad_control_dict=args_train_dict,
    )
    total_loss += loss_val
    rollout["completion_mask"] = completion_mask_copy
    return total_loss, avg_r, rdict


def train_with_grpo(
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
    epsilon: float = 0.2,
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
            logging.info("=" * 50)
            logging.info(f" step {step+1}/{min(steps_per_iteration, len(dataloader))}")
            logging.info("=" * 50)
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
                    loss_val, avg_r, rdict = maximize_grpo_objective(
                        config,
                        policy_model,
                        ref_model,
                        rollout,
                        tokenizer,
                        reward_strategy,
                        optimizer,
                        beta,
                        epsilon,
                        accelerator,
                    )
                else:
                    loss_val, avg_r, rdict = train_with_layered_optimization(
                        config,
                        policy_model,
                        ref_model,
                        rollout,
                        tokenizer,
                        reward_function,
                        optimizer,
                        beta,
                        epsilon,
                        accelerator,
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
                    if zero_stage == 2:
                        save_lora_only_in_zero2(policy_model, tokenizer, ckpt, accelerator)
                    else:
                        policy_model.save_pretrained(ckpt)
                        tokenizer.save_pretrained(ckpt)
            if step >= steps_per_iteration:
                break
            accelerator.wait_for_everyone()
        del ref_model
        torch.cuda.empty_cache()
