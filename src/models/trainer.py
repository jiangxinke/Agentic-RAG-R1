import copy
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import deepspeed
import swanlab
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from deepspeed import DeepSpeedEngine
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.models.model import AgenticRAGModel
from src.models.reward import overall_reward
from src.utils.extractor import analyze_completions
from src.utils.utils import optimize_model_memory


def create_completion_mask(
    completion_ids: torch.Tensor,
    eos_token_id: int,
    observation_start_ids: List[int],
    observation_end_ids: List[int],
) -> torch.Tensor:
    """
    Generate a mask for valid completion tokens, excluding tokens within observation spans.

    This function identifies tokens up to the first EOS (inclusive) and filters out any
    tokens occurring between observation start and end markers.

    Args:
        completion_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) with token IDs.
        eos_token_id (int): ID of the end-of-sequence token.
        observation_start_ids (List[int]): Sequence of token IDs marking the start of an observation.
        observation_end_ids (List[int]): Sequence of token IDs marking the end of an observation.

    Returns:
        torch.Tensor: Binary mask of shape (batch_size, seq_len), where 1 indicates tokens
            to keep and 0 indicates tokens to mask out.

    Raises:
        ValueError: If observation start/end lists are empty or longer than sequence length.
    """
    batch_size, seq_len = completion_ids.shape

    # 1) Build mask up to and including first EOS
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((batch_size,), seq_len, dtype=torch.long, device=completion_ids.device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    seq_indices = torch.arange(seq_len, device=completion_ids.device).unsqueeze(0).expand(batch_size, -1)
    completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

    # 2) Locate observation start/end positions via sliding window
    if not observation_start_ids or not observation_end_ids:
        raise ValueError("Observation start/end ID lists must be non-empty.")
    obs_start_len = len(observation_start_ids)
    obs_end_len = len(observation_end_ids)
    if obs_start_len > seq_len or obs_end_len > seq_len:
        raise ValueError("Observation marker length exceeds sequence length.")

    is_obs_start = torch.zeros_like(completion_ids, dtype=torch.bool)
    is_obs_end = torch.zeros_like(completion_ids, dtype=torch.bool)
    start_tensor = torch.tensor(observation_start_ids, device=completion_ids.device)
    end_tensor = torch.tensor(observation_end_ids, device=completion_ids.device)

    for b in range(batch_size):
        for i in range(seq_len - obs_start_len + 1):
            if torch.all(completion_ids[b, i : i + obs_start_len] == start_tensor):
                is_obs_start[b, i] = True
        for i in range(seq_len - obs_end_len + 1):
            if torch.all(completion_ids[b, i : i + obs_end_len] == end_tensor):
                is_obs_end[b, i] = True

    # 3) Build flag for tokens within observation spans
    observation_flag = torch.zeros_like(completion_mask, dtype=torch.int)
    for b in range(batch_size):
        in_obs = False
        for i in range(seq_len):
            if is_obs_start[b, i]:
                in_obs = True
            if in_obs:
                observation_flag[b, i] = 1
            if is_obs_end[b, i]:
                in_obs = False

    # 4) Combine masks: keep only completion_mask positions outside observation spans
    final_mask = completion_mask & (1 - observation_flag)
    return final_mask


def generate_completions(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    num_generations: int = 4,
    max_new_tokens: int = 128,
    max_length_for_gather: int = 2000,
    temperature: float = 0.7,
    do_sample: bool = True,
    max_generate_iterations: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multiple completions per prompt and compute masks for valid tokens.

    Tokenizes prompts, repeats each for num_generations, generates new tokens,
    and applies create_completion_mask to exclude observation spans.

    Args:
        model (torch.nn.Module): Language model with generate-like interface.
        tokenizer (AutoTokenizer): Corresponding tokenizer.
        prompts (List[str]): List of input prompt strings.
        num_generations (int): Number of completions per prompt.
        max_new_tokens (int): Maximum tokens to generate.
        max_length_for_gather (int): Max total length including prompt.
        temperature (float): Sampling temperature.
        do_sample (bool): Whether to sample (True) or use greedy (False).
        max_generate_iterations (int): Maximum generate iterations.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - prompt_ids: shape (batch_size*num_generations, prompt_len)
            - prompt_mask: shape (batch_size*num_generations, prompt_len)
            - completion_ids: shape (batch_size*num_generations, completion_len)
            - completion_mask: binary mask of same shape

    Raises:
        RuntimeError: On generation failure or device mismatch.
    """
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"

    # Tokenize and prepare inputs
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    prompt_len = prompt_ids.size(1)

    # Repeat for multiple generations
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

    # Generate sequences
    outputs = model(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_new_tokens,
        max_length_for_gather=max_length_for_gather,
        do_sample=do_sample,
        temperature=temperature,
        max_generate_iterations=max_generate_iterations,
    )
    completion_ids = outputs[:, prompt_len:]

    # Compute mask
    start_ids = tokenizer("<observation>").input_ids
    end_ids = tokenizer("</observation>").input_ids
    completion_mask = create_completion_mask(
        completion_ids,
        tokenizer.eos_token_id,
        start_ids,
        end_ids,
    )
    return prompt_ids, prompt_mask, completion_ids, completion_mask


def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute log probabilities only for specified token IDs.

    Args:
        logits (torch.Tensor): Raw model logits (batch, seq_len, vocab_size).
        input_ids (torch.Tensor): Token IDs to select (batch, seq_len).

    Returns:
        torch.Tensor: Log probabilities for each input_id (batch, seq_len).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))
    return selected.squeeze(-1)


def compute_log_probabilities(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    """
    Compute log probabilities for the last logits_to_keep tokens.

    Args:
        model (torch.nn.Module): Model supporting logits_to_keep & obtain_logits flags.
        input_ids (torch.Tensor): Combined prompt+completion IDs.
        attention_mask (torch.Tensor): Corresponding attention mask.
        logits_to_keep (int): Number of final tokens to evaluate.

    Returns:
        torch.Tensor: Log probabilities of shape (batch, logits_to_keep).

    Raises:
        RuntimeError: If model does not support required kwargs.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,
        obtain_logits=True,
    )
    logits = outputs[:, :-1, :]
    ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, ids)


def generate_rollout_data(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    batch_samples: Dict[str, List[Any]],
    num_generations: int,
    max_new_tokens: int,
    max_length_for_gather: int,
    temperature: float,
    do_sample: bool,
    max_generate_iterations: int,
) -> Dict[str, Any]:
    """
    Generate completions and compute log-probabilities for rollouts.

    Args:
        model (torch.nn.Module): Current policy model.
        ref_model (torch.nn.Module): Reference (static) model.
        tokenizer (AutoTokenizer): Tokenizer for decoding.
        batch_samples (Dict[str, List[Any]]): Contains "prompt", "question", "answer" lists.
        num_generations (int): Completions per prompt.
        max_new_tokens (int): Maximum new tokens.
        max_length_for_gather (int): Maximum total length.
        temperature (float): Sampling temperature.
        do_sample (bool): Sampling flag.
        max_generate_iterations (int): Maximum generate iterations.
    Returns:
        Dict[str, Any]: Rollout data including IDs, masks, log-probs, completions, etc.
    """
    device = next(model.parameters()).device
    prompts = batch_samples["prompt"]
    answers = batch_samples["answer"]

    with torch.no_grad():
        p_ids, p_mask, c_ids, c_mask = generate_completions(
            model,
            tokenizer,
            prompts,
            num_generations,
            max_new_tokens,
            max_length_for_gather,
            temperature,
            do_sample,
            max_generate_iterations,
        )
        input_ids = torch.cat([p_ids, c_ids], dim=1)
        attention_mask = torch.cat([p_mask, c_mask], dim=1)
        k = c_ids.size(1)

        old_log_probs = compute_log_probabilities(model, input_ids, attention_mask, k)
        ref_log_probs = compute_log_probabilities(ref_model, input_ids, attention_mask, k)

    formatted = [[{"content": tokenizer.decode(ids, skip_special_tokens=True)}] for ids in c_ids]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": c_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": k,
        "batch_size": len(prompts),
        "num_generations": num_generations,
    }


def compute_group_relative_advantages(
    rewards: torch.Tensor,
    num_generations: int,
) -> torch.Tensor:
    """
    Normalize rewards within each prompt group and handle degenerate cases.

    Args:
        rewards (torch.Tensor): Flat tensor of rewards (batch*num_gen,).
        num_generations (int): Number of completions per prompt.

    Returns:
        torch.Tensor: Advantages of shape (batch*num_gen, 1).
    """
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
    # Random ±1 for degenerate groups
    rand = (torch.randint(0, 2, rewards.shape, device=rewards.device) * 2 - 1).float()
    adv[mask] = rand[mask]
    return adv.unsqueeze(1)


def maximize_grpo_objective(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    rollout_data: Dict[str, Any],
    tokenizer: AutoTokenizer,
    reward_function: Callable[..., Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    beta: float,
    epsilon: float,
    accelerator: Accelerator,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Perform a single GRPO update step, computing loss and backpropagating.

    Args:
        model (torch.nn.Module): Policy model.
        ref_model (torch.nn.Module): Reference model.
        rollout_data (Dict[str, Any]): Output from generate_rollout_data.
        tokenizer (AutoTokenizer): For decoding completions.
        reward_function (Callable): Function to compute rewards.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        beta (float): KL penalty coefficient.
        epsilon (float): Clipping parameter.
        accelerator (Accelerator): For distributed training.

    Returns:
        Tuple[float, float, Dict[str, Any]]: Loss value, average reward, full reward dict.
    """
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    comp_mask = rollout_data["completion_mask"]
    old_lp = rollout_data["old_log_probs"]
    ref_lp = rollout_data["ref_log_probs"]
    k = rollout_data["logits_to_keep"]

    # Current policy log probs
    curr_lp = compute_log_probabilities(model, input_ids, attention_mask, k)
    ratio = torch.exp(curr_lp - old_lp)

    rewards_dict = reward_function(
        prompts=rollout_data["repeated_prompts"],
        completions=rollout_data["formatted_completions"],
        answers=rollout_data["repeated_answers"],
    )
    rewards = torch.tensor(rewards_dict["total_scores"], dtype=torch.float32, device=curr_lp.device)
    avg_reward = float(rewards.mean())

    adv = compute_group_relative_advantages(rewards, rollout_data["num_generations"])
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
    surr = torch.min(surr1, surr2)

    kl = torch.exp(ref_lp - curr_lp) - (ref_lp - curr_lp) - 1
    per_token = surr - beta * kl
    loss = -((per_token * comp_mask).sum(dim=1) / comp_mask.sum(dim=1)).mean()

    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    return float(loss), avg_reward, rewards_dict


from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.models.model import AgenticRAGModel
from src.utils.utils import optimize_model_memory


def build_agentic_rag_model(
    config,
    device: torch.device,
) -> AgenticRAGModel:
    """
    Build and return an AgenticRAGModel based on the provided config and device.
    This function handles tokenizer loading, (Q)LoRA application, and memory optimization.
    """
    continue_training = config.training.continue_training
    checkpoint_step = config.training.current_step
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    # Quantization config (if using QLoRA)
    if config.training.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.qlora.load_in_4bit,
            bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
            bnb_4bit_quant_storage=getattr(torch, config.qlora.bnb_4bit_quant_storage),
        )
    else:
        bnb_config = None
    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        trust_remote_code=True,
        quantization_config=bnb_config,
    ).to(device)
    # Apply LoRA if needed
    if config.training.use_lora:
        lora_cfg = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )
        if continue_training:
            weights_path = f"checkpoints/{config.experiment.name}/step-{checkpoint_step:04d}"
            base = PeftModel.from_pretrained(base, weights_path, config=lora_cfg, is_trainable=True)
        else:
            base = get_peft_model(base, lora_cfg)
    # Optimize memory usage
    base = optimize_model_memory(base)
    # Wrap and return AgenticRAGModel
    return AgenticRAGModel(base, tokenizer)


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
    num_generations: int = 4,
    max_new_tokens: int = 128,
    max_length_for_gather: int = 2000,
    max_generate_iterations: int = 8,
    temperature: float = 0.7,
    do_sample: bool = True,
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    mu: int = 1,
    epsilon: float = 0.2,
    reward_function: Callable[..., Dict[str, Any]] = overall_reward,
    checkpoint_dir: Optional[str] = None,
    current_step: int = 0,
    save_interval: int = 5,
) -> None:
    """
    Train policy model using GRPO fine-tuning with periodic checkpointing.

    Args:
        policy_model (torch.nn.Module): The policy network.
        base_reference_model (torch.nn.Module): Base model for reference rollouts.
        tokenizer (AutoTokenizer): Tokenizer for data processing.
        accelerator (Optional[Accelerator]): Accelerator for distributed training.
        dataloader (Optional[DataLoader]): Training data loader.
        num_iterations (int): Number of outer iterations.
        steps_per_iteration (int): Max steps per iteration.
        num_generations (int): Completions per prompt.
        max_new_tokens (int): Max tokens to generate.
        max_length_for_gather (int): Max sequence length.
        temperature (float): Sampling temperature.
        do_sample (bool): Whether to sample or greedy.
        beta (float): KL penalty coefficient.
        learning_rate (float): Optimizer learning rate.
        mu (int): GRPO updates per batch.
        epsilon (float): Clipping epsilon.
        reward_function (Callable): Reward computation function.
        checkpoint_dir (Optional[str]): Path to save checkpoints.
        current_step (int): Starting training step.
        save_interval (int): Steps between saves.

    Raises:
        RuntimeError: On training failures or save errors.
    """
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    policy_model.train()
    policy_model, optimizer, dataloader = accelerator.prepare(policy_model, optimizer, dataloader)

    sum_steps = current_step
    for it in range(1, num_iterations + 1):
        logging.info(f"start GRPO iteration {it}/{num_iterations}")
        torch.cuda.empty_cache()

        ref_model = build_agentic_rag_model(config, device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        # Sync LoRA weights to reference
        lora_params = [p for n, p in policy_model.named_parameters() if "lora" in n]
        with deepspeed.zero.GatheredParameters(lora_params, enabled=True):
            sd = policy_model.state_dict()
            lora_sd = {k: v for k, v in sd.items() if "lora" in k}
            ref_model.load_state_dict(lora_sd, strict=False)
            ref_model.to(accelerator.device)
        ref_model = accelerator.prepare(ref_model)


        step = 0
        for batch in dataloader:
            logging.info(f"start to generate rollout data, step {step+1}/{min(steps_per_iteration, len(dataloader))}")
            with torch.no_grad():
                rollout = generate_rollout_data(
                    policy_model,
                    ref_model,
                    tokenizer,
                    batch,
                    num_generations,
                    max_new_tokens,
                    max_length_for_gather,
                    temperature,
                    do_sample,
                    max_generate_iterations,
                )
            logging.info(f"success to generate rollout data")
            for _ in range(mu):
                loss_val, avg_r, rdict = maximize_grpo_objective(
                    policy_model, ref_model, rollout, tokenizer, reward_function, optimizer, beta, epsilon, accelerator
                )
            logging.info(f"success to maximize grpo objective")

            print(
                f"Iteration {it}/{num_iterations}, Step {step+1}/{min(steps_per_iteration, len(dataloader))}, "
                f"Loss: {loss_val:.6f}, Avg Reward: {avg_r:.2f}"
            )
            if accelerator.is_local_main_process:
                swanlab.log(
                    {
                        "Iteration": it,
                        "Step": step+1,
                        "Loss": loss_val,
                        "Avg Reward": avg_r,
                    }
                )

            # Logging and checkpointing omitted for brevity
            sum_steps += 1
            step += 1
            if sum_steps % save_interval == 0 and sum_steps > current_step:
                if accelerator.is_local_main_process:
                    ckpt = f"{checkpoint_dir}/step-{sum_steps:04d}"
                    os.makedirs(ckpt, exist_ok=True)
                    policy_model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
            if step >= steps_per_iteration:
                break

            accelerator.wait_for_everyone()

        del ref_model
        torch.cuda.empty_cache()
