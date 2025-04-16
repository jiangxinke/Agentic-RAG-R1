import copy
import logging
import os
import pdb
import random
import re
import time

import deepspeed
import numpy as np
import swanlab
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from deepspeed import DeepSpeedEngine
from tqdm import tqdm
from trl import SFTTrainer

from model.reward_function import combined_reward
from utils.answer_extractor import extract_answer_from_model_output
from utils.protoco import DataProto
from utils.utils import print_memory_usage


def create_completion_mask(completion_ids, eos_token_id, observation_start_token_id, observation_end_token_id):
    """
    Create a binary mask for the generated completion tokens so that tokens after the first EOS or
    between <observation> and </observation> are ignored. The <observation> and </observation> tags
    and all tokens between them are included in the mask.

    Args:
        completion_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) with generated token ids.
        eos_token_id (int): The token id representing the end-of-sequence.
        observation_start_token_id (int): The token id representing the start of <observation>.
        observation_end_token_id (int): The token id representing the end of </observation>.

    Returns:
        torch.Tensor: A mask tensor of shape (batch_size, seq_len) with 1s for tokens up to and including the first EOS
                      and tokens between <observation> and </observation>, and 0s for tokens following the first EOS
                      or outside the <observation>... </observation> range.
    """
    # Determine positions of EOS, <observation>, and </observation> tokens.
    is_eos = completion_ids == eos_token_id
    is_observation_start = completion_ids == observation_start_token_id
    is_observation_end = completion_ids == observation_end_token_id

    # Initialize a tensor to store the index of the first EOS for each sequence.
    eos_idx = torch.full(
        (is_eos.size(0),),
        is_eos.size(1),
        dtype=torch.long,
        device=completion_ids.device,
    )

    # Identify sequences that contain at least one EOS.
    mask_exists = is_eos.any(dim=1)
    # For sequences with an EOS, update eos_idx to the index of the first occurrence.
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]

    # Create a tensor of indices [0, 1, 2, ..., seq_len-1] and replicate it for each sequence in the batch.
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)

    # Build the mask based on EOS.
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    # Create a mask for the <observation> and </observation> tokens, as well as everything between them.
    observation_range_mask = torch.zeros_like(completion_mask, dtype=torch.int)

    for batch_idx in range(completion_ids.size(0)):
        in_observation_range = False
        for seq_idx in range(completion_ids.size(1)):
            if is_observation_start[batch_idx, seq_idx]:
                in_observation_range = True
            if in_observation_range:
                observation_range_mask[batch_idx, seq_idx] = 0  # CHECK mask observation
            if is_observation_end[batch_idx, seq_idx]:
                in_observation_range = False

    # Combine the EOS mask and the <observation>... </observation> mask.
    final_mask = completion_mask | observation_range_mask

    return final_mask


def create_completion_mask(
    completion_ids: torch.Tensor, eos_token_id: int, observation_start_ids: list, observation_end_ids: list
):
    """
    根据给定的 completion_ids，生成一个 mask：
      - 在第一个 EOS 之后（不含 EOS 本身）全部置 0
      - <observation> ... </observation> 标签内内容全部置 0
      - 其余部分置 1

    Args:
        completion_ids (torch.Tensor): [batch_size, seq_len] 的模型生成结果 token id 序列
        eos_token_id (int): 结束符（EOS）的 token id
        observation_start_ids (list[int]): "<observation>" 对应的多个 token id
        observation_end_ids   (list[int]): "</observation>" 对应的多个 token id

    Returns:
        torch.Tensor: [batch_size, seq_len] 的 0/1 掩码张量
    """
    batch_size, seq_len = completion_ids.shape

    # 1) 找到第一个EOS的索引 => 构造 completion_mask
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full(
        (batch_size,),
        seq_len,  # 若没出现EOS就默认到结尾
        dtype=torch.long,
        device=completion_ids.device,
    )
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]

    # 构造 shape=[batch_size,seq_len] 的序号矩阵 (0,1,2,...,seq_len-1)
    seq_indices = torch.arange(seq_len, device=completion_ids.device).unsqueeze(0).expand(batch_size, -1)
    # <= eos_idx 的地方为1 (含EOS自身), 否则0
    completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

    # 2) 通过滑窗匹配找到 observation_start / observation_end
    is_observation_start = torch.zeros_like(completion_ids, dtype=torch.bool)
    is_observation_end = torch.zeros_like(completion_ids, dtype=torch.bool)

    obs_start_len = len(observation_start_ids)
    obs_end_len = len(observation_end_ids)

    for b_idx in range(batch_size):
        for i in range(seq_len - obs_start_len + 1):
            if torch.all(
                completion_ids[b_idx, i : i + obs_start_len] == torch.tensor(observation_start_ids, device=completion_ids.device)
            ):
                is_observation_start[b_idx, i] = True

        for i in range(seq_len - obs_end_len + 1):
            if torch.all(
                completion_ids[b_idx, i : i + obs_end_len] == torch.tensor(observation_end_ids, device=completion_ids.device)
            ):
                is_observation_end[b_idx, i] = True

    # 3) 构造 observation_flag: 在 <observation> ... </observation> 之内的内容置 1，否则 0
    observation_flag = torch.zeros_like(completion_mask, dtype=torch.int)
    for b_idx in range(batch_size):
        in_obs = False
        for i in range(seq_len):
            if is_observation_start[b_idx, i]:
                in_obs = True
            if in_obs:
                observation_flag[b_idx, i] = 1
            if is_observation_end[b_idx, i]:
                in_obs = False

    # 4) 最终 mask：保留 completion_mask（1）里、且不在 observation 区间内
    #    => final = completion_mask & (1 - observation_flag)
    final_mask = completion_mask & (1 - observation_flag)

    return final_mask


def generate_completions(
    model, tokenizer, prompts, num_generations=4, max_new_tokens=128, max_length_for_gather=2000, temperature=0.7, do_sample=True
):
    """
    Generate multiple completions for each prompt and create corresponding attention masks.

    Args:
        model: The language model used for generation.
        tokenizer: The tokenizer to process the prompts and decode the outputs.
        prompts (list of str): List of input prompt strings.
        num_generations (int): Number of completions to generate per prompt.
        max_new_tokens (int): Maximum number of new tokens to generate for the completion.
        max_length_for_gather (int): Maximum length of the input sequence.

    Returns:
        tuple: Contains the following tensors:
            - prompt_ids: (batch_size * num_generations, prompt_seq_len)
            - prompt_mask: (batch_size * num_generations, prompt_seq_len)
            - completion_ids: (batch_size * num_generations, completion_seq_len)
            - completion_mask: (batch_size * num_generations, completion_seq_len)

    Explanation:
        1. The prompts are tokenized and padded (with padding added to the left).
        2. Each prompt is repeated num_generations times so that multiple completions are generated per prompt.
        3. The model.generate() function is called to generate new tokens.
        4. The generated output contains the prompt followed by the completion; we remove the prompt part to get the completions.
        5. A mask is created (via create_completion_mask) so that only tokens up to the first EOS are considered.
    """
    device = next(model.parameters()).device

    # Tokenize the list of prompts with padding. The padding_side="left" ensures alignment on the right.
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)  # Shape: (batch_size, prompt_seq_len)
    prompt_mask = inputs["attention_mask"].to(device)  # Shape: (batch_size, prompt_seq_len)
    prompt_length = prompt_ids.size(1)  # Save the prompt length to later separate prompt from completion.

    # Repeat each prompt num_generations times.
    ## FIXME 旧版代码
    # New shape: (batch_size*num_generations, prompt_seq_len)
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    # New shape: (batch_size*num_generations, prompt_seq_len)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

    # pdb.set_trace()
    # Generate new tokens for each prompt. The output includes the original prompt and the generated tokens.
    outputs = model(  # old 方法是直接generate
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_new_tokens,
        max_length_for_gather=max_length_for_gather,
        do_sample=do_sample,
        temperature=temperature,
        # pad_token_id=tokenizer.pad_token_id,
        # eos_token_id=tokenizer.eos_token_id,
    )
    # Remove the prompt portion from the generated output to isolate the completion tokens.
    completion_ids = outputs[:, prompt_length:]  # Shape: (batch_size*num_generations, completion_seq_len)
    # pdb.set_trace()
    # pdb.set_trace()
    # Create a binary mask that ignores tokens beyond the first EOS token.
    # FIXME ?
    observation_start_token_id = tokenizer("<observation>").input_ids
    observation_end_token_id = tokenizer("</observation>").input_ids
    # completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    completion_mask = create_completion_mask(
        completion_ids,
        tokenizer.eos_token_id,
        observation_start_token_id,
        observation_end_token_id,
    )

    return prompt_ids, prompt_mask, completion_ids, completion_mask


def selective_log_softmax(logits, input_ids):
    """
    Compute the log probabilities for the tokens specified in input_ids using a selective log-softmax.

    Args:
        logits (torch.Tensor): A tensor of shape (batch_size, seq_len, vocab_size) containing raw logits from the model.
        input_ids (torch.Tensor): A tensor of shape (batch_size, seq_len) containing the token indices for which we want the log probabilities.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, seq_len) where each element is the log probability
                      corresponding to the token in input_ids at that position.

    Explanation:
        1. F.log_softmax is applied along the vocabulary dimension (dim=-1) to convert logits into log probabilities.
        2. The tensor input_ids is reshaped (via unsqueeze) to have an extra dimension so that we can use it as indices
           in the log_probs tensor.
        3. torch.gather collects the log probability at the index specified in input_ids for each position.
        4. Finally, squeeze(-1) removes the extra dimension, returning a tensor with the same shape as input_ids.
    """
    # FIXME use many memory usage
    # Convert raw logits into log probabilities along the vocabulary axis.
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)

    # Reshape input_ids from (batch_size, seq_len) to (batch_size, seq_len, 1) for gathering.
    # Then, gather the log probability for each token in input_ids.
    selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))

    # Remove the extra last dimension to get back to shape (batch_size, seq_len).
    return selected_log_probs.squeeze(-1)


def compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep):
    """
    Compute per-token log probabilities for a subset of tokens (typically the completion tokens).

    Args:
        model: The language model to use.
        input_ids (torch.Tensor): Tensor of shape (batch_size, total_seq_len) containing token ids
                                  for both prompt and completion.
        attention_mask (torch.Tensor): Tensor of shape (batch_size, total_seq_len) indicating which tokens are real (1) or padding (0).
        logits_to_keep (int): Number of tokens (from the completion part) for which we need log probabilities.

    Returns:
        torch.Tensor: Log probabilities for the last `logits_to_keep` tokens of each sequence.

    Explanation:
        1. We call the model with logits_to_keep + 1 so that the model outputs one extra logit than needed.
           This is common in next-token prediction setups.
        2. We slice off the last logit along the sequence dimension because it does not correspond to any input token.
        3. We then restrict both the input_ids and logits to the last logits_to_keep tokens, which should
           correspond to the generated completion portion.
        4. Finally, we use the selective_log_softmax to compute log probabilities only for those tokens.
    """
    # Run the model forward pass and obtain logits.
    # FIXME use many memory usage
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,
        obtain_logits=True,  # Request one extra logit for proper alignment.
    )  # Shape: (batch_size, total_seq_len, vocab_size)

    # Remove the last logit as it does not have a corresponding target token.
    logits = logits[:, :-1, :]  # New shape: (batch_size, total_seq_len - 1, vocab_size)

    # Slice the input_ids to keep only the last logits_to_keep tokens.
    # This corresponds to the generated completion tokens.
    input_ids = input_ids[:, -logits_to_keep:]  # Shape: (batch_size, logits_to_keep)

    # Also slice the logits to keep only those corresponding to the completion tokens.
    logits = logits[:, -logits_to_keep:, :]  # Shape: (batch_size, logits_to_keep, vocab_size)

    # FIXME use many memory usage
    # Compute and return the log probabilities for the selected tokens.
    return selective_log_softmax(logits, input_ids)


def generate_rollout_data(
    model, ref_model, tokenizer, batch_samples, num_generations, max_new_tokens, max_length_for_gather, temperature, do_sample
):
    """
    Generate rollouts and compute static log probabilities for both the old policy (current model)
    and the reference model. Gradients are disabled so that these remain fixed.

    Args:
        model: The current model (policy) used to generate rollouts.
        ref_model: The static reference model.
        tokenizer: The tokenizer.
        batch_samples: List of training samples.
        num_generations: Number of completions to generate per prompt.
        max_new_tokens: Maximum completion length.

    Returns:
        A dictionary with rollout data including both old and reference log probabilities.
    """
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    # Extract prompts and answers.
    # prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    # answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]
    prompts = batch_samples["prompt"]
    questions = batch_samples["question"]
    answers = batch_samples["answer"]

    # Generate completions and associated masks.
    # We generate once, and then use the same completions to compute both sets of log probabilities.
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_new_tokens, max_length_for_gather, temperature, do_sample
        )
        # FIXME gjr question 这里是想 不观察到 observation 只观察到 最终输出结果来计算概率这些吗？
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # FIXME many GPU memory
        # Compute old_log_probs from the current model, with gradients disabled.
        old_log_probs = compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep)

        # FIXME RuntimeError: 'weight' must be 2-D
        # Compute ref_log_probs from the reference model, which remains static.
        ref_log_probs = compute_log_probabilities(ref_model, input_ids, attention_mask, logits_to_keep)

    formatted_completions = [[{"content": tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids]

    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,  # Static log probs from the current model (old policy)
        "ref_log_probs": ref_log_probs,  # Static log probs from the reference model
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations,
    }


def res_compute_group_relative_advantages(rewards, num_generations):
    """
    Compute group-relative advantages for each prompt group.

    Args:
        rewards (torch.Tensor): Tensor of shape (batch_size * num_generations) containing rewards.
        num_generations (int): Number of completions generated per prompt.

    Returns:
        torch.Tensor: Tensor of advantages computed relative to the group mean.
    """
    # Reshape rewards to group by prompt
    rewards_by_group = rewards.view(-1, num_generations)

    # Compute mean and standard deviation for each prompt group
    group_means = rewards_by_group.mean(dim=1)
    group_stds = rewards_by_group.std(dim=1)

    # Expand the means and stds to match the original flat rewards tensor shape
    expanded_means = group_means.repeat_interleave(num_generations)
    expanded_stds = group_stds.repeat_interleave(num_generations)

    # Normalize rewards to get advantages
    # FIXME  TODO
    # rewards_
    advantages = (rewards - expanded_means) / (expanded_stds + 1e-4)

    return advantages.unsqueeze(1)  # Add dimension for token-wise operations


def compute_group_relative_advantages(rewards, num_generations):
    """
    Compute group-relative advantages for each prompt group.

    Args:
        rewards (torch.Tensor): Tensor of shape (batch_size * num_generations) containing rewards.
        num_generations (int): Number of completions generated per prompt.

    Returns:
        torch.Tensor: Tensor of advantages computed relative to the group mean.
    """
    # Reshape rewards to group by prompt
    rewards_by_group = rewards.view(-1, num_generations)

    # Compute mean, std, min, max for each prompt group
    group_means = rewards_by_group.mean(dim=1)
    group_stds = rewards_by_group.std(dim=1)
    group_mins = rewards_by_group.min(dim=1).values
    group_maxs = rewards_by_group.max(dim=1).values

    # Identify groups where mean == min or mean == max
    # (If all samples in a group are the same, its mean == min == max.)
    mean_equals_min_or_max = (group_means == group_mins) | (group_means == group_maxs)

    # Expand mean, std, and the boolean mask to match the original flat shape
    expanded_means = group_means.repeat_interleave(num_generations)
    expanded_stds = group_stds.repeat_interleave(num_generations)
    expanded_mask = mean_equals_min_or_max.repeat_interleave(num_generations)

    # Normal advantage computation
    advantages = (rewards - expanded_means) / (expanded_stds + 1e-4)

    # For groups where mean == min or max, replace advantage with ±1 (random)
    random_signs = (torch.randint(0, 2, rewards.shape, device=rewards.device) * 2 - 1).float()
    advantages[expanded_mask] = random_signs[expanded_mask]

    return advantages.unsqueeze(1)  # Add dimension for token-wise operations


def maximize_grpo_objective(model, ref_model, rollout_data, tokenizer, reward_function, optimizer, beta, epsilon, accelerator):
    """
    Update the policy model by maximizing the GRPO objective.

    Args:
        model: The current policy model.
        ref_model: The reference model.
        rollout_data: Dictionary containing rollout data.
        tokenizer: The tokenizer.
        reward_function: Function to compute rewards.
        optimizer: The optimizer.
        beta (float): KL penalty coefficient.
        epsilon (float): Clipping parameter.

    Returns:
        float: The loss value.
    """
    # Extract data from rollout
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    logits_to_keep = rollout_data["logits_to_keep"]

    # Compute current log probabilities
    current_log_probs = compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep)

    # Compute policy ratio
    ratio = torch.exp(current_log_probs - old_log_probs)

    # Get rewards data
    formatted_completions = rollout_data["formatted_completions"]
    repeated_prompts = rollout_data["repeated_prompts"]
    repeated_answers = rollout_data["repeated_answers"]

    # Compute rewards
    reward_dict = reward_function(
        prompts=repeated_prompts,
        completions=formatted_completions,
        answer=repeated_answers,
    )
    rewards = torch.tensor(
        reward_dict["total_scores"],
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    avg_reward = rewards.mean().item()
    print(f"Average Reward: {avg_reward:.4f}")

    # Compute advantages using group-relative normalization
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    advantages = compute_group_relative_advantages(rewards, num_generations)

    # Compute surrogate loss with clipping
    surrogate1 = ratio * advantages
    # surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.35) * advantages
    surrogate_loss = torch.min(surrogate1, surrogate2)

    # Compute KL divergence penalty
    kl_div = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1

    # Combine losses
    per_token_loss = surrogate_loss - beta * kl_div
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    print(loss.item())
    # Optimization step
    print(accelerator.device, "in optimizer.zero_grad()")
    optimizer.zero_grad()
    print(accelerator.device, "in accelerator.backward(loss)")
    accelerator.backward(loss)
    print(accelerator.device, "in optimizer.step()")
    optimizer.step()
    print(accelerator.device, "in return loss.item(), avg_reward, reward_dict")
    return loss.item(), avg_reward, reward_dict


def analyze_completions(completions, reward_dict):
    import re

    processed_completions = []
    for comp in completions:
        if isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], dict) and "content" in comp[0]:
            processed_completions.append(comp[0]["content"])
        elif isinstance(comp, dict) and "content" in comp:
            processed_completions.append(comp["content"])
        elif isinstance(comp, str):
            processed_completions.append(comp)
        else:
            continue

    if not processed_completions:
        return {
            "avg_completion_length": 0,
            "avg_completion_length_noobservation": 0,
            "answer_format_accuracy": 0,
            "avg_search_pairs": 0,
            "avg_search_content_length": 0,
            "avg_reasoning_pairs": 0,
            "avg_reasoning_content_length": 0,
            "avg_backtrack_pairs": 0,
            "avg_backtrack_content_length": 0,
        }

    # 1. 完成长度
    completion_lengths = [len(comp) for comp in processed_completions]
    print(f"completion_lengths: {completion_lengths}")
    avg_completion_length = sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0

    # 1.1 计算去除observation标签内容后的长度
    completion_lengths_noobservation = []
    for comp in processed_completions:
        # 移除所有<observation>...</observation>内容
        cleaned_comp = re.sub(r"<observation>.*?</observation>", "", comp, flags=re.DOTALL)
        completion_lengths_noobservation.append(len(cleaned_comp))

    avg_completion_length_noobservation = (
        sum(completion_lengths_noobservation) / len(completion_lengths_noobservation) if completion_lengths_noobservation else 0
    )

    # 2. 答案格式准确率
    answer_format_count = sum(1 for comp in processed_completions if "<answer>" in comp and "</answer>" in comp)
    avg_answer_format_accuracy = answer_format_count / len(processed_completions) if processed_completions else 0

    # 3. 搜索相关指标
    search_pairs_count = 0
    search_content_lengths = []

    for comp in processed_completions:
        search_starts = [m.start() for m in re.finditer("<search>", comp)]
        search_ends = [m.start() for m in re.finditer("</search>", comp)]

        valid_pairs = 0
        for start_pos in search_starts:
            valid_end = next((end for end in search_ends if end > start_pos), None)
            if valid_end is not None:
                valid_pairs += 1
                content_length = len(comp[start_pos + len("<search>") : valid_end].strip().split())
                search_content_lengths.append(content_length)
                search_ends.remove(valid_end)

        search_pairs_count += valid_pairs

    # 4. 推理标志使用情况
    reasoning_pairs_count = 0
    reasoning_content_lengths = []

    for comp in processed_completions:
        if "<reasoning>" in comp and "</reasoning>" in comp:
            reasoning_pairs_count += 1
            reasoning_start = comp.find("<reasoning>") + len("<reasoning>")
            reasoning_end = comp.find("</reasoning>")
            if reasoning_start < reasoning_end:
                content_length = len(comp[reasoning_start:reasoning_end].strip().split())
                reasoning_content_lengths.append(content_length)

    # 5. 反思标签使用情况
    backtrack_pairs_count = 0
    backtrack_content_lengths = []

    for comp in processed_completions:
        if "<backtrack>" in comp and "</backtrack>" in comp:
            backtrack_pairs_count += 1
            backtrack_start = comp.find("<backtrack>") + len("<backtrack>")
            backtrack_end = comp.find("</backtrack>")
            if backtrack_start < backtrack_end:
                content_length = len(comp[backtrack_start:backtrack_end].strip().split())
                backtrack_content_lengths.append(content_length)

    total_completions = len(processed_completions)

    avg_search_pairs = search_pairs_count / total_completions if total_completions else 0
    avg_search_content_length = sum(search_content_lengths) / len(search_content_lengths) if search_content_lengths else 0

    avg_reasoning_pairs = reasoning_pairs_count / total_completions if total_completions else 0
    avg_reasoning_content_length = (
        sum(reasoning_content_lengths) / len(reasoning_content_lengths) if reasoning_content_lengths else 0
    )

    avg_backtrack_pairs = backtrack_pairs_count / total_completions if total_completions else 0
    avg_backtrack_content_length = (
        sum(backtrack_content_lengths) / len(backtrack_content_lengths) if backtrack_content_lengths else 0
    )
    # 汇总奖励和性能指标
    metrics = {
        "avg_completion_length": avg_completion_length,
        "avg_completion_length_noobservation": avg_completion_length_noobservation,
        "avg_answer_format_accuracy": avg_answer_format_accuracy,
        "avg_search_pairs": avg_search_pairs,
        "avg_search_content_length": avg_search_content_length,
        "avg_reasoning_pairs": avg_reasoning_pairs,
        "avg_reasoning_content_length": avg_reasoning_content_length,
        "avg_backtrack_pairs": avg_backtrack_pairs,
        "avg_backtrack_content_length": avg_backtrack_content_length,
    }

    # 添加奖励指标
    if "correctness_scores" in reward_dict:
        metrics["correctness_reward"] = torch.tensor(reward_dict["correctness_scores"]).mean().item()
    if "format_scores" in reward_dict:
        metrics["format_reward"] = torch.tensor(reward_dict["format_scores"]).mean().item()
    if "rag_scores" in reward_dict:
        metrics["rag_reward"] = torch.tensor(reward_dict["rag_scores"]).mean().item()

    return metrics


def train_with_grpo(
    policy_model,
    base_reference_model,
    tokenizer,
    accelerator=None,
    dataloader=None,
    num_iterations=1,
    steps_per_iteration=500,
    num_generations=4,
    max_new_tokens=128,
    max_length_for_gather=2000,
    temperature=0.7,
    do_sample=True,
    beta=0.1,
    learning_rate=5e-6,
    mu=1,
    epsilon=0.2,
    reward_function=combined_reward,
    model_saver=None,
    checkpoint_dir=None,
    current_step=0,
    save_interval=10,
):
    """
    Iterative Group Relative Policy Optimization algorithm.

    Args:
        model: The initial policy model to be fine-tuned.
        tokenizer: The tokenizer used for encoding prompts and decoding completions.
        train_data (list): List of training samples with "prompt" and "answer" fields.
        num_iterations (int): Number of outer iterations (reward model updates).
        steps_per_iteration (int): Number of policy update steps per iteration.
        batch_size (int): Number of prompt samples per batch.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum token length for completions.
        beta (float): KL-divergence penalty coefficient.
        learning_rate (float): Learning rate for optimizer.
        mu (int): Number of GRPO updates per batch of generations.
        epsilon (float): Clipping parameter for surrogate objective.
        reward_function: Function that evaluates completions and returns rewards.

    Returns:
        The fine-tuned policy model.
    """

    # pdb.set_trace()
    # Outer loop for iterations with reward model updates
    sum_steps = current_step
    for iteration in tqdm(range(1, num_iterations + 1), desc="GRPO Iterations", position=0, leave=True):
        reference_model = copy.deepcopy(base_reference_model)
        torch.cuda.empty_cache()
        # tqdm.write(f"Starting iteration {iteration}/{num_iterations}")

        # 检查policy_model是否是DeepSpeedEngine
        is_deepspeed = hasattr(policy_model, "module") and isinstance(policy_model, DeepSpeedEngine)

        # 如果是DeepSpeedEngine，先解开模型
        if is_deepspeed:
            policy_model = accelerator.unwrap_model(policy_model)

        accelerator.wait_for_everyone()
        # Initialize optimizer
        optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)

        # policy_model.to(accelerator.device)
        policy_model.train()
        policy_model, optimizer, dataloader = accelerator.prepare(policy_model, optimizer, dataloader)
        

        # Create reference model for KL constraint
        # Zero3
        # reference_model = copy_policy_to_reference(accelerator, policy_model, reference_model)
        # reference_model = copy.deepcopy(policy_model)

        # aggregated_model = accelerator.unwrap_model(policy_model)
        # 再进行深拷贝得到参考模型
        # reference_model = copy.deepcopy(aggregated_model)
        # for param in reference_model.parameters():
        #     param.requires_grad = False
        # reference_model.to(accelerator.device)

        # pdb.set_trace()
        # 收集所有 LoRA 参数
        # pdb.set_trace()
        policy_model_lora_params = [p for name, p in policy_model.named_parameters() if "lora" in name]
        with deepspeed.zero.GatheredParameters(policy_model_lora_params, enabled=True):
            # 从所有 GPU 收集 LoRA 参数
            policy_model_state_dict = policy_model.state_dict()

            # 创建一个新的字典，只包含 LoRA 参数
            lora_state_dict = {k: v for k, v in policy_model_state_dict.items() if "lora" in k}

            # 将 policy 模型的 LoRA 参数复制到 reference 模型
            missing_keys, unexpected_keys = reference_model.load_state_dict(lora_state_dict, strict=False)
            reference_model.to(accelerator.device)

            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Copied LoRA parameters from policy to reference model")
            print(f"Missing keys: {missing_keys if len(missing_keys) < 10 else f'{len(missing_keys)} keys'}")
            print(f"Unexpected keys: {unexpected_keys if len(unexpected_keys) < 10 else f'{len(unexpected_keys)} keys'}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        reference_model = accelerator.prepare(reference_model)
        # Inner loop for policy updates
        step = 0
        for batch in dataloader:
            torch.cuda.empty_cache()
            start_time = time.time()
            # Set old policy for this step
            with torch.no_grad():
                # Generate completions and compute log probs
                rollout_data = generate_rollout_data(
                    policy_model,
                    reference_model,
                    tokenizer,
                    batch,
                    num_generations,
                    max_new_tokens,
                    max_length_for_gather,
                    temperature,
                    do_sample,
                )
            # pdb.set_trace()

            # Multiple GRPO updates per batch of generations
            for grpo_iter in range(1, mu + 1):
                loss_value, avg_reward, reward_dict = maximize_grpo_objective(
                    policy_model, reference_model, rollout_data, tokenizer, reward_function, optimizer, beta, epsilon, accelerator
                )

            end_time = time.time()
            elapsed_time = end_time - start_time

            all_completions = []
            for comp_list in rollout_data["formatted_completions"]:
                for comp_dict in comp_list:
                    all_completions.append(comp_dict["content"])
            # pdb.set_trace()

            completion_stats = analyze_completions(all_completions, reward_dict)

            completion_length = completion_stats["avg_completion_length"]
            answer_format_accuracy = completion_stats["avg_answer_format_accuracy"]
            correctness_reward = completion_stats["correctness_reward"]
            format_reward = completion_stats["format_reward"]
            rag_reward = completion_stats["rag_reward"]
            completion_length_noobservation = completion_stats["avg_completion_length_noobservation"]
            # pdb.set_trace()

            accelerator.wait_for_everyone()
            all_sum_steps = accelerator.gather_for_metrics([sum_steps])
            all_loss_value = accelerator.gather_for_metrics([loss_value])
            all_avg_reward = accelerator.gather_for_metrics([avg_reward])
            all_correctness_reward = accelerator.gather_for_metrics([correctness_reward])
            all_format_reward = accelerator.gather_for_metrics([format_reward])
            all_rag_reward = accelerator.gather_for_metrics([rag_reward])
            all_time_per_step = accelerator.gather_for_metrics([elapsed_time])
            all_completion_length = accelerator.gather_for_metrics([completion_length])
            all_answer_format_accuracy = accelerator.gather_for_metrics([answer_format_accuracy])
            all_completion_length_noobservation = accelerator.gather_for_metrics([completion_length_noobservation])
            if accelerator.is_local_main_process:
                swanlab.log(
                    {
                        # 训练进度指标
                        "sum_steps": sum(all_sum_steps) / len(all_sum_steps),
                        "loss": sum(all_loss_value) / len(all_loss_value),
                        # 奖励指标
                        "total_reward": sum(all_avg_reward) / len(all_avg_reward),
                        "correctness_reward": sum(all_correctness_reward) / len(all_correctness_reward),
                        "format_reward": sum(all_format_reward) / len(all_format_reward),
                        "rag_reward": sum(all_rag_reward) / len(all_rag_reward),
                        # 性能指标
                        "time_per_step": sum(all_time_per_step) / len(all_time_per_step),
                        # 生成内容指标
                        "completion_length": sum(all_completion_length) / len(all_completion_length),
                        "completion_length_noobservation": sum(all_completion_length_noobservation)
                        / len(all_completion_length_noobservation),
                        "answer_format_accuracy": sum(all_answer_format_accuracy) / len(all_answer_format_accuracy),
                    }
                )

            print(
                f"Iteration {iteration}/{num_iterations}, Step {step+1}/{min(steps_per_iteration, len(dataloader))}, "
                f"Loss: {loss_value:.4f}, Avg Reward: {avg_reward:.4f}, Time: {elapsed_time:.2f}s"
            )

            step += 1
            sum_steps += 1

            print(f"sum_steps: {sum_steps}")
            if sum_steps % save_interval == 0 and sum_steps > current_step:
                accelerator.wait_for_everyone()

                if accelerator.is_local_main_process:
                    checkpoint_path = f"{checkpoint_dir}/step-{sum_steps:04d}"
                    os.makedirs(checkpoint_path, exist_ok=True)
                    # 保存 LoRA 部分的参数
                    policy_model.model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)

                accelerator.wait_for_everyone()
            if step >= steps_per_iteration:
                break

        tqdm.write(f"Completed iteration {iteration}. Reward model update would happen here.")
