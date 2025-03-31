import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import copy
import random
from tqdm import tqdm

from trl import SFTTrainer

from grpo.reward_function import combined_reward
from utils.answer_extractor import extract_answer_from_model_output
from utils.protoco import DataProto
from utils.utils import print_memory_usage
from accelerate import Accelerator


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


def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    """
    Generate multiple completions for each prompt and create corresponding attention masks.

    Args:
        model: The language model used for generation.
        tokenizer: The tokenizer to process the prompts and decode the outputs.
        prompts (list of str): List of input prompt strings.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum number of new tokens to generate for the completion.

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

    # Generate new tokens for each prompt. The output includes the original prompt and the generated tokens.
    outputs = model(  # old 方法是直接generate
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Remove the prompt portion from the generated output to isolate the completion tokens.
    completion_ids = outputs[:, prompt_length:]  # Shape: (batch_size*num_generations, completion_seq_len)

    # Create a binary mask that ignores tokens beyond the first EOS token.
    observation_start_token_id = tokenizer("<observation>").input_ids[0]
    observation_end_token_id = tokenizer("</observation>").input_ids[0]
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


def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    """
    Generate rollouts and compute static log probabilities for both the old policy (current model)
    and the reference model. Gradients are disabled so that these remain fixed.

    Args:
        model: The current model (policy) used to generate rollouts.
        ref_model: The static reference model.
        tokenizer: The tokenizer.
        batch_samples: List of training samples.
        num_generations: Number of completions to generate per prompt.
        max_completion_length: Maximum completion length.

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
            model, tokenizer, prompts, num_generations, max_completion_length
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

    # Compute mean and standard deviation for each prompt group
    group_means = rewards_by_group.mean(dim=1)
    group_stds = rewards_by_group.std(dim=1)

    # Expand the means and stds to match the original flat rewards tensor shape
    expanded_means = group_means.repeat_interleave(num_generations)
    expanded_stds = group_stds.repeat_interleave(num_generations)

    # Normalize rewards to get advantages
    advantages = (rewards - expanded_means) / (expanded_stds + 1e-4)

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
    rewards = torch.tensor(
        reward_function(
            prompts=repeated_prompts,
            completions=formatted_completions,
            answer=repeated_answers,
        ),
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
    surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surrogate1, surrogate2)

    # Compute KL divergence penalty
    kl_div = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1

    # Combine losses
    per_token_loss = surrogate_loss - beta * kl_div
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    print(loss.item())
    # Optimization step
    optimizer.zero_grad()
    # loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    accelerator.backward(loss)
    optimizer.step()

    return loss.item(), avg_reward


# def copy_policy_to_reference(accelerator, policy_model, reference_model):
#     # reference_model = copy.deepcopy(policy_model)
#     # 获取 policy_model 的完整参数（ZeRO-3 需通过 Accelerator 处理）
#     policy_state_dict = accelerator.get_state_dict(policy_model)
#     # 加载到 reference_model
#     accelerator.unwrap_model(reference_model).load_state_dict(policy_state_dict)
#     return reference_model


def initialize_model(model, tokenizer, device):
    # 构造一个 dummy 输入（根据模型输入要求调整形状和内容）
    dummy_input = tokenizer("Hello, world!", return_tensors="pt").input_ids.to(device)
    # 执行一次前向传播触发参数初始化
    with torch.no_grad():
        _ = model(dummy_input)
    return model


# def copy_policy_to_reference(accelerator, policy_model, reference_model):
#     # 获取 policy_model 的完整参数（例如 ZeRO-3 模式下需要 Accelerator 处理）
#     policy_state_dict = accelerator.get_state_dict(policy_model)

#     # 创建一个新的 state_dict，调整键名（这里示例为去除前缀 "model."）
#     adjusted_state_dict = {}
#     for key, value in policy_state_dict.items():
#         # 如果键以 "model." 开头，则去掉这个前缀
#         if key.startswith("model."):
#             new_key = key[len("model."):]
#         else:
#             new_key = key
#         adjusted_state_dict[new_key] = value

#     # 加载调整后的参数到 reference_model
#     accelerator.unwrap_model(reference_model).load_state_dict(adjusted_state_dict)
#     return reference_model


# def copy_policy_to_reference(accelerator, policy_model, reference_model):
#     # 假设0号GPU是主设备
#     # accelerator.broadcast(policy_model.state_dict(), src=0)
#     # 然后将这些参数加载到reference model
#     # reference_model.load_state_dict(policy_model.state_dict(), strict=False)

#     # 尝试使用 accelerator.get_state_dict；如果返回 None，则直接调用 state_dict()
#     policy_state_dict = accelerator.get_state_dict(policy_model)

#     # 创建一个新的 state_dict，调整键名（这里示例为去除前缀 "model."）
#     adjusted_state_dict = {}
#     for key, value in policy_state_dict.items():
#         # 如果键以 "model." 开头，则去掉这个前缀
#         if key.startswith("module."):
#             new_key = key[len("module.") :]
#         else:
#             new_key = key
#         adjusted_state_dict[new_key] = value

#     # 加载状态字典
#     reference_model.load_state_dict(adjusted_state_dict)

#     accelerator.wait_for_everyone()
#     return reference_model


# def copy_policy_to_reference(accelerator, policy_model, reference_model):
#     # 获取 policy_model 的参数，只在 rank 0 上执行
#     if accelerator.is_local_main_process:
#         policy_state_dict = policy_model.state_dict()  # 改为直接从模型获取状态字典
#     else:
#         policy_state_dict = {}

#     reference_model = accelerator.unwrap_model(reference_model)

#     # # 获取并打印 policy_model 的第一个参数
#     # policy_state_dict = policy_model.state_dict()
#     # first_policy_param = list(policy_state_dict.items())[0]  # 获取第一个参数
#     # print("Policy model first parameter:", first_policy_param)

#     # # 获取并打印 reference_model 的第一个参数
#     # reference_state_dict = reference_model.state_dict()
#     # first_reference_param = list(reference_state_dict.items())[0]  # 获取第一个参数
#     # print("Reference model first parameter:", first_reference_param)

#     # exit()


#     # # 创建一个新的 state_dict，调整键名
#     # adjusted_state_dict = {}
#     # for key, value in policy_state_dict.items():
#     #     if key.startswith("module."):
#     #         new_key = key[len("module."):]
#     #         # model.base_model.model.model.layers
#     #         # new_key = key
#     #     else:
#     #         new_key = key
#     #     adjusted_state_dict[new_key] = value

#     # # 确保 reference_model 的参数是从正确的状态字典加载
#     # reference_model = accelerator.unwrap_model(reference_model)
#     # reference_model.load_state_dict(adjusted_state_dict)
#     reference_model.eval()

#     # 在所有卡上同步 reference_model
#     reference_model = accelerator.prepare(reference_model)

#     return reference_model


# def copy_policy_to_reference(accelerator, policy_model, reference_model):
#     # 1. 从分布式环境中获取合并后的 policy_model 的参数
#     accelerator.wait_for_everyone()

#     # 确保在分布式训练时解包模型
#     if accelerator.is_local_main_process:
#         # FIXME 拿不到lora矩阵
#         policy_model = accelerator.unwrap_model(policy_model, keep_fp32_wrapper=True, keep_torch_compile=True)  # 解包分布式模型

#     # 获取状态字典
#     policy_state_dict = policy_model.model.state_dict()  # 获取 policy_model 的状态字典
#     reference_state_dict = reference_model.model.state_dict()  # 获取 reference_model 的状态字典

#     # FIXME 暂时跳过了那些没有对齐的情况(LoRA)
#     reference_model.model.load_state_dict(policy_state_dict, strict=False)

#     # 3. 将 reference_model 转换到分布式环境
#     # 不考虑reference_model的初始化，直接load到卡上 => 不会报错
#     reference_model = accelerator.prepare(reference_model)  # 转换为适应多GPU环境

#     # 设置 reference_model 为评估模式（如果需要的话）
#     reference_model.eval()

#     print("Policy 参数已同步到 Reference Model，并转为分布式模式")

#     return reference_model


def train_with_grpo(
    policy_model,
    reference_model,
    tokenizer,
    accelerator=None,
    dataloader=None,
    num_iterations=1,
    steps_per_iteration=500,
    num_generations=4,
    max_completion_length=128,
    beta=0.1,
    learning_rate=5e-6,
    mu=1,
    epsilon=0.2,
    reward_function=combined_reward,
    model_saver=None,
    checkpoint_dir=None,
    current_step=0,
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

    # Outer loop for iterations with reward model updates
    sum_steps = current_step
    for iteration in tqdm(range(1, num_iterations + 1), desc="GRPO Iterations", position=0, leave=True):
        torch.cuda.empty_cache()
        # tqdm.write(f"Starting iteration {iteration}/{num_iterations}")

        # Initialize optimizer
        optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)

        # policy_model.to(accelerator.device)
        policy_model.train()
        policy_model, optimizer, dataloader = accelerator.prepare(policy_model, optimizer, dataloader)

        # Create reference model for KL constraint
        # Zero3
        # reference_model = copy_policy_to_reference(accelerator, policy_model, reference_model)
        # reference_model = copy.deepcopy(policy_model)
        aggregated_model = accelerator.unwrap_model(policy_model)
        # 再进行深拷贝得到参考模型
        reference_model = copy.deepcopy(aggregated_model)
        for param in reference_model.parameters():
            param.requires_grad = False
        reference_model.to(accelerator.device)

        # Inner loop for policy updates
        step = 0
        for batch in dataloader:
            torch.cuda.empty_cache()
            start_time = time.time()
            # Set old policy for this step
            with torch.no_grad():
                # Generate completions and compute log probs
                rollout_data = generate_rollout_data(
                    policy_model, reference_model, tokenizer, batch, num_generations, max_completion_length
                )

            # Multiple GRPO updates per batch of generations
            for grpo_iter in range(1, mu + 1):
                loss_value, avg_reward = maximize_grpo_objective(
                    policy_model, reference_model, rollout_data, tokenizer, reward_function, optimizer, beta, epsilon, accelerator
                )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"Iteration {iteration}/{num_iterations}, Step {step+1}/{min(steps_per_iteration, len(dataloader))}, "
                f"Loss: {loss_value:.4f}, Avg Reward: {avg_reward:.4f}, Time: {elapsed_time:.2f}s"
            )

            step += 1
            sum_steps += 1

            print(f"sum_steps: {sum_steps}")
            if sum_steps % 2 == 0 and sum_steps > current_step:
                accelerator.wait_for_everyone()

                if accelerator.is_local_main_process:
                    checkpoint_path = f"{checkpoint_dir}/step-{sum_steps:04d}"
                    os.makedirs(checkpoint_path, exist_ok=True)
                    # 保存 LoRA 部分的参数
                    policy_model.model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
            if step >= steps_per_iteration:
                break

        tqdm.write(f"Completed iteration {iteration}. Reward model update would happen here.")

    return policy_model
