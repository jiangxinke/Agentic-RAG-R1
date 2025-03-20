from custom_reward_function import *
from answer_extractor import *

import numpy as np
import torch
import torch.nn.functional as F
import copy
import random
from protoco import DataProto

def train_with_grpo_mu_GPU(model, tokenizer, train_data, num_iterations=1, 
                           steps_per_iteration=500, batch_size=4, num_generations=4, 
                           max_completion_length=128, beta=0.1, learning_rate=5e-6, 
                           mu=3, epsilon=0.2, reward_function=combined_reward, device_ids=None):
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
    assert device_ids is not None and len(device_ids) > 1, "This code needs at least 2 GPU cores to run!"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import torch.nn as nn
    # Wrap model with DataParallel if multiple GPUs are available.

    model = nn.DataParallel(model, device_ids=device_ids)
    print(f"Model wrapped with DataParallel across GPUs: {device_ids}")

    # Outer loop: iterative GRPO updates.
    for iteration in range(1, 1+num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")

        # Create a reference model (deep copy) and set it to eval mode.
        ref_model = copy.deepcopy(model.module)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print("Reference model created.")

        # Reinitialize the optimizer for this iteration.
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()

        # Inner loop: your original training steps.
        for step in range(1, steps_per_iteration + 1):
            batch_samples = random.sample(train_data, batch_size)
            with torch.no_grad():
                rollout_data = generate_rollout_data(
                    model,       # model is policy_model
                    ref_model,
                    tokenizer,
                    batch_samples,
                    num_generations,
                    max_completion_length
                )
            for grpo_iter in range(1, mu + 1):
                loss = maximize_grpo_objective(
                    model,
                    ref_model,
                    rollout_data,
                    tokenizer,
                    reward_function,
                    optimizer,
                    beta=beta,
                    epsilon=epsilon
                )
                # Optimization step
                optimizer.zero_grad()

                print("@@@"*30, "\nRUN Here")

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"!!! WARNING: Loss is NaN or Inf at step {step}, skipping backward")
                    return loss.item()  # 直接跳过这个 batch，避免反向传播 NaN

                print(f"Loss at step {step}: {loss.item()}")  # 确保 loss 在合理范围

                torch.cuda.synchronize()    # 这样可以确保所有 GPU 在进入 backward() 之前已经完成了 forward()。
                loss = loss.mean()
                loss.backward()     # FIXME 这一行一致没解决
                print("DDD"*30, "\nRUN Here")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                print(f"Iteration {iteration+1}/{num_iterations+1}, Step {step+1}/{steps_per_iteration + 1}, "
                      f"GRPO iter {grpo_iter+1}/{mu}, loss: {loss:.4f}")
                for i in range(torch.cuda.device_count()):
                   print(f"GPU {i} Usage: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MiB, "
                         f"Utilization: {torch.cuda.utilization(i)}%")
                # Uncomment to see the GPU utilization stats
    return model.module

def evaluate_model(model, tokenizer, eval_examples, device):
    """
    Evaluates the model on a set of examples and prints detailed results.
    
    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for encoding inputs and decoding outputs.
        eval_examples (list): List of evaluation examples, each containing "prompt" and "answer".
        device: The device (CPU or GPU) to run evaluation on.
        
    Returns:
        float: The accuracy percentage (correct predictions / total examples * 100).
        
    Explanation:
        1. Sets the model to evaluation mode.
        2. For each example in the evaluation set:
           - Encodes the prompt and generates a response using the model.
           - Extracts the predicted answer from the generated response.
           - Compares the predicted answer with the expected answer using multiple methods:
             a. Exact string matching
             b. Single number extraction and comparison
             c. Last number extraction and comparison
           - Prints detailed information about each example.
        3. Calculates and returns the overall accuracy.
        4. Returns the model to training mode.
    """
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "="*50)
    print("EVALUATION ON", total, "EXAMPLES")
    print("="*50)
    
    for example in eval_examples:
        # Build the full prompt using the same method as training.
        full_prompt = example["prompt"]
        expected = example["answer"]
        
        # Tokenize the full prompt and generate a response from the model.
        # inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, padding_side="left").to(device)
        # FIXME （这里的generate需要做处理）
        prompt_ids = inputs["input_ids"].to(device)
        prompt_ids = prompt_ids.repeat_interleave(1, dim=0)

        # todo，需要专门针对infer写
        outputs = model(
            prompt_ids,
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the predicted answer from the model output.
        try:
            predicted = extract_answer_from_model_output(response)
            
            # Check correctness in multiple ways
            if predicted == expected:  # First try exact match
                is_correct = True
            else:
                # Try single number
                pred_num = extract_single_number(str(predicted))
                exp_num = extract_single_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    # Try last number
                    pred_num = extract_last_number(str(predicted))
                    exp_num = extract_last_number(str(expected))
                    is_correct = (pred_num is not None and exp_num is not None and
                                pred_num == exp_num)

            if is_correct:
                correct += 1
                
            # Print details of the evaluation.
            print("\nPrompt:")
            print(full_prompt)
            print("\nExpected Answer:")
            print(expected)
            print("\nExtracted Answer:")
            print(predicted)
            print("\nFull Generated Response:")
            print(response)
            print("\nCorrect:", "✓" if is_correct else "✗")
            print("-"*50)
            
        except Exception as e:
            print("\nFailed to parse model output for prompt:")
            print(full_prompt)
            print("Error:", e)
            print("-"*50)
            
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*50)
    
    model.train()
    return accuracy


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
    # print("$$$"*30, "\nRUN Here")
    logits = model(            # FIXME
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,  # Request one extra logit for proper alignment.
        obtain_logits=True
    )  # Shape: (batch_size, total_seq_len, vocab_size)
    # print("$$$"*30)

    # Remove the last logit as it does not have a corresponding target token.
    logits = logits[:, :-1, :]  # New shape: (batch_size, total_seq_len - 1, vocab_size)

    # Slice the input_ids to keep only the last logits_to_keep tokens.
    # This corresponds to the generated completion tokens.
    input_ids = input_ids[:, -logits_to_keep:]  # Shape: (batch_size, logits_to_keep)

    # Also slice the logits to keep only those corresponding to the completion tokens.
    logits = logits[:, -logits_to_keep:, :]  # Shape: (batch_size, logits_to_keep, vocab_size)

    # Compute and return the log probabilities for the selected tokens.
    return selective_log_softmax(logits, input_ids)

# def create_completion_mask(completion_ids, eos_token_id): 
#     """
#     Create a binary mask for the generated completion tokens so that tokens after the first EOS are ignored.

#     Args:
#         completion_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) with generated token ids.
#         eos_token_id (int): The token id representing the end-of-sequence.

#     Returns:
#         torch.Tensor: A mask tensor of shape (batch_size, seq_len) with 1s for tokens up to and including the first EOS
#                       and 0s for tokens following the first EOS.

#     Explanation:
#         1. First, a boolean mask (is_eos) is created indicating where in the sequence the EOS token appears.
#         2. An index tensor (eos_idx) is initialized, assuming that no EOS is found (defaulting to the sequence length).
#         3. For sequences where EOS exists, eos_idx is updated to the position (index) of the first EOS.
#         4. A sequence index tensor is created that contains indices for each position in the sequence.
#         5. The final mask is computed by comparing the sequence indices to eos_idx (after adding a dimension).
#     """
#     # Determine which positions in each sequence equal the EOS token.
#     is_eos = completion_ids == eos_token_id  # Boolean tensor of shape (batch_size, seq_len)

#     # Initialize a tensor to store the index of the first EOS for each sequence.
#     # If no EOS is found, default to the full sequence length (is_eos.size(1)).
#     eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)

#     # Identify sequences that contain at least one EOS.
#     mask_exists = is_eos.any(dim=1)
#     # For sequences with an EOS, update eos_idx to the index of the first occurrence.
#     eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]

#     # Create a tensor of indices [0, 1, 2, ..., seq_len-1] and replicate it for each sequence in the batch.
#     sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)

#     # Build the mask: positions with an index less than or equal to the first EOS index are marked as 1.
#     completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

#     return completion_mask


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
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)

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
                observation_range_mask[batch_idx, seq_idx] = 0      # CHECK mask observation
            if is_observation_end[batch_idx, seq_idx]:
                in_observation_range = False

    # Combine the EOS mask and the <observation>... </observation> mask.
    final_mask = (completion_mask | observation_range_mask)

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
    tokenizer.padding_side  = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)      # Shape: (batch_size, prompt_seq_len)
    prompt_mask = inputs["attention_mask"].to(device)  # Shape: (batch_size, prompt_seq_len)
    prompt_length = prompt_ids.size(1)  # Save the prompt length to later separate prompt from completion.

    # Repeat each prompt num_generations times.
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)   # New shape: (batch_size*num_generations, prompt_seq_len)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0) # New shape: (batch_size*num_generations, prompt_seq_len)

    # Generate new tokens for each prompt. The output includes the original prompt and the generated tokens.
    outputs = model(       # old 方法是直接generate
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Remove the prompt portion from the generated output to isolate the completion tokens.
    completion_ids = outputs[:, prompt_length:]  # Shape: (batch_size*num_generations, completion_seq_len)

    # Create a binary mask that ignores tokens beyond the first EOS token.
    observation_start_token_id = tokenizer("<observation>").input_ids[0]
    observation_end_token_id = tokenizer("</observation>").input_ids[0]
    # completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id, observation_start_token_id, observation_end_token_id)

    return prompt_ids, prompt_mask, completion_ids, completion_mask

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
    tokenizer.padding_side  = "left"
    device = next(model.parameters()).device

    # Extract prompts and answers.
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]

    # Generate completions and associated masks.
    # We generate once, and then use the same completions to compute both sets of log probabilities.
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Compute old_log_probs from the current model, with gradients disabled.
        # 在 PPO/强化学习场景下，两次调用同一个模型是常见做法：第一次做采样(rollout)，第二次计算 log_probs(评估)。
        old_log_probs = compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep)     
        
        # Compute ref_log_probs from the reference model, which remains static.
        ref_log_probs = compute_log_probabilities(ref_model, input_ids, attention_mask, logits_to_keep)

    formatted_completions = [
        [{'content': tokenizer.decode(ids, skip_special_tokens=True)}]
        for ids in completion_ids
    ]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,   # Static log probs from the current model (old policy)
        "ref_log_probs": ref_log_probs,     # Static log probs from the reference model
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations
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


def maximize_grpo_objective(model, ref_model, rollout_data, tokenizer, reward_function, 
                          optimizer, beta, epsilon):
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
        reward_function(prompts=repeated_prompts, completions=formatted_completions, answer=repeated_answers),
        dtype=torch.float32,
        device=next(model.parameters()).device
    )
    avg_reward = rewards.mean().item()
    print(f"Average Reward: {avg_reward:.4f}")

    # print("+++"*30, "\nRUN Here")
    
    # Compute advantages using group-relative normalization
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    advantages = compute_group_relative_advantages(rewards, num_generations)

    # print("---"*30, "\nRUN Here")
    
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

    return loss