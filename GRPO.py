import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn.functional as F
import copy
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


from prepare_dataset import *
from utils import *
from rl_prompt import *
from answer_extractor import *
from custom_reward_function import *
from grpo_trainer import *

from generation_interrupt_new import CustomModel
# from generation_interrupt_new import LLMGenerationManager, GenerationConfig


def main():
    """
    Main function to run the complete training and evaluation pipeline.

    The process consists of:
      1. Loading the pre-trained model and tokenizer.
      2. Evaluating the initial model performance (before any finetuning).
      3. Performing reinforcement learning (GRPO) finetuning.
      4. Evaluating the final model after GRPO finetuning.
      5. Saving the finetuned model and tokenizer.

    Note: Functions such as prepare_dataset, evaluate_model, and reward_function 
          are assumed to be defined elsewhere.
    """
    # Determine the device (GPU if available, otherwise CPU) from the model's parameters.
    set_random_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the model name and output directory.
    model_name = "Qwen/Qwen2.5-7B-Instruct" # The 0.5B model is not smart enough
                                              # to generate the <reasoning> and <answer> tags
                                              # so several iterations of SFT to teach it these tags
                                              # are recommended before RL
    output_dir = "math_solver_model"

    # Load the pre-trained causal language model.
    # - torch_dtype specifies the precision (bfloat16 for efficiency on supported hardware).
    # - attn_implementation selects an optimized attention mechanism.
    # - device_map="auto" automatically distributes the model across available devices.
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2",
        device_map=None
    )
    dataset_name = "medmcqa"
    print("Downloaded model")
    # Move the model to the determined device.
    model = model.to(device)

    lora_config = LoraConfig(
        r=8,  # LoRA 的秩
        lora_alpha=32,  # LoRA 的缩放因子
        target_modules=["q_proj", "v_proj"],  # 应用 LoRA 的目标模块
        lora_dropout=0.1,  # LoRA 的 dropout 率
        bias="none",  # 是否在 LoRA 中引入偏置
        task_type="CAUSAL_LM"  # 任务类型
    )

    model = get_peft_model(model, lora_config)

    # Load the tokenizer corresponding to the model.
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    # Set the pad token to be the same as the end-of-sequence token.
    tokenizer.pad_token = tokenizer.eos_token
    # Update the model configuration with the correct token IDs.
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # -------------------------------
    # Step 0: INITIAL EVALUATION
    # -------------------------------
    # Load the complete training dataset using a helper function (assumed defined elsewhere).
    all_data = prepare_dataset("train", dataset_name)
    # Randomize the order of examples.
    random.shuffle(all_data)
    # Use a small subset (e.g., 30 examples) for evaluation.
    num_eval_examples = 1
    eval_data = all_data[:num_eval_examples]

    # Evaluate the initial performance of the model before any finetuning.
    print("\nInitial model evaluation before GRPO:")
    # pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    # print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

    model = optimize_model_memory(model)
    
    # -------------------------------
    # Step 1: RL FINETUNING (GRPO)
    # -------------------------------
    print("\nStarting RL finetuning using GRPO...")

    # Use the remaining examples (beyond the evaluation subset) for RL finetuning.
    train_data = all_data[num_eval_examples:]

    # Define RL training configuration.
    training_config = {
        'num_iterations' : 1,                # epoch 
        'steps_per_iteration': 150,          # Total number of RL training steps.
        'batch_size': 1,                     # Number of samples per training step.
        'num_generations': 4,                # Number of completions generated per prompt.
        'max_completion_length': 5000,        # Maximum token length for each generated completion.
        'beta': 0.04,                         # KL divergence penalty coefficient.
        'learning_rate': 5e-6,                # Learning rate for RL fine-tuning.
        'mu': 1,
        'epsilon': 0.1,
        'reward_function': combined_reward
    }
    # Fine-tune the model using GRPO RL training.
    custom_model = CustomModel(model, tokenizer)    

    model = train_with_grpo(
        model=custom_model,
        tokenizer=tokenizer,
        train_data=train_data,
        **training_config
    )

    # -------------------------------
    # Step 2: FINAL EVALUATION & SAVING
    # -------------------------------
    print("\nFinal model evaluation after GRPO RL finetuning:")
    # Evaluate the final model performance using the evaluation dataset.
    post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)        # TODO
    print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")
    # print(f"Total Improvement: {post_grpo_accuracy - pre_grpo_accuracy:.2f}%")

    print("\nSaving GRPO finetuned model...")
    # Save the final finetuned model and tokenizer to disk.
    model.save_pretrained("grpo_finetuned_model")
    tokenizer.save_pretrained("grpo_finetuned_model")

if __name__ == "__main__":
    main()
