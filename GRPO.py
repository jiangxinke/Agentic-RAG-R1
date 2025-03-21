import os
import json
import logging
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import datetime
import torch
import random


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from prepare_dataset import prepare_dataset
from utils import load_config, set_random_seed, optimize_model_memory
from custom_reward_function import combined_reward
from generation_interrupt_new import CustomModel
from grpo_trainer import train_with_grpo, evaluate_model
from grpo_trainer_mu_GPU import train_with_grpo_mu_GPU


def main():
    """
    Main function to run the complete training and evaluation pipeline.

    The process includes:
      1. Loading configurations and setting environment variables.
      2. Loading the pre-trained model and tokenizer.
      3. Evaluating the initial model performance.
      4. Performing reinforcement learning fine-tuning (GRPO).
      5. Evaluating the final model performance.
      6. Saving the fine-tuned model and tokenizer.
    """

    # Set up logging
    now = datetime.datetime.now()
    abbr = str(now)
    log_dir = Path(f"logs/{abbr}")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # 1. Loading configurations and setting environment variables.
    config = load_config("config/config.yaml")

    # Save configuration
    with open(log_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    set_random_seed(config.random_seed)
    logger.info(f"Set random seed to {config.random_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Loading the pre-trained model and tokenizer.
    logger.info("Downloading model...")
    torch_dtype = getattr(torch, config.model.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name, torch_dtype=torch_dtype, device_map=config.model.device_map
    )
    logger.info("Downloaded model")
    model = model.to(device)

    # LoRA configuration
    lora_cfg = LoraConfig(
        r=config.lora_config.r,
        lora_alpha=config.lora_config.lora_alpha,
        target_modules=config.lora_config.target_modules,
        lora_dropout=config.lora_config.lora_dropout,
        bias=config.lora_config.bias,
        task_type=config.lora_config.task_type,
    )
    model = get_peft_model(model, lora_cfg)
    logger.info(f"Applied LoRA with config: {lora_cfg}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # multi-GPU usage
    device_ids = None
    if config.training.mu_gpu:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPUs")
        device_ids = list(range(num_gpus)) if num_gpus > 1 else None

    # dataset
    all_data = prepare_dataset("train", config.dataset.name)
    random.shuffle(all_data)
    train_data = all_data[config.dataset.num_eval_examples :]
    eval_data = all_data[: config.dataset.num_eval_examples]
    logger.info(f"Dataset split - Train: {len(train_data)}, Eval: {len(eval_data)}")

    # 3. Evaluating the initial model performance.
    logger.info("\nInitial model evaluation before GRPO:")
    pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    logger.info(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

    # Save initial results
    results = {
        "pre_grpo_accuracy": pre_grpo_accuracy,
    }
    with open(log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    exit()

    # 4. Performing reinforcement learning fine-tuning (GRPO).
    logger.info("\nStarting RL finetuning using GRPO...")
    training_config = {
        "num_iterations": config.training.num_iterations,
        "steps_per_iteration": config.training.steps_per_iteration,
        "batch_size": config.training.batch_size,
        "num_generations": config.training.num_generations,
        "max_completion_length": config.training.max_completion_length,
        "beta": config.training.beta,
        "learning_rate": config.training.learning_rate,
        "mu": config.training.mu,
        "epsilon": config.training.epsilon,
        "reward_function": combined_reward,
    }

    model = optimize_model_memory(model)
    custom_model = CustomModel(model, tokenizer)

    if config.training.mu_gpu:
        model = train_with_grpo_mu_GPU(
            model=custom_model,
            tokenizer=tokenizer,
            train_data=train_data,
            device_ids=device_ids,
            **training_config,
        )
    else:
        model = train_with_grpo(
            model=custom_model,
            tokenizer=tokenizer,
            train_data=train_data,
            **training_config,
        )

    # 5. Evaluating the final model performance.
    logger.info("\nFinal model evaluation after GRPO RL finetuning:")
    post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    improvement = post_grpo_accuracy - pre_grpo_accuracy
    logger.info(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")
    logger.info(f"Total Improvement: {improvement:.2f}%")

    # Update and save final results
    results.update(
        {
            "post_grpo_accuracy": post_grpo_accuracy,
            "improvement": improvement,
        }
    )
    with open(log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 6. Saving the fine-tuned model and tokenizer.
    save_dir = f"{config.save_path}/{abbr}/grpo_finetuned_model"
    logger.info(f"\nSaving GRPO finetuned model to {save_dir} ...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
