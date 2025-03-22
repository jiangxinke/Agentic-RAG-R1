import os
import json
import logging
import pdb
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import datetime
import torch
import random


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from data.prepare_dataset import prepare_dataset
from utils.utils import (
    load_config,
    set_random_seed,
    optimize_model_memory,
    setup_logging,
)
from grpo.custom_reward_function import combined_reward
from grpo.generation_interrupt_new import CustomModel
from grpo.grpo_trainer import train_with_grpo, evaluate_model
from archive.grpo_trainer_mu_GPU import train_with_grpo_mu_GPU


def main():
    """
    Main function to run the training and evaluation pipeline.

    Steps:
    1. Load config and setup environment
    2. Initialize model and tokenizer
    3. Evaluate initial model (optional)
    4. Fine-tune with GRPO
    5. Evaluate final model
    6. Save results
    """
    # 1. Setup environment
    config = load_config("config/config.yaml")

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    checkpoint_dir = Path(f"checkpoint/{config.exp}/{time_str}")
    output_dir = Path(f"output/{config.exp}/{time_str}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    set_random_seed(config.random_seed)
    logger.info(f"Set random seed to {config.random_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Initialize model and tokenizer
    logger.info("Loading model...")
    torch_dtype = getattr(torch, config.model.torch_dtype)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name, torch_dtype=torch_dtype, device_map=config.model.device_map
    ).to(device)
    logger.info("Model loaded successfully")

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=config.lora_config.r,
        lora_alpha=config.lora_config.lora_alpha,
        target_modules=config.lora_config.target_modules,
        lora_dropout=config.lora_config.lora_dropout,
        bias=config.lora_config.bias,
        task_type=config.lora_config.task_type,
    )
    model = get_peft_model(model, lora_cfg)
    logger.info(f"Applied LoRA configuration")

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id

    # Prepare dataset
    all_data = prepare_dataset("train", config.dataset.name)
    random.shuffle(all_data)
    eval_size = config.dataset.num_eval_examples
    train_data = all_data[eval_size:]
    eval_data = all_data[:eval_size]
    logger.info(f"Dataset split - Train: {len(train_data)}, Eval: {len(eval_data)}")

    # 3. Initial evaluation
    results = {}
    if config.evaluation_before_grpo:
        logger.info("Evaluating initial model...")
        evaluation_results = evaluate_model(model, tokenizer, eval_data, device)
        evaluation_before_grpo = output_dir / "evaluation_before_grpo.json"
        with open(evaluation_before_grpo, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        # 过滤条目 predicted 是 None 的
        evaluation_results = [
            item for item in evaluation_results if item["predicted"] is not None
        ]

        evaluation_before_grpo_filtered = output_dir / "evaluation_before_grpo_filtered.json"
        with open(evaluation_before_grpo_filtered, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        pre_grpo_accuracy = 0
        logger.info(f"Initial accuracy: {pre_grpo_accuracy:.2f}%")
        results["pre_grpo_accuracy"] = pre_grpo_accuracy

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    exit()

    # 4. GRPO fine-tuning
    logger.info("Starting GRPO fine-tuning...")
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

    # Optimize model memory usage
    model = optimize_model_memory(model)
    custom_model = CustomModel(model, tokenizer)

    # Setup multi-GPU if available
    device_ids = None
    if config.training.mu_gpu and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        logger.info(f"Using {len(device_ids)} GPUs for training")

    # Select training function based on GPU configuration
    train_func = train_with_grpo_mu_GPU if config.training.mu_gpu else train_with_grpo
    # Run training
    model = train_func(
        model=custom_model,
        tokenizer=tokenizer,
        train_data=train_data,
        device_ids=device_ids,
        **training_config,
    )

    # 5. Final evaluation
    if config.evaluation_after_grpo:
        logger.info("Evaluating fine-tuned model...")
        post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
        logger.info(f"Final accuracy: {post_grpo_accuracy:.2f}%")

        results["post_grpo_accuracy"] = post_grpo_accuracy

        if config.evaluation_before_grpo:
            improvement = post_grpo_accuracy - results["pre_grpo_accuracy"]
            logger.info(f"Improvement: {improvement:.2f}%")
            results["improvement"] = improvement

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # 6. Save model
    logger.info(f"Saving model to {checkpoint_dir}")
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


if __name__ == "__main__":
    main()
