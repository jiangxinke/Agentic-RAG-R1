from rich.traceback import install
from dotenv import load_dotenv

load_dotenv()
install()

from utils.saver import ModelSaver
import os
import json
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import time
import datetime
import torch
import pdb
import logging
import swanlab
from swanlab.integration.accelerate import SwanLabTracker
from accelerate.logging import get_logger

# pdb.set_trace = lambda *args, **kwargs: None  # 将这个函数变为空

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

from data.prepare_dataset import prepare_dataset
from utils.utils import (
    load_config,
    set_random_seed,
    optimize_model_memory,
    setup_logging,
)
from grpo.reward_function import combined_reward
from grpo.model import CustomModel
from grpo.evaluater import evaluate
from accelerate import Accelerator

# Import train_with_grpo after other imports to avoid circular imports
from grpo.trainer import train_with_grpo


def main(config):
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
    # Initialize accelerator first
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        swanlab.init(
            project=config.project_name,
            experiment_name=config.exp,
            config=config.__dict__,
        )

    # 1. Setup environment

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    today = now.strftime("%Y-%m-%d")
    checkpoint_dir = Path(f"checkpoint/{config.exp}/{today}")
    output_dir = Path(f"output_train/{config.exp}/{time_str}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, level=logging.WARNING)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    set_random_seed(config.random_seed)
    logging.info(f"Set random seed to {config.random_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset("train", config.dataset.name, eval_size=config.dataset.num_eval)
    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    logging.info(f"Train dataloader: {len(train_dataloader)}, Eval dataloader: {len(eval_dataloader)}")

    # 2. Initialize model and tokenizer
    logging.info("Loading model...")
    torch_dtype = getattr(torch, config.model.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(config.model.name, torch_dtype=torch_dtype, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(config.model.name, torch_dtype=torch_dtype, trust_remote_code=True)
    model = model.to(device)
    ref_model = ref_model.to(device)
    logging.info("Model loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    ref_model.config.pad_token_id = ref_model.config.eos_token_id = tokenizer.eos_token_id

    # Apply LoRA
    ## TODO standard lora
    if config.training.use_lora:
        lora_cfg = LoraConfig(
            r=config.lora_config.r,
            lora_alpha=config.lora_config.lora_alpha,
            target_modules=config.lora_config.target_modules,
            lora_dropout=config.lora_config.lora_dropout,
            bias=config.lora_config.bias,
            task_type=config.lora_config.task_type,
        )
        if not config.training.continue_training:
            model = get_peft_model(model, lora_cfg)
            ref_model = get_peft_model(ref_model, lora_cfg)
        else:
            weights_path = f"checkpoint/{config.exp}/2025-04-06/step-{config.training.current_step:04d}"
            model = PeftModel.from_pretrained(model, weights_path, config=lora_cfg, is_trainable=True)
            ref_model = PeftModel.from_pretrained(ref_model, weights_path, config=lora_cfg, is_trainable=True)
            logging.info(f"Continue training from {weights_path}")
        logging.info(f"Using lora:\n {lora_cfg}")
        model.print_trainable_parameters()
    else:
        logging.info("Not using LoRA")

    # 4. GRPO fine-tuning
    logging.info("Starting GRPO fine-tuning...")
    training_config = {
        "num_iterations": config.training.num_iterations,
        "steps_per_iteration": config.training.steps_per_iteration,
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
    ref_model = optimize_model_memory(ref_model)

    custom_model = CustomModel(model, tokenizer)
    ref_custom_model = CustomModel(ref_model, tokenizer)
    model_saver = ModelSaver()
    # Run training
    start_time = time.time()
    if config.training.continue_training:
        current_step = config.training.current_step
    else:
        current_step = 0

    logger = get_logger(__name__)
    model = train_with_grpo(
        policy_model=custom_model,
        reference_model=ref_custom_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        dataloader=train_dataloader,
        model_saver=model_saver,
        checkpoint_dir=checkpoint_dir,
        current_step=current_step,
        save_interval=config.save.save_interval,
        **training_config,
    )
    end_time = time.time()
    training_time = end_time - start_time
    logging.info(f"Training completed, total time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # 确保所有进程在继续之前等待训练完成
    accelerator.wait_for_everyone()

    # 确保所有进程在退出前等待主进程完成评估
    # accelerator.wait_for_everyone()

    # # 6. Save model
    # if accelerator.is_local_main_process:
    #     # FIXME 拿不到lora矩阵
    #     model.save_pretrained(checkpoint_dir)
    #     tokenizer.save_pretrained(checkpoint_dir)


if __name__ == "__main__":
    config = load_config("config/config.yaml")
    main(config)
