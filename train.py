from dotenv import load_dotenv
from rich.traceback import install

load_dotenv()
install()

import datetime
import json
import logging
import os
import pdb
import time
from pathlib import Path

import deepspeed
import swanlab
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from peft import LoraConfig, PeftModel, get_peft_model
from swanlab.integration.accelerate import SwanLabTracker
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data.prepare_dataset import prepare_dataset
from model.evaluater import evaluate
from model.model import CustomModel
from model.reward_function import combined_reward
from model.trainer import train_with_grpo
from utils.saver import ModelSaver
from utils.utils import (
    load_config,
    optimize_model_memory,
    set_random_seed,
    setup_logging,
)


def main():
    # 1. Setup environment
    config = load_config("config/config.yaml")

    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        swanlab.init(
            project=config.project_name,
            experiment_name=config.exp,
            config=config.__dict__,
        )

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

    # qlora
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_storage=torch.bfloat16,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    # pdb.set_trace()
    # params = [p for name, p in model.named_parameters()]
    # with deepspeed.zero.GatheredParameters(params, enabled=True):
    #     model_state_dict = model.state_dict()
    #     state_dict = {k: v for k, v in model_state_dict.items()}

    # # model1 = accelerator.unwrap_model(model)
    # # model1 = model.unwrap_model()
    # model_dict = accelerator.get_state_dict(model)
    model = model.to(device)
    ref_model = ref_model.to(device)
    logging.info("Model loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    ref_model.config.pad_token_id = ref_model.config.eos_token_id = tokenizer.eos_token_id

    # print(f"tokenizer.decode([522, 1836, 29]): ->{tokenizer.decode([522, 1836, 29])}<-")
    # print(f"tokenizer.decode([522, 1836, 397]): ->{tokenizer.decode([522, 1836, 397])}<-")

    # observation_start_token_id = tokenizer("<observation>").input_ids[0]
    # observation_end_token_id = tokenizer("</observation>").input_ids[0]

    # print(f"observation_start_token_id: {observation_start_token_id},")
    # print(f"observation_end_token_id: {observation_end_token_id}")

    # print(f"tokenizer.decode({observation_end_token_id}): ->{tokenizer.decode([observation_end_token_id])}<-")
    # print(f"tokenizer.decode({observation_start_token_id}): ->{tokenizer.decode([observation_start_token_id])}<-")

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
        "max_new_tokens": config.training.max_new_tokens,
        "max_length_for_gather": config.training.max_length_for_gather,
        "temperature": config.training.temperature,
        "do_sample": config.training.do_sample,
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

    train_with_grpo(
        policy_model=custom_model,
        base_reference_model=ref_custom_model,
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
    accelerator.end_training()


if __name__ == "__main__":
    main()
