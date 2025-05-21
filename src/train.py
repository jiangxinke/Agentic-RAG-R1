from dotenv import load_dotenv
from rich import print
from rich.logging import RichHandler
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
from accelerate import Accelerator, init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.prepare_dataset import prepare_dataset
from src.models.critic import AgenticRAGCritic
from src.models.model import AgenticRAGModel
from src.models.reward import overall_reward
from src.models.reward_token_level import overall_reward_token_level
from src.models.trainer import train_with_grpo, train_with_ppo, train_with_sft
from src.utils.utils import (
    load_config,
    optimize_model_memory,
    set_random_seed,
    setup_logging,
)


def main():
    # Setup environment
    config = load_config("src/config/config.yaml")

    accelerator = Accelerator()
    if accelerator.is_local_main_process and config.swanlab:
        swanlab.init(
            project=config.project.name,
            experiment_name=config.experiment.name,
            config=config.__dict__,
        )

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    today = now.strftime("%Y-%m-%d")
    checkpoint_dir = Path(f"checkpoints/{config.experiment.name}/{today}")
    output_dir = Path(f"experiments/training/{config.experiment.name}/{time_str}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, level=logging.INFO)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)
    logging.info(f"Saving config to {output_dir / 'config.json'}")

    set_random_seed(config.experiment.random_seed)
    logging.info(f"Set random seed to {config.experiment.random_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_dataset, eval_dataset = prepare_dataset("train", config.dataset.name, eval_size=config.dataset.num_eval)
    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    logging.info(f"Train dataloader: {len(train_dataloader)}, Eval dataloader: {len(eval_dataloader)}")

    # Initialize model and tokenizer
    logging.info("Loading model...")

    # with init_empty_weights():
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        trust_remote_code=True,
        # empty_init=True
    )

    reference_base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        trust_remote_code=True,
        # empty_init=True
    )

    base_model = base_model.to(device)
    reference_base_model = reference_base_model.to(device)
    logging.info("Base model loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = base_model.config.eos_token_id = tokenizer.eos_token_id
    reference_base_model.config.pad_token_id = reference_base_model.config.eos_token_id = tokenizer.eos_token_id
    logging.info("Tokenizer loaded successfully")

    # lora
    if config.training.use_lora:
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )
        if not config.training.continue_training:
            base_model = get_peft_model(base_model, lora_config)
            reference_base_model = get_peft_model(reference_base_model, lora_config)
        else:
            weights_path = f"checkpoints/{config.experiment.name}/step-{config.training.current_step:04d}"
            base_model = PeftModel.from_pretrained(base_model, weights_path, config=lora_config, is_trainable=True)
            reference_base_model = PeftModel.from_pretrained(
                reference_base_model, weights_path, config=lora_config, is_trainable=True
            )
            logging.info(f"Continue training from {weights_path}")
        logging.info(f"Using lora:\n {lora_config}")

        base_model.print_trainable_parameters()
        reference_base_model.print_trainable_parameters()

    else:
        logging.info("Not using LoRA")

    # quant
    if config.training.use_quant:
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=config.qlora.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),  # optional
            bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,  # optional
            bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,  # optional
            load_in_8bit=config.qlora.load_in_8bit,  # enable 8bit quantization
            llm_int8_threshold=config.qlora.llm_int8_threshold,  # if load_in_8bit is True
        )

        base_model = load_and_quantize_model(base_model, bnb_quantization_config=bnb_quantization_config, device_map="auto")

        reference_base_model = load_and_quantize_model(
            reference_base_model, bnb_quantization_config=bnb_quantization_config, device_map="auto"
        )

        logging.info(f"Using Quant: {config.qlora}")
    else:
        bnb_quantization_config = None
        logging.info("Not using Quant")

    # GRPO fine-tuning
    logging.info("Starting GRPO fine-tuning...")

    if config.training.reward_token_level:
        reward_func = overall_reward_token_level
        logging.info("Using token-level reward")
    else:
        reward_func = overall_reward
        logging.info("Using normal reward")

    training_config = {
        "use_KV_Cache": config.training.use_KV_Cache,
        "num_iterations": config.training.num_iterations,
        "steps_per_iteration": config.training.steps_per_iteration,
        "num_generations": config.training.generation.num_generations,
        "max_new_tokens": config.training.generation.max_new_tokens,
        "max_length_for_gather": config.training.generation.max_length_for_gather,
        "max_generate_iterations": config.training.generation.max_generate_iterations,
        "temperature": config.training.generation.temperature,
        "do_sample": config.training.generation.do_sample,
        "beta": config.training.optimizer.beta,
        "learning_rate": config.training.learning_rate,
        "mu": config.training.optimizer.mu,
        "epsilon": config.training.optimizer.epsilon,
        "reward_function": reward_func,
        "save_interval": config.training.save_interval,
        "use_diverse_sampling": config.training.generation.use_diverse_sampling,
        "diversity_penalty": config.training.generation.diversity_penalty,
    }
    logging.info(f"Training config: {training_config}")

    # Optimize model memory usage
    base_model = optimize_model_memory(base_model)
    reference_base_model = optimize_model_memory(reference_base_model)

    policy_model = AgenticRAGModel(base_model, tokenizer)
    reference_model = AgenticRAGModel(reference_base_model, tokenizer)
    logging.info("AgenticRAGModel loaded successfully")

    if config.training.continue_training:
        current_step = config.training.current_step
    else:
        current_step = 0

    if getattr(config.training, "train_method", "grpo") == "grpo":
        train_with_grpo(
            config=config,
            device=device,
            policy_model=policy_model,
            ref_base_model=reference_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            dataloader=train_dataloader,
            checkpoint_dir=checkpoint_dir,
            current_step=current_step,
            **training_config,
        )
    elif config.training.train_method == "ppo":
        # 初始化Critic模型
        critic_model = AgenticRAGCritic(
            model_name=config.model.critic_name,
            device=device,
            torch_dtype=config.model.torch_dtype
        )
        train_with_ppo(
            config=config,
            device=device,
            policy_model=policy_model,
            critic_model=critic_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            dataloader=train_dataloader,
            checkpoint_dir=checkpoint_dir,
            current_step=current_step,
            **training_config,
        )
    elif config.training.train_method == "sft":
        train_with_sft(
            config=config,
            device=device,
            model=policy_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            dataloader=train_dataloader,
            checkpoint_dir=checkpoint_dir,
            current_step=current_step,
            num_epochs=getattr(config.training, "num_epochs", 1),
            steps_per_epoch=getattr(config.training, "steps_per_epoch", 500),
            learning_rate=config.training.learning_rate,
            save_interval=config.training.save_interval,
        )
    else:
        raise ValueError(f"Unknown train_method: {config.training.train_method}")
    logging.info("Training completed")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
