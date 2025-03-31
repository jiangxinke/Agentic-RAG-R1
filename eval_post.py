import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

from data.prepare_dataset import prepare_dataset
from utils.utils import load_config, set_random_seed, optimize_model_memory, setup_logging
from grpo.model import CustomModel
from grpo.evaluater import evaluate
from data.prompt import LLM_EVAL_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    # parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved LoRA checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    args.checkpoint_path = "/home/haoyu.zhang/jxk/gjr/250329/checkpoint/xiaobeir1-Qwen2.5-7B-Instruct/step-0060"
    # Setup environment
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"eval_output/post/{config.exp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, level=logging.INFO)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    set_random_seed(config.random_seed)
    logging.info(f"Set random seed to {config.random_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Prepare dataset
    _, eval_dataset = prepare_dataset("train", config.dataset.name, eval_size=config.dataset.num_eval)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    logging.info(f"Eval dataloader: {len(eval_dataloader)}")

    # Initialize model and tokenizer
    logging.info("Loading model...")
    torch_dtype = getattr(torch, config.model.torch_dtype)

    base_model = AutoModelForCausalLM.from_pretrained(config.model.name, torch_dtype=torch_dtype, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = base_model.config.eos_token_id = tokenizer.eos_token_id

    # Apply LoRA configuration
    if config.training.use_lora:
        logging.info(f"Applying LoRA configuration...")
        lora_cfg = LoraConfig(
            r=config.lora_config.r,
            lora_alpha=config.lora_config.lora_alpha,
            target_modules=config.lora_config.target_modules,
            lora_dropout=config.lora_config.lora_dropout,
            bias=config.lora_config.bias,
            task_type=config.lora_config.task_type,
        )

        # Load the saved LoRA weights
        logging.info(f"Loading LoRA weights from {args.checkpoint_path}")
        base_model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
        logging.info("LoRA weights loaded successfully")
    else:
        logging.warning("Not using LoRA but checkpoint path was provided")

    base_model = base_model.to(device)

    # Optimize model memory usage
    base_model = optimize_model_memory(base_model)

    # Wrap the model with CustomModel
    model = CustomModel(base_model, tokenizer)

    # Run evaluation
    logging.info("Starting evaluation...")
    evaluate(
        model,
        tokenizer,
        eval_dataloader,
        device,
        output_dir,
        evaluation_before_grpo=False,
        evaluation_after_grpo=True,
        LLM_EVAL_PROMPT=LLM_EVAL_PROMPT,
    )

    logging.info(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
