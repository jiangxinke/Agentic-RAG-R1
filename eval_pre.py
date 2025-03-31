import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prepare_dataset import prepare_dataset
from utils.utils import load_config, set_random_seed, optimize_model_memory, setup_logging
from grpo.model import CustomModel
from grpo.evaluater import evaluate
from data.prompt import LLM_EVAL_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup environment
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"eval_output/pre/test1000-{config.exp}")
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

    model = AutoModelForCausalLM.from_pretrained(config.model.name, torch_dtype=torch_dtype, trust_remote_code=True)
    model = model.to(device)
    logging.info("Model loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id

    # Optimize model memory usage
    model = optimize_model_memory(model)

    # # Wrap model with CustomModel for evaluation
    # custom_model = CustomModel(model, tokenizer)

    # Run evaluation
    logging.info("Starting evaluation...")
    evaluate(
        model,
        tokenizer,
        eval_dataloader,
        device,
        output_dir,
        evaluation_before_grpo=True,
        evaluation_after_grpo=False,
        LLM_EVAL_PROMPT=LLM_EVAL_PROMPT,
    )

    logging.info(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
