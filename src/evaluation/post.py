import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel
from rich.traceback import install
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import (
    load_config,
    optimize_model_memory,
    set_random_seed,
    setup_logging,
)

from ..data.prepare_dataset import prepare_dataset
from ..data.prompt import LLM_EVAL_PROMPT
from ..models.evaluater import evaluate
from ..models.model import CustomModel

load_dotenv()
install()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluation.

    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            date (str): Date of the checkpoint directory.
            checkpoint_step (int): Checkpoint step number.
            num_eval (int): Number of examples to evaluate.
    """
    parser = argparse.ArgumentParser(description="Evaluate model after GRPO training")
    parser.add_argument("--date", type=str, required=True, help="Date of the checkpoint (e.g., 2025-04-16)")
    parser.add_argument("--checkpoint_step", type=int, required=True, help="Step number of the checkpoint to load")
    parser.add_argument("--num_eval", type=int, default=1000, help="Number of examples to evaluate")
    return parser.parse_args()


def main() -> None:
    """
    Entrypoint for post-GRPO model evaluation.

    This function:
      1. Parses CLI arguments.
      2. Loads configuration and sets up logging and random seed.
      3. Prepares the evaluation dataset and dataloader.
      4. Loads the model checkpoint with optional LoRA weights.
      5. Runs evaluation and saves result files.

    Raises:
        FileNotFoundError: If the specified checkpoint path does not exist.
        IOError: If writing configuration or results to disk fails.
    """
    # 1. Parse arguments
    args: argparse.Namespace = parse_args()

    # 2. Setup accelerator and configuration
    accelerator: Accelerator = Accelerator()
    config: Any = load_config("config/config.yaml")
    config.dataset.num_eval = args.num_eval

    # 3. Determine checkpoint and output directories
    checkpoint_dir = Path(f"./checkpoint/{config.exp}/{args.date}")
    checkpoint_path = checkpoint_dir / f"step-{args.checkpoint_step:04d}"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    output_dir: Path = Path(f"output_eval/post/{config.exp}/{args.date}/step-{args.checkpoint_step:04d}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Initialize logging and save config
    setup_logging(output_dir, level=logging.INFO)
    try:
        with open(output_dir / "config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2)
    except Exception as err:
        raise IOError(f"Failed to save config: {err}")

    logging.info(f"Set random seed to {config.random_seed}")
    set_random_seed(config.random_seed)

    # 5. Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 6. Prepare evaluation dataset
    _, eval_dataset = prepare_dataset(
        split="train",
        name=config.dataset.name,
        eval_size=config.dataset.num_eval,
    )
    eval_dataloader: DataLoader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    logging.info(f"Eval dataloader size: {len(eval_dataloader)} examples")

    # 7. Load base model and tokenizer
    logging.info("Loading base model and tokenizer...")
    torch_dtype = getattr(torch, config.model.torch_dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = tokenizer.eos_token_id

    # 8. Apply LoRA weights if configured
    if config.training.use_lora:
        logging.info("Applying LoRA configuration...")
        lora_cfg = LoraConfig(
            r=config.lora_config.r,
            lora_alpha=config.lora_config.lora_alpha,
            target_modules=config.lora_config.target_modules,
            lora_dropout=config.lora_config.lora_dropout,
            bias=config.lora_config.bias,
            task_type=config.lora_config.task_type,
        )
        logging.info(f"Loading LoRA weights from {checkpoint_path}")
        base_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        logging.info("LoRA weights loaded successfully")
    else:
        logging.warning("LoRA not enabled; using base model only.")

    # 9. Move model to device and optimize memory
    base_model = base_model.to(device)
    base_model = optimize_model_memory(base_model)

    # 10. Wrap with CustomModel and prepare accelerator
    model = CustomModel(base_model, tokenizer)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    # 11. Run evaluation
    logging.info("Starting evaluation...")
    evaluate(
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        eval_dataloader=eval_dataloader,
        device=device,
        output_dir=output_dir,
        evaluation_before_grpo=False,
        evaluation_after_grpo=True,
        LLM_EVAL_PROMPT=LLM_EVAL_PROMPT,
    )

    logging.info(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
