from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from dotenv import load_dotenv
from rich.traceback import install
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prepare_dataset import prepare_dataset
from data.prompt import LLM_EVAL_PROMPT
from models.evaluater import evaluate
from utils.utils import (
    load_config,
    optimize_model_memory,
    set_random_seed,
    setup_logging,
)


def main() -> None:
    load_dotenv()
    install()

    accelerator: Accelerator = Accelerator()
    config: Any = load_config("config/config.yaml")
    config.dataset.num_eval = 100

    output_dir: Path = Path(f"output_eval/pre_nosearch/{config.exp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, level=logging.INFO)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)

    set_random_seed(config.random_seed)
    logging.info("Set random seed to %d", config.random_seed)

    device: torch.device = accelerator.device
    logging.info("Using device: %s", device)

    _, eval_dataset = prepare_dataset(
        split="train",
        name=config.dataset.name,
        eval_size=config.dataset.num_eval,
    )
    if len(eval_dataset) == 0:
        raise ValueError("Evaluation dataset is empty.")
    eval_dataloader: DataLoader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    logging.info("Eval dataloader size: %d", len(eval_dataloader))

    logging.info("Loading model...")
    torch_dtype: torch.dtype = getattr(torch, config.model.torch_dtype)
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    logging.info("Model loaded successfully")

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id

    optimized_model: AutoModelForCausalLM = optimize_model_memory(model)

    prepared_model, prepared_dataloader = accelerator.prepare(
        optimized_model,
        eval_dataloader,
    )

    logging.info("Starting evaluation...")
    evaluate(
        model=prepared_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        dataloader=prepared_dataloader,
        device=device,
        output_dir=output_dir,
        evaluation_before_grpo=True,
        evaluation_after_grpo=False,
        LLM_EVAL_PROMPT=LLM_EVAL_PROMPT,
    )
    logging.info("Evaluation completed. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()