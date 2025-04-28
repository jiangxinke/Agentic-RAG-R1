import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel
from rich.traceback import install
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.prepare_dataset import prepare_dataset
from src.data.prompt import LLM_EVAL_PROMPT
from src.models.evaluater import evaluate
from src.models.model import AgenticRAGModel
from src.utils.utils import (
    load_config,
    optimize_model_memory,
    set_random_seed,
    setup_logging,
)

load_dotenv()
install()


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate model after GRPO training")
    parser.add_argument("--date", required=True, help="Date of the checkpoint (e.g., 2025‑04‑16)")
    parser.add_argument("--checkpoint_step", type=int, required=True, help="Step number of the checkpoint to load")
    parser.add_argument("--num_eval", type=int, default=1000, help="Number of examples to evaluate")
    return parser.parse_args()


def main() -> None:  # noqa: C901  (function is long but clear)
    # 1. Args / config --------------------------------------------------------
    args = parse_args()
    accelerator = Accelerator()
    config: Any = load_config("src/config/config.yaml")
    config.dataset.num_eval = args.num_eval

    # 2. Paths ---------------------------------------------------------------
    output_dir = Path(f"experiments/pre_search/{config.model.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Logging / random seed ----------------------------------------------
    setup_logging(output_dir, level=logging.INFO)
    with (output_dir / "config.json").open("w") as fp:
        json.dump(config.__dict__, fp, indent=2)
    logging.info("Set random seed to %s", config.experiment.random_seed)
    set_random_seed(config.experiment.random_seed)

    # 4. Device --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # 5. Dataset -------------------------------------------------------------
    _, eval_dataset = prepare_dataset(
        split="train",
        name=config.dataset.name,
        eval_size=config.dataset.num_eval,
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    logging.info("Eval dataloader size: %d examples", len(eval_dataloader))

    # 6. Base model + tokenizer ---------------------------------------------
    logging.info("Loading base model and tokenizer…")
    torch_dtype = getattr(torch, config.model.torch_dtype)

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = tokenizer.eos_token_id

    # 7. Device + memory tweaks ---------------------------------------------
    base_model.to(device)
    base_model = optimize_model_memory(base_model)

    # 8. Wrap + prepare -----------------------------------------------------
    model = base_model
    # model = AgenticRAGModel(model, tokenizer)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    # 9. Evaluate -----------------------------------------------------------
    logging.info("Starting evaluation…")
    evaluate(
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        eval_dataloader=eval_dataloader,
        device=device,
        output_dir=output_dir,
        evaluation_before_grpo=False,
        evaluation_after_grpo=True,
        llm_eval_prompt=LLM_EVAL_PROMPT,
    )
    logging.info("Evaluation completed. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
