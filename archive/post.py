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


def strip_prefix(state_dict: dict[str, torch.Tensor], pattern: str = r"^(?:model\.|base_model\.)+") -> dict[str, torch.Tensor]:
    """Remove leading ``model.`` / ``base_model.`` (possibly repeated) from keys."""
    return {re.sub(pattern, "", k): v for k, v in state_dict.items()}


def load_lora_weights(base_model: torch.nn.Module, checkpoint_path: Path, lora_cfg: LoraConfig) -> torch.nn.Module:
    """Try to load LoRA adapter; if direct load fails, strip prefixes and retry."""

    try:
        logging.info("Attempting direct `PeftModel.from_pretrained()` …")
        return PeftModel.from_pretrained(base_model, str(checkpoint_path), config=lora_cfg)
    except Exception as err:  # noqa: BLE001  (broad but intentional)
        logging.warning(f"Direct load failed ({err}). Falling back to manual load with prefix‑stripping…")

    adapter_file = checkpoint_path / "adapter_model.bin"
    if not adapter_file.exists():
        raise FileNotFoundError(f"Expected adapter file not found: {adapter_file}")

    peft_model = PeftModel(base_model, lora_cfg)
    raw_sd = torch.load(adapter_file, map_location="cpu")
    clean_sd = strip_prefix(raw_sd)
    missing, unexpected = peft_model.load_state_dict(clean_sd, strict=False)
    logging.info("LoRA weights loaded with missing=%d, unexpected=%d", len(missing), len(unexpected))
    return peft_model


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
    checkpoint_dir = Path(f"./checkpoints/{config.experiment.name}/{args.date}")
    checkpoint_path = checkpoint_dir / f"step-{args.checkpoint_step:04d}"
    checkpoint_path.exists() or (_ := (_ for _ in ()).throw(FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")))

    output_dir = Path(f"experiments/post/{config.experiment.name}/{args.date}/step-{args.checkpoint_step:04d}")
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

    # 7. LoRA ---------------------------------------------------------------
    if config.training.use_lora:
        logging.info("Applying LoRA configuration…")
        lora_cfg = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )
        logging.info("Loading LoRA weights from %s", checkpoint_path)
        base_model = load_lora_weights(base_model, checkpoint_path, lora_cfg)
    else:
        logging.warning("LoRA not enabled; using base model only.")

    # 8. Quantization --------------------------------------------------------
    if config.training.use_quant:
        bnb_quant_cfg = BnbQuantizationConfig(
            load_in_4bit=config.qlora.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
            load_in_8bit=config.qlora.load_in_8bit,
            llm_int8_threshold=config.qlora.llm_int8_threshold,
        )
        base_model = load_and_quantize_model(base_model, bnb_quantization_config=bnb_quant_cfg, device_map="auto")
        logging.info("Quantization applied: %s", config.qlora)
    else:
        logging.info("Not using quantization.")

    # 9. Device + memory tweaks ---------------------------------------------
    base_model.to(device)
    base_model = optimize_model_memory(base_model)

    # 10. Wrap + prepare -----------------------------------------------------
    model = AgenticRAGModel(base_model, tokenizer)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    # 11. Evaluate -----------------------------------------------------------
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
