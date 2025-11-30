from dotenv import load_dotenv
from rich.traceback import install

load_dotenv()
install()

import json
import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.prepare_dataset import prepare_dataset
from src.core.model import AgenticRAGModel
from src.eval.metrics_driver import safe_export
from src.utils.utils import (load_config, optimize_model_memory,
                             set_random_seed, setup_logging)


def main():
    # Setup environment
    config = load_config("src/config/config.yaml")

    output_dir: Path = Path(f"output_eval/infer/{config.exp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, level=logging.INFO)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=4)

    set_random_seed(config.experiment.random_seed)
    logging.info(f"Set random seed to {config.experiment.random_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    _, eval_dataset = prepare_dataset("train", config.dataset.name, eval_size=config.dataset.num_eval)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    logging.info(f"Eval dataloader: {len(eval_dataloader)}")

    # Initialize model and tokenizer
    logging.info("Loading model...")

    # with init_empty_weights():
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        trust_remote_code=True,
    )

    base_model = base_model.to(device)
    # reference_base_model = reference_base_model.to(device)
    logging.info("Base model loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = base_model.config.eos_token_id = tokenizer.eos_token_id
    # reference_base_model.config.pad_token_id = reference_base_model.config.eos_token_id = tokenizer.eos_token_id
    logging.info("Tokenizer loaded successfully")

    optimized_model: AutoModelForCausalLM = optimize_model_memory(base_model)
    retrieval_model = AgenticRAGModel(optimized_model, tokenizer)
    logging.info("AgenticRAGModel loaded successfully")


    accelerator: Accelerator = Accelerator()
    prepared_model, prepared_dataloader = accelerator.prepare(
        retrieval_model,
        eval_dataloader,
    )
    
    try:
        exp_dir = Path(f"experiments/training/{config.experiment.name}")
        latest = sorted(exp_dir.glob("*/"))[-1] if exp_dir.exists() else None
        metrics_path = latest / "metrics.jsonl" if latest else None
        if metrics_path and metrics_path.exists():
            safe_export(str(metrics_path))
            logging.info("Metrics exported to %s", latest)
        else:
            logging.info("No metrics.jsonl found under %s", exp_dir)
    except Exception:
        logging.warning("Failed to export metrics")




if __name__ == "__main__":
    main()
