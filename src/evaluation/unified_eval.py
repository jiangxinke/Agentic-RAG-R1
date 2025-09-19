#!/usr/bin/env python3
"""
Unified Evaluation Script for Agentic RAG Models

This script provides a unified interface for evaluating both:
1. Pre-training models (base model with/without search capabilities)
2. Post-training models (LoRA fine-tuned models)

Usage:
    # Evaluate base model without search
    python unified_eval.py --mode pre_no_search --num_eval 100

    # Evaluate base model with search
    python unified_eval.py --mode pre_search --num_eval 100

    # Evaluate post-training LoRA model
    python unified_eval.py --mode post --date 2025-04-16 --checkpoint_step 150 --num_eval 1000
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from accelerate import Accelerator
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel
from rich.traceback import install
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.prepare_dataset import prepare_dataset
from src.data.prompt import LLM_EVAL_PROMPT
from src.models.evaluater import evaluate
from src.models.model import AgenticRAGModel
from src.utils.extractor import extract_answer_from_model_output
from src.utils.utils import (
    load_config,
    optimize_model_memory,
    set_random_seed,
    setup_logging,
)

# Initialize environment
load_dotenv()
install()


# ============================================================================
# Utility Functions
# ============================================================================

def strip_prefix(state_dict: dict[str, torch.Tensor], pattern: str = r"^(?:model\.|base_model\.)+") -> dict[str, torch.Tensor]:
    """Remove leading model/base_model prefixes from state dict keys."""
    return {re.sub(pattern, "", k): v for k, v in state_dict.items()}


def load_lora_weights(base_model: torch.nn.Module, checkpoint_path: Path, lora_cfg: LoraConfig) -> torch.nn.Module:
    """Load LoRA adapter weights with fallback to manual loading."""
    # 首先尝试直接加载，如果失败再使用手动加载
    try:
        logging.info("Attempting direct PeftModel.from_pretrained()...")
        return PeftModel.from_pretrained(base_model, str(checkpoint_path), config=lora_cfg)
    except Exception as err:
        logging.warning(f"Direct load failed ({err}). Falling back to manual load with prefix stripping...")
    
    # 查找适配器文件，优先使用 .safetensors，然后是 .bin
    adapter_file = None
    for ext in [".safetensors", ".bin"]:
        potential_file = checkpoint_path / f"adapter_model{ext}"
        if potential_file.exists():
            adapter_file = potential_file
            break
    
    if adapter_file is None:
        raise FileNotFoundError(f"No adapter file found in {checkpoint_path}. Expected adapter_model.safetensors or adapter_model.bin")

    logging.info(f"Loading adapter weights from: {adapter_file}")
    peft_model = PeftModel(base_model, lora_cfg)
    
    # 根据文件格式加载权重
    if adapter_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        raw_sd = load_file(adapter_file)
    else:
        raw_sd = torch.load(adapter_file, map_location="cpu")
    
    # 打印前几个原始键来调试
    raw_keys = list(raw_sd.keys())
    logging.info("First 5 raw state dict keys: %s", raw_keys[:5])
    logging.info("Total raw keys count: %d", len(raw_keys))
    
    # 获取 peft_model 期望的键
    expected_keys = list(peft_model.state_dict().keys())
    logging.info("First 5 expected keys: %s", expected_keys[:5])
    logging.info("Total expected keys count: %d", len(expected_keys))
    
    # 尝试不同的键匹配策略
    strategies = [
        ("原始键", raw_sd),
        ("strip_prefix", strip_prefix(raw_sd)),
    ]
    
    # 如果原始键没有 base_model 前缀，尝试添加
    if raw_keys and not raw_keys[0].startswith("base_model"):
        prefixed_sd = {f"base_model.{k}": v for k, v in raw_sd.items()}
        strategies.append(("添加base_model前缀", prefixed_sd))
    
    best_strategy = None
    best_missing_count = float('inf')
    
    for strategy_name, state_dict in strategies:
        missing, unexpected = peft_model.load_state_dict(state_dict, strict=False)
        missing_count = len(missing)
        logging.info("策略 '%s': missing=%d, unexpected=%d", strategy_name, missing_count, len(unexpected))
        
        if missing_count < best_missing_count:
            best_missing_count = missing_count
            best_strategy = (strategy_name, state_dict, missing, unexpected)
    
    # 使用最佳策略
    strategy_name, best_state_dict, missing, unexpected = best_strategy
    logging.info("使用最佳策略: %s", strategy_name)
    
    # 重新加载使用最佳策略
    peft_model.load_state_dict(best_state_dict, strict=False)
    
    print("=================")
    logging.info("Missing keys count: %d", len(missing))
    if missing and len(missing) <= 5:
        logging.warning("Missing keys: %s", missing)
    elif missing:
        logging.warning("First 5 missing keys: %s", missing[:5])
    
    print("=================")
    logging.info("Unexpected keys count: %d", len(unexpected))
    if unexpected and len(unexpected) <= 5:
        logging.warning("Unexpected keys: %s", unexpected)
    elif unexpected:
        logging.warning("First 5 unexpected keys: %s", unexpected[:5])
    
    print("=================")
    logging.info("LoRA weights loaded with missing=%d, unexpected=%d", len(missing), len(unexpected))
    return peft_model


# ============================================================================
# Model Setup Functions
# ============================================================================

def setup_model_and_tokenizer(config: Any, mode: str, checkpoint_path: Optional[Path] = None) -> tuple[torch.nn.Module, Any]:
    """Setup model and tokenizer based on evaluation mode."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    torch_dtype = getattr(torch, config.model.torch_dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    base_model.config.pad_token_id = tokenizer.eos_token_id

    # Apply LoRA for post-training evaluation
    if mode == "post":
        if not config.training.use_lora:
            raise ValueError("LoRA must be enabled in config for post-training evaluation")
        
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

    # Apply quantization if enabled
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
        logging.info("Quantization applied")
    else:
        logging.info("Not using quantization")

    # Optimize memory usage
    base_model = optimize_model_memory(base_model)

    # Wrap with AgenticRAGModel for search-enabled modes
    if mode in ["pre_search", "post"]:
        model = AgenticRAGModel(base_model, tokenizer)
        logging.info("Using AgenticRAGModel with search capabilities")
    else:
        model = base_model
        logging.info("Using base model without search capabilities")

    return model, tokenizer


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(
    model: nn.Module,
    tokenizer: Any,
    eval_dataloader: DataLoader,
    device: torch.device,
    evaluation_before_grpo: bool = False,
) -> List[Dict[str, Union[int, str]]]:
    """Evaluate the model on a dataset and return detailed results."""
    model.eval()
    results: List[Dict[str, Union[int, str]]] = []
    total_batches = len(eval_dataloader)
    
    logging.info(f"Starting evaluation on {total_batches} batches")

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            prompt = batch["prompt"]
            question = batch["question"]
            expected = batch["answer"]
            sample_id = batch["id"]

            # Encode inputs
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Generation settings
            max_new_tokens = 200 if evaluation_before_grpo else 1000
            
            try:
                actual_model = getattr(model, "module", model)
                output_ids = actual_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                )
            except Exception as gen_err:
                raise RuntimeError("Model generation failed") from gen_err

            # Decode response
            seq = output_ids[0].tolist()
            input_len = input_ids.shape[1]
            response_text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)

            result = {
                "id": int(sample_id) if isinstance(sample_id, torch.Tensor) else sample_id,
                "prompt": prompt,
                "question": question,
                "expected": expected,
                "response": response_text,
            }

            # Extract predicted answer
            try:
                predicted = extract_answer_from_model_output(response_text)
                result["predicted"] = predicted
            except Exception:
                logging.error("Failed to extract answer from model output for id %s", result["id"])
                result["predicted"] = ""

            results.append(result)

    model.train()
    return results


def run_evaluation(
    model: nn.Module,
    tokenizer: Any,
    accelerator: Any,
    eval_dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    evaluation_before_grpo: bool = False,
    evaluation_after_grpo: bool = False,
) -> Dict[str, Union[int, float]]:
    """Run evaluation and save results."""
    stats = {}

    def _save_results(key: str, results: List[Dict[str, Any]]) -> None:
        json_path = output_dir / f"{key}.json"
        txt_path = output_dir / f"{key}.txt"
        
        with open(json_path, "w") as jf:
            json.dump(results, jf, indent=2, ensure_ascii=False)
        with open(txt_path, "w") as tf:
            for item in results:
                tf.write(json.dumps(item, ensure_ascii=False, indent=2) + "\n")
            tf.write(f"\nTotal: {len(results)}\n")

    if evaluation_before_grpo:
        logging.info("Running pre-GRPO evaluation")
        pre_results = evaluate_model(model, tokenizer, eval_dataloader, device, evaluation_before_grpo=True)
        accelerator.wait_for_everyone()
        gathered = accelerator.gather_for_metrics(pre_results)
        
        if accelerator.is_main_process:
            gathered = sorted(gathered, key=lambda x: x["id"])
            _save_results("evaluation_before_grpo", gathered)
            filtered = [r for r in gathered if r["predicted"]]
            
            try:
                from src.utils.evaluate import evaluate_with_llm
                c, t, acc, _ = evaluate_with_llm(LLM_EVAL_PROMPT, filtered)
                stats["pre_grpo_correct"] = c
                stats["pre_grpo_total"] = t
                stats["pre_grpo_accuracy"] = acc
                _save_results("evaluation_before_grpo_filtered", filtered)
            except ImportError:
                logging.warning("LLM evaluation not available, skipping accuracy calculation")

    if evaluation_after_grpo:
        logging.info("Running post-GRPO evaluation")
        post_results = evaluate_model(model, tokenizer, eval_dataloader, device, evaluation_before_grpo=False)
        accelerator.wait_for_everyone()
        gathered = accelerator.gather_for_metrics(post_results)
        
        if accelerator.is_main_process:
            gathered = sorted(gathered, key=lambda x: x["id"])
            _save_results("evaluation_after_grpo", gathered)
            filtered = [r for r in gathered if r["predicted"]]
            
            try:
                from src.utils.evaluate import evaluate_with_llm
                c, t, acc, _ = evaluate_with_llm(LLM_EVAL_PROMPT, filtered)
                stats["post_grpo_correct"] = c
                stats["post_grpo_total"] = t
                stats["post_grpo_accuracy"] = acc
                _save_results("evaluation_after_grpo_filtered", filtered)
            except ImportError:
                logging.warning("LLM evaluation not available, skipping accuracy calculation")

    # Save final stats
    if accelerator.is_main_process and stats:
        with open(output_dir / "results.json", "w") as rf:
            json.dump(stats, rf, indent=2, ensure_ascii=False)
        with open(output_dir / "results.txt", "w") as rf:
            rf.write(json.dumps(stats, ensure_ascii=False, indent=2))

    return stats


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified evaluation script for Agentic RAG models")
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        required=True, 
        choices=["pre_no_search", "pre_search", "post"],
        help="Evaluation mode: pre_no_search (base model without search), pre_search (base model with search), post (LoRA fine-tuned model)"
    )
    
    # Common arguments
    parser.add_argument("--num_eval", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--config_path", default="src/config/config.yaml", help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--output_dir", help="Custom output directory (optional)")
    
    # Post-training specific arguments
    parser.add_argument("--date", help="Date of the checkpoint (e.g., 2025-04-16) - required for post mode")
    parser.add_argument("--checkpoint_step", type=int, help="Step number of the checkpoint to load - required for post mode")
    
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if args.mode == "post":
        if not args.date or not args.checkpoint_step:
            raise ValueError("--date and --checkpoint_step are required for post-training evaluation")


# ============================================================================
# Main Function
# ============================================================================

def main() -> None:
    """Main evaluation function."""
    # Parse and validate arguments
    args = parse_args()
    validate_args(args)
    
    # Setup accelerator and config
    accelerator = Accelerator()
    config = load_config(args.config_path)
    config.dataset.num_eval = args.num_eval

    # Setup paths
    if args.mode == "post":
        checkpoint_dir = Path(f"./checkpoints/{config.experiment.name}/{args.date}")
        checkpoint_path = checkpoint_dir / f"step-{args.checkpoint_step:04d}"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        output_dir = Path(args.output_dir) if args.output_dir else Path(f"experiments/post/{config.experiment.name}/{args.date}/step-{args.checkpoint_step:04d}")
    else:
        checkpoint_path = None
        mode_name = args.mode.replace("_", "")
        output_dir = Path(args.output_dir) if args.output_dir else Path(f"output_eval/{mode_name}/{config.experiment.name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging and save config
    setup_logging(output_dir, level=logging.INFO)
    with (output_dir / "config.json").open("w") as fp:
        json.dump(config.__dict__, fp, indent=2)
    
    # Set random seed
    logging.info("Set random seed to %s", config.experiment.random_seed)
    set_random_seed(config.experiment.random_seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # Prepare dataset
    _, eval_dataset = prepare_dataset(
        split="train",
        name=config.dataset.name,
        eval_size=config.dataset.num_eval,
    )
    if len(eval_dataset) == 0:
        raise ValueError("Evaluation dataset is empty.")
    
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    logging.info("Eval dataloader size: %d examples", len(eval_dataloader))

    # Setup model and tokenizer
    logging.info("Loading model for mode: %s", args.mode)
    model, tokenizer = setup_model_and_tokenizer(config, args.mode, checkpoint_path)
    model.to(device)

    # Prepare model and dataloader with accelerator
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    # Run evaluation
    logging.info("Starting evaluation in %s mode...", args.mode)
    
    # Determine evaluation flags based on mode
    evaluation_before_grpo = args.mode.startswith("pre")
    evaluation_after_grpo = args.mode == "post"
    
    run_evaluation(
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        eval_dataloader=eval_dataloader,
        device=device,
        output_dir=output_dir,
        evaluation_before_grpo=evaluation_before_grpo,
        evaluation_after_grpo=evaluation_after_grpo,
    )
    
    logging.info("Evaluation completed. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()