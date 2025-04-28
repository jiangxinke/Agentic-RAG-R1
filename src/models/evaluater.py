import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.prompt import LLM_EVAL_PROMPT
from src.utils.extractor import extract_answer_from_model_output


def evaluate_model(
    model: nn.Module,
    tokenizer: Any,
    eval_dataloader: DataLoader,
    device: torch.device,
    use_interrupt: bool = False,
    evaluation_before_grpo: bool = False,
    evaluation_after_grpo: bool = False,
) -> List[Dict[str, Union[int, str]]]:
    """
    Evaluate the model on a dataset and return detailed results.

    This function runs the model in evaluation mode over the provided DataLoader,
    generates outputs for each batch, extracts predicted answers, and collects
    the results along with prompts, questions, and expected answers.

    Args:
        model (nn.Module): The language model to evaluate.
        tokenizer (Any): Tokenizer for encoding inputs and decoding outputs.
        eval_dataloader (DataLoader): DataLoader yielding batches with keys
            'prompt', 'question', 'answer', and 'id'.
        device (torch.device): Device on which to perform computation.
        use_interrupt (bool): Whether to use interruption-based generation.
        evaluation_before_grpo (bool): Flag for pre-GRPO evaluation settings.
        evaluation_after_grpo (bool): Flag for post-GRPO evaluation settings.

    Returns:
        List[Dict[str, Union[int, str]]]: A list of result dicts containing 'id',
            'prompt', 'question', 'expected', 'response', and 'predicted'.

    Raises:
        RuntimeError: If generation fails or unexpected output format is encountered.
    """
    model.eval()
    results: List[Dict[str, Union[int, str]]] = []

    total_batches = len(eval_dataloader)
    logging.info(f"Starting evaluation on {total_batches} batches")

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            prompt: str = batch["prompt"]
            question: str = batch["question"]
            expected: str = batch["answer"]
            sample_id: Union[int, torch.Tensor] = batch["id"]

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
            full_text = tokenizer.decode(seq, skip_special_tokens=True)

            result: Dict[str, Union[int, str]] = {
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


def evaluate(
    model: nn.Module,
    tokenizer: Any,
    accelerator: Any,
    eval_dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    evaluation_before_grpo: bool = False,
    evaluation_after_grpo: bool = False,
    llm_eval_prompt: Optional[str] = None,
) -> Dict[str, Union[int, float]]:
    """
    Conduct full evaluation before and/or after GRPO training and save results.

    This function orchestrates pre- and post-training evaluations, gathers metrics
    across distributed processes via the accelerator, computes accuracies using an LLM,
    and writes JSON and TXT result files to the output directory.

    Args:
        model (nn.Module): The language model to evaluate.
        tokenizer (Any): Tokenizer for encoding and decoding.
        accelerator (Any): Accelerator for distributed synchronization.
        eval_dataloader (DataLoader): DataLoader for evaluation examples.
        device (torch.device): Computation device.
        output_dir (Path): Directory for saving result files.
        evaluation_before_grpo (bool): Whether to run evaluation before training.
        evaluation_after_grpo (bool): Whether to run evaluation after training.
        llm_eval_prompt (Optional[str]): Template prompt for LLM-based accuracy.

    Returns:
        Dict[str, Union[int, float]]: Dictionary with keys 'pre_grpo_accuracy',
            'post_grpo_accuracy', etc., depending on flags.

    Raises:
        IOError: If writing result files fails.
    """
    stats: Dict[str, Union[int, float]] = {}
    prompt = llm_eval_prompt or LLM_EVAL_PROMPT

    def _save_results(key: str, results: List[Dict[str, Any]]) -> None:
        json_path = output_dir / f"{key}.json"
        txt_path = output_dir / f"{key}.txt"
        try:
            with open(json_path, "w") as jf:
                json.dump(results, jf, indent=2, ensure_ascii=False)
            with open(txt_path, "w") as tf:
                for item in results:
                    tf.write(json.dumps(item, ensure_ascii=False, indent=2) + "\n")
                tf.write(f"\nTotal: {len(results)}\n")
        except Exception as io_err:
            raise IOError(f"Failed to save {key} results") from io_err

    if evaluation_before_grpo:
        logging.info("Running pre-GRPO evaluation")
        pre_results = evaluate_model(model, tokenizer, eval_dataloader, device, evaluation_before_grpo=True)
        accelerator.wait_for_everyone()
        gathered = accelerator.gather_for_metrics(pre_results)
        if accelerator.is_main_process:
            gathered = sorted(gathered, key=lambda x: x["id"])
            _save_results("evaluation_before_grpo", gathered)
            filtered = [r for r in gathered if r["predicted"]]
            from utils.evaluate import evaluate_with_llm

            c, t, acc, _ = evaluate_with_llm(prompt, filtered)
            stats["pre_grpo_correct"] = c
            stats["pre_grpo_total"] = t
            stats["pre_grpo_accuracy"] = acc
            _save_results("evaluation_before_grpo_filtered", filtered)
            with open(output_dir / "results.json", "w") as rf:
                json.dump(stats, rf, indent=2, ensure_ascii=False)
            with open(output_dir / "results.txt", "w") as rf:
                rf.write(json.dumps(stats, ensure_ascii=False, indent=2))

    if evaluation_after_grpo:
        logging.info("Running post-GRPO evaluation")
        post_results = evaluate_model(model, tokenizer, eval_dataloader, device, evaluation_after_grpo=True)
        accelerator.wait_for_everyone()
        gathered = accelerator.gather_for_metrics(post_results)
        if accelerator.is_main_process:
            gathered = sorted(gathered, key=lambda x: x["id"])
            _save_results("evaluation_after_grpo", gathered)
            filtered = [r for r in gathered if r["predicted"]]
            from src.utils.evaluate import evaluate_with_llm

            c, t, acc, _ = evaluate_with_llm(prompt, filtered)
            stats["post_grpo_correct"] = c
            stats["post_grpo_total"] = t
            stats["post_grpo_accuracy"] = acc
            _save_results("evaluation_after_grpo_filtered", filtered)
            with open(output_dir / "results.json", "w") as rf:
                json.dump(stats, rf, indent=2, ensure_ascii=False)
            with open(output_dir / "results.txt", "w") as rf:
                rf.write(json.dumps(stats, ensure_ascii=False, indent=2))

    return stats
