import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def inference_model(
    model: nn.Module,
    tokenizer: Any,
    eval_dataloader: DataLoader,
    device: torch.device,
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
            try:
                actual_model = getattr(model, "module", model)
                neuron_importance_dict = actual_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=200,
                    # do_sample=False,
                    # temperature=0.0,
                    do_sample=True,
                    temperature=0.7,
                    locate_params=True,
                )
            except Exception as gen_err:
                raise RuntimeError("Model generation failed") from gen_err

            print(neuron_importance_dict)
            exit(0)
            # results.append(result)

    return results
