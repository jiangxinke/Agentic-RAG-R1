
import logging
from typing import Any, Dict, List, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rich import print

from src.data.prompt import LLM_EVAL_PROMPT
from src.utils.evaluate import eval_item
from src.utils.extractor import extract_answer_from_model_output

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

    total_batches = len(eval_dataloader)
    logging.info(f"Starting evaluation on {total_batches} batches")

    cnt = 0
    epoch_neuron_importance_dict = {}
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=80):
            prompt: str = batch["prompt"]
            question: str = batch["question"]
            expected: str = batch["answer"]
            sample_id: Union[int, torch.Tensor] = batch["id"]
            sample_id = int(sample_id) if isinstance(sample_id, torch.Tensor) else sample_id
            
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
                response_text, neuron_importance_dict = actual_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1000,
                    do_sample=True,
                    temperature=0.7,
                    calculate_param_importance=True,
                    use_KV_Cache=False,
                )
            except Exception as gen_err:
                logging.error(f"Generation failed for prompt: {prompt}\nError: {gen_err}")
                continue
                # raise RuntimeError("Model generation failed") from gen_err
            
            if not response_text:
                logging.error("Failed to get valid response from model output for id %s", sample_id)
                continue

            try:
                predicted = extract_answer_from_model_output(response_text)
            except Exception:
                logging.error("Failed to extract answer from model output for id %s", sample_id)
                continue

            try:
                eval_result = eval_item(LLM_EVAL_PROMPT, question, expected, predicted)
                if not eval_result:
                    continue
                # logging.info(f"Evaluation result for id {sample_id}: Correct={c}, Total={t}, Accuracy={acc:.4f}")
            except Exception as eval_err:
                logging.error(f"Evaluation failed for id {sample_id}: {eval_err}")
                continue

            cnt += 1
            if cnt % 10 == 0:
                print(neuron_importance_dict)
                print(f"Response: {response_text}")
                # break

            try:
                for key, value in neuron_importance_dict.items():
                    if key not in epoch_neuron_importance_dict:
                        epoch_neuron_importance_dict[key] = value
                    else:
                        for layer_idx, activation in value.items():
                            if layer_idx not in epoch_neuron_importance_dict[key]:
                                epoch_neuron_importance_dict[key][layer_idx] = activation
                            else:
                                # check the shape matches
                                assert epoch_neuron_importance_dict[key][layer_idx].shape == activation.shape, \
                                    f"Shape mismatch for {key} layer {layer_idx}: {epoch_neuron_importance_dict[key][layer_idx].shape} vs {activation.shape}"
                                epoch_neuron_importance_dict[key][layer_idx] += activation
            except Exception as e:
                logging.error(f"Failed to accumulate neuron importance for id {sample_id}: {e}")
                continue

    return epoch_neuron_importance_dict
