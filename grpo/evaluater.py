import json
import logging

import torch
from torch import nn
from tqdm import tqdm

from data.prompt import LLM_EVAL_PROMPT
from utils.answer_extractor import extract_answer_from_model_output


def evaluate_model(
    model, tokenizer, eval_dataloader, device, use_interrupt=False, evaluation_before_grpo=False, evaluation_after_grpo=False
):
    """
    Evaluates the model on a set of examples and prints detailed results.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for encoding inputs and decoding outputs.
        eval_examples (list): List of evaluation examples, each containing "prompt" and "answer".
        device: The device (CPU or GPU) to run evaluation on.

    Returns:
        list: evaluation_results containing detailed results for each example for further evaluation

    Explanation:
        1. Sets the model to evaluation mode.
        2. For each example in the evaluation set:
           - Encodes the prompt and generates a response using the model.
           - Extracts the predicted answer from the generated response.
           - Prints detailed information about each example.
        3. Returns the detailed results.
        4. Returns the model to training mode.
    """
    model.eval()
    total = len(eval_dataloader)
    print("\n" + "=" * 50)
    print("EVALUATION ON", total, "EXAMPLES")
    print("=" * 50)

    # Create a list to store evaluation results
    evaluation_results = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # Build the full prompt using the same method as training.
        full_prompt = batch["prompt"]
        expected = batch["answer"]
        question = batch["question"]
        id = batch["id"]
        # Tokenize the full prompt and generate a response from the model.
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, padding_side="left").to(device)
        prompt_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # prompt_ids = prompt_ids.repeat_interleave(1, dim=0)
        # attention_mask = attention_mask.repeat_interleave(1, dim=0)
        if evaluation_before_grpo:
            max_new_tokens = 200
        else:
            max_new_tokens = 1000
        with torch.no_grad():
            actual_model = model.module if hasattr(model, "module") else model
            output = actual_model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # import pdb; pdb.set_trace()

        if evaluation_before_grpo:
            output = output[0].tolist()  # 获取生成的token序列
        if evaluation_after_grpo:
            output = output[0].tolist()

        # 获取输入的长度，以便只保留模型生成的部分
        input_length = prompt_ids.shape[1]
        if evaluation_before_grpo:
            response_only = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        elif evaluation_after_grpo:
            response_only = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        else:
            response_only = tokenizer.decode(output[0, input_length:], skip_special_tokens=True)

        # 完整响应（仅用于调试）
        full_response = tokenizer.decode(output, skip_special_tokens=True)

        # Create result entry for this example
        result_entry = {
            "id": id.item() if isinstance(id, torch.Tensor) else id,  # Convert tensor to Python value
            "prompt": full_prompt,
            "question": question,
            "expected": expected,
            "response": response_only,
        }

        try:
            predicted = extract_answer_from_model_output(response_only)
            result_entry["predicted"] = predicted

            # Print details of the evaluation.
            if False:
                print("\n==> Prompt:")
                print(full_prompt)
                print("\n==> Expected Answer:")
                print(expected)
                print("\n==> Extracted Answer:")
                print(predicted)
                print("\n==> Full Generated Response:")
                print(response)
                print("-" * 50)

        except Exception as e:
            logging.error(f"Error: Failed to parse model output for prompt")
            result_entry["predicted"] = "Error: Failed to parse model output"

        evaluation_results.append(result_entry)

    return evaluation_results


def evaluate(
    model,
    tokenizer,
    accelerator,
    eval_dataloader,
    device,
    output_dir,
    evaluation_before_grpo,
    evaluation_after_grpo,
    LLM_EVAL_PROMPT=LLM_EVAL_PROMPT,
):
    results = {}

    if evaluation_before_grpo:
        logging.info("Evaluating initial model before training...")
        evaluation_results = evaluate_model(
            model,
            tokenizer,
            eval_dataloader,
            device,
            use_interrupt=False,
            evaluation_before_grpo=evaluation_before_grpo,
            evaluation_after_grpo=evaluation_after_grpo,
        )

        # 确保所有进程在继续之前等待训练完成
        accelerator.wait_for_everyone()

        # 收集所有进程的结果
        all_evaluation_results = accelerator.gather_for_metrics([item for item in evaluation_results])

        if accelerator.is_main_process:
            # 按照id排序结果
            all_evaluation_results = sorted(all_evaluation_results, key=lambda x: x["id"])

            evaluation_before_grpo_path = output_dir / "evaluation_before_grpo.json"
            with open(evaluation_before_grpo_path, "w") as f:
                json.dump(all_evaluation_results, f, indent=2)

            # 同时保存为txt格式
            evaluation_before_grpo_txt_path = output_dir / "evaluation_before_grpo.txt"
            with open(evaluation_before_grpo_txt_path, "w") as f:
                for item in all_evaluation_results:
                    print(json.dumps(item, indent=4, ensure_ascii=False), file=f)
                f.write(f"\n共有 {len(all_evaluation_results)} 条数据")

            # 过滤条目 predicted 是 None 的
            evaluation_results_filtered = [
                item for item in all_evaluation_results if item["predicted"] is not None and item["predicted"] != ""
            ]

            from utils.evaluate import evaluate_with_llm

            correct, total, pre_grpo_accuracy, evaluation_results_filtered = evaluate_with_llm(
                LLM_EVAL_PROMPT, evaluation_results_filtered
            )
            evaluation_before_grpo_filtered_path = output_dir / "evaluation_before_grpo_filtered.json"
            with open(evaluation_before_grpo_filtered_path, "w") as f:
                json.dump(evaluation_results_filtered, f, indent=2)

            # 同时保存为txt格式
            evaluation_before_grpo_filtered_txt_path = output_dir / "evaluation_before_grpo_filtered.txt"
            with open(evaluation_before_grpo_filtered_txt_path, "w") as f:
                for item in evaluation_results_filtered:
                    print(json.dumps(item, indent=4, ensure_ascii=False), file=f)
                f.write(f"\n共有 {len(evaluation_results_filtered)} 条数据")

            logging.info(f"Initial accuracy: {correct}/{total} = {pre_grpo_accuracy:.2f}%")
            results["pre_grpo_correct"] = correct
            results["pre_grpo_total"] = total
            results["pre_grpo_accuracy"] = pre_grpo_accuracy

            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

            # 同时保存为txt格式
            with open(output_dir / "results.txt", "w") as f:
                print(json.dumps(results, indent=4, ensure_ascii=False), file=f)
                f.write(f"\n共有 {len(results)} 条数据")

    if evaluation_after_grpo:
        logging.info("Evaluating model after training...")
        evaluation_results = evaluate_model(
            model,
            tokenizer,
            eval_dataloader,
            device,
            use_interrupt=False,
            evaluation_before_grpo=evaluation_before_grpo,
            evaluation_after_grpo=evaluation_after_grpo,
        )

        # 确保所有进程在继续之前等待
        accelerator.wait_for_everyone()

        # 收集所有进程的结果
        all_evaluation_results = accelerator.gather_for_metrics([item for item in evaluation_results])

        if accelerator.is_main_process:
            # 按照id排序结果，确保id是Python原生类型
            all_evaluation_results = sorted(all_evaluation_results, key=lambda x: x["id"])

            evaluation_after_grpo_path = output_dir / "evaluation_after_grpo.json"
            with open(evaluation_after_grpo_path, "w") as f:
                json.dump(all_evaluation_results, f, indent=2)

            # 同时保存为txt格式
            evaluation_after_grpo_txt_path = output_dir / "evaluation_after_grpo.txt"
            with open(evaluation_after_grpo_txt_path, "w") as f:
                for item in all_evaluation_results:
                    print(json.dumps(item, indent=4, ensure_ascii=False), file=f)
                f.write(f"\n共有 {len(all_evaluation_results)} 条数据")

            # 过滤条目 predicted 是 None 的
            evaluation_results_filtered = [
                item for item in all_evaluation_results if item["predicted"] is not None and item["predicted"] != ""
            ]

            from utils.evaluate import evaluate_with_llm

            correct, total, post_grpo_accuracy, evaluation_results_filtered = evaluate_with_llm(
                LLM_EVAL_PROMPT, evaluation_results_filtered
            )
            evaluation_after_grpo_filtered_path = output_dir / "evaluation_after_grpo_filtered.json"
            with open(evaluation_after_grpo_filtered_path, "w") as f:
                json.dump(evaluation_results_filtered, f, indent=2)

            # 同时保存为txt格式
            evaluation_after_grpo_filtered_txt_path = output_dir / "evaluation_after_grpo_filtered.txt"
            with open(evaluation_after_grpo_filtered_txt_path, "w") as f:
                for item in evaluation_results_filtered:
                    print(json.dumps(item, indent=4, ensure_ascii=False), file=f)
                f.write(f"\n共有 {len(evaluation_results_filtered)} 条数据")

            logging.info(f"Final accuracy: {correct}/{total} = {post_grpo_accuracy:.2f}%")
            results["post_grpo_correct"] = correct
            results["post_grpo_total"] = total
            results["post_grpo_accuracy"] = post_grpo_accuracy

            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

            # 同时保存为txt格式
            with open(output_dir / "results.txt", "w") as f:
                print(json.dumps(results, indent=4, ensure_ascii=False), file=f)
                f.write(f"\n共有 {len(results)} 条数据")
