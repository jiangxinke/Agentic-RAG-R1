import os
import json
import logging
import pdb
from pathlib import Path
from pprint import pprint
import time
from trl import GRPOTrainer

from data.prompt import LLM_EVAL_PROMPT
from utils.evaluate import evaluate_with_llm
from torch.utils.data import DataLoader

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import datetime
import torch
import random
import pdb

pdb.set_trace = lambda *args, **kwargs: None

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

from data.prepare_dataset import prepare_dataset
from utils.utils import (
    load_config,
    set_random_seed,
    optimize_model_memory,
    setup_logging,
)
from grpo.custom_reward_function import combined_reward
from grpo.generation_interrupt import CustomModel
from grpo.grpo_trainer import train_with_grpo, evaluate_model
from archive.grpo_trainer_mu_GPU import train_with_grpo_mu_GPU
from accelerate import Accelerator


def main(config):
    # Select training function based on GPU configuration
    accelerator = Accelerator()
    accelerator_ref = Accelerator()
    """
    Main function to run the training and evaluation pipeline.

    Steps:
    1. Load config and setup environment
    2. Initialize model and tokenizer
    3. Evaluate initial model (optional)
    4. Fine-tune with GRPO
    5. Evaluate final model
    6. Save results
    """
    # 1. Setup environment

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    checkpoint_dir = Path(f"checkpoint/{config.exp}/{time_str}")
    output_dir = Path(f"output/{config.exp}/{time_str}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    set_random_seed(config.random_seed)
    logger.info(f"Set random seed to {config.random_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Initialize model and tokenizer
    logger.info("Loading model...")
    torch_dtype = getattr(torch, config.model.torch_dtype)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(config.model.name, torch_dtype=torch_dtype).to(device)
    logger.info("Model loaded successfully")

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=config.lora_config.r,
        lora_alpha=config.lora_config.lora_alpha,
        target_modules=config.lora_config.target_modules,
        lora_dropout=config.lora_config.lora_dropout,
        bias=config.lora_config.bias,
        task_type=config.lora_config.task_type,
    )
    model = get_peft_model(model, lora_cfg)
    logger.info(f"Applied LoRA configuration")
    # print("hello")
    # time.sleep(1000000)
    # exit()
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    print(tokenizer.eos_token_id)
    print(tokenizer.eos_token)

    print("</search> 的 ID: ", tokenizer.encode("</search>", add_special_tokens=False))  # [522, 1836, 29]
    print("522 的 token: ", tokenizer.decode([522]))  # "</"
    print("1836 的 token: ", tokenizer.decode([1836]))  # "search"
    print("29 的 token: ", "==", tokenizer.decode([29]), "==")  # ">"
    print("397 的 token: ", "==", tokenizer.decode([397]), "==")  # ">"

    # if tokenizer.pad_token_id is None:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # model.config.pad_token_id = tokenizer.pad_token_id
    # model.config.eos_token_id = tokenizer.eos_token_id

    # tokenizer.add_special_tokens
    #     special_token_dict = {
    #     "additional_special_tokens": [
    #         "<observation>",
    #         "</observation>",
    #         "<reasoning>",
    #         "</reasoning>",
    #         "<search>",
    #         "</search>",
    #         "<answer>",
    #         "</answer>"
    #     ]
    # }
    # tokenizer.add_special_tokens(special_token_dict)
    # tokenizer.add_tokens(special_token_dict["additional_special_tokens"])
    # model.resize_token_embeddings(len(tokenizer))
    # logger.info(f"Added special tokens: {tokenizer.special_tokens_map}")
    # token_id = tokenizer.convert_tokens_to_ids("<reasoning>")
    # token_id = tokenizer.convert_tokens_to_ids("<answer>")
    # logger.info(f"<answer> 的 ID: {token_id}")
    # logger.info(f"<reasoning> 的 ID: {token_id}")

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(
        "train", config.dataset.name, eval_size=config.dataset.num_eval_examples
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    logger.info(f"Train dataloader: {len(train_dataloader)}, Eval dataloader: {len(eval_dataloader)}")

    # 3. Initial evaluation
    results = {}
    if config.evaluation_before_grpo:
        logger.info("Evaluating initial model...")
        evaluation_results = evaluate_model(model, tokenizer, eval_dataloader, device)
        evaluation_before_grpo = output_dir / "evaluation_before_grpo.json"
        with open(evaluation_before_grpo, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        # 过滤条目 predicted 是 None 的
        evaluation_results = [
            item for item in evaluation_results if item["predicted"] is not None and item["predicted"] != ""
        ]

        evaluation_before_grpo_filtered = output_dir / "evaluation_before_grpo_filtered.json"
        with open(evaluation_before_grpo_filtered, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        correct, total, pre_grpo_accuracy = evaluate_with_llm(LLM_EVAL_PROMPT, evaluation_results)

        logger.info(f"Initial accuracy: {correct}/{total} = {pre_grpo_accuracy:.2f}%")
        results["pre_grpo_correct"] = correct
        results["pre_grpo_total"] = total
        results["pre_grpo_accuracy"] = pre_grpo_accuracy

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # 4. GRPO fine-tuning
    logger.info("Starting GRPO fine-tuning...")
    training_config = {
        "num_iterations": config.training.num_iterations,
        "steps_per_iteration": config.training.steps_per_iteration,
        "batch_size": config.training.batch_size,
        "num_generations": config.training.num_generations,
        "max_completion_length": config.training.max_completion_length,
        "beta": config.training.beta,
        "learning_rate": config.training.learning_rate,
        "mu": config.training.mu,
        "epsilon": config.training.epsilon,
        "reward_function": combined_reward,
    }

    # Optimize model memory usage
    model = optimize_model_memory(model)
    custom_model = CustomModel(model, tokenizer)

    # Setup multi-GPU if available
    device_ids = None

    # train_func = train_with_grpo_mu_GPU if config.training.mu_gpu else train_with_grpo
    train_func = train_with_grpo
    # Run training
    start_time = time.time()
    model = train_func(
        model=custom_model,
        tokenizer=tokenizer,
        device_ids=device_ids,
        accelerator=accelerator,
        dataloader=train_dataloader,
        **training_config,
    )
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"训练完成，总耗时: {training_time:.2f}秒 ({training_time/60:.2f}分钟)")

    # 5. Final evaluation
    if config.evaluation_after_grpo:
        logger.info("Evaluating fine-tuned model...")
        evaluation_results = evaluate_model(model, tokenizer, eval_data, device)
        evaluation_after_grpo = output_dir / "evaluation_after_grpo.json"
        with open(evaluation_after_grpo, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        # 同时保存为txt格式，方便查看
        with open(output_dir / "evaluation_after_grpo.txt", "w") as f:
            for item in evaluation_results:
                pprint(item, indent=4, width=120, stream=f)
            f.write(f"\n共有 {len(evaluation_results)} 条数据")
        print(f"数据已输出到 evaluation_after_grpo.txt 文件，共有 {len(evaluation_results)} 条数据")

        # 过滤条目 predicted 是 None 的
        evaluation_results = [
            item for item in evaluation_results if item["predicted"] is not None and item["predicted"] != ""
        ]

        evaluation_after_grpo_filtered = output_dir / "evaluation_after_grpo_filtered.json"
        with open(evaluation_after_grpo_filtered, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        # 同时保存过滤后的结果为txt格式
        with open(output_dir / "evaluation_after_grpo_filtered.txt", "w") as f:
            for item in evaluation_results:
                pprint(item, indent=4, width=120, stream=f)
            f.write(f"\n共有 {len(evaluation_results)} 条数据")
        print(f"数据已输出到 evaluation_after_grpo_filtered.txt 文件，共有 {len(evaluation_results)} 条数据")

        correct, total, post_grpo_accuracy = evaluate_with_llm(LLM_EVAL_PROMPT, evaluation_results)

        logger.info(f"Final accuracy: {correct}/{total} = {post_grpo_accuracy:.2f}%")
        results["post_grpo_correct"] = correct
        results["post_grpo_total"] = total
        results["post_grpo_accuracy"] = post_grpo_accuracy

        if config.evaluation_before_grpo:
            improvement = post_grpo_accuracy - results["pre_grpo_accuracy"]
            logger.info(f"Improvement: {improvement:.2f}%")
            results["improvement"] = improvement

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # 6. Save model
    # logger.info(f"Saving model to {checkpoint_dir}")
    # model.save_pretrained(checkpoint_dir)
    # tokenizer.save_pretrained(checkpoint_dir)


if __name__ == "__main__":
    config = load_config("config/config.yaml")
    if config.load_from_checkpoint:
        # 设置环境
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, config.model.torch_dtype)

        # 创建输出目录
        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        output_dir = Path(f"output/{config.exp}/{time_str}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        logger = setup_logging(output_dir)
        logger.info(f"从检查点加载模型: {config.checkpoint_path}")

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=torch_dtype,
            # device_map=config.model.device_map,
        ).to(device)

        lora_cfg = LoraConfig(
            r=config.lora_config.r,
            lora_alpha=config.lora_config.lora_alpha,
            target_modules=config.lora_config.target_modules,
            lora_dropout=config.lora_config.lora_dropout,
            bias=config.lora_config.bias,
            task_type=config.lora_config.task_type,
        )
        # 从检查点加载 LoRA 权重
        model = PeftModel.from_pretrained(base_model, config.checkpoint_path, config=lora_cfg, repo_type="model")
        logger.info("成功加载检查点中的模型权重")

        exit()

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
        logger.info("成功加载检查点中的分词器")

        # 使用 CustomModel 包装模型和分词器
        model = CustomModel(model, tokenizer)
        logger.info("成功包装为 CustomModel")

        # 准备数据集
        all_data = prepare_dataset("train", config.dataset.name)
        random.shuffle(all_data)
        eval_size = config.dataset.num_eval_examples
        eval_data = all_data[:eval_size]
        logger.info(f"评估数据集大小: {len(eval_data)}")

        # 评估模型
        logger.info("开始评估模型...")
        evaluation_results = evaluate_model(model, tokenizer, eval_data, device)

        # 保存评估结果
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(evaluation_results, f, indent=2)

        # 过滤无效预测
        filtered_results = [
            item for item in evaluation_results if item["predicted"] is not None and item["predicted"] != ""
        ]

        with open(output_dir / "evaluation_results_filtered.json", "w") as f:
            json.dump(filtered_results, f, indent=2)

        # 使用LLM评估结果
        correct, total, accuracy = evaluate_with_llm(LLM_EVAL_PROMPT, filtered_results)

        # 记录结果
        results = {
            "load_from_checkpoint_correct": correct,
            "load_from_checkpoint_total": total,
            "load_from_checkpoint_accuracy": accuracy,
        }

        logger.info(f"从检查点加载模型评估准确率: {correct}/{total} = {accuracy:.2f}%")

        with open(output_dir / "evaluation_summary.json", "w") as f:
            json.dump(results, f, indent=2)
    else:
        main(config)
