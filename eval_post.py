from rich.traceback import install

install()

import os
import json
import logging
import argparse
from pathlib import Path
from time import sleep
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from torch import nn

from data.prepare_dataset import prepare_dataset
from utils.utils import load_config, set_random_seed, optimize_model_memory, setup_logging
from grpo.model import CustomModel
from grpo.evaluater import evaluate
from data.prompt import LLM_EVAL_PROMPT
from accelerate import Accelerator


def main():
    # 添加命令行参数解析
    accelerator = Accelerator()
    config = load_config("config/config.yaml")
    config.dataset.num_eval = 100

    # 从命令行获取checkpoint_step
    checkpoint_step = config.eval_checkpoint_step
    checkpoint_path = f"./checkpoint/{config.exp}/2025-04-01/step-{checkpoint_step:04d}"

    # 使用默认的输出目录
    output_dir = Path(f"output_eval/post/{config.exp}/step-{checkpoint_step:04d}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 日志初始化
    setup_logging(output_dir, level=logging.INFO)

    # 保存一下config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # 设置随机种子
    set_random_seed(config.random_seed)
    logging.info(f"Set random seed to {config.random_seed}")

    # 根据环境变量和PyTorch自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 准备数据集
    _, eval_dataset = prepare_dataset("train", config.dataset.name, eval_size=config.dataset.num_eval)

    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    logging.info(f"Eval dataloader size: {len(eval_dataloader)}")

    # 初始化模型 & tokenizer
    logging.info("Loading model...")
    torch_dtype = getattr(torch, config.model.torch_dtype)
    base_model = AutoModelForCausalLM.from_pretrained(config.model.name, torch_dtype=torch_dtype, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = base_model.config.eos_token_id = tokenizer.eos_token_id

    # 如果要使用LoRA
    if config.training.use_lora:
        logging.info("Applying LoRA configuration...")
        lora_cfg = LoraConfig(
            r=config.lora_config.r,
            lora_alpha=config.lora_config.lora_alpha,
            target_modules=config.lora_config.target_modules,
            lora_dropout=config.lora_config.lora_dropout,
            bias=config.lora_config.bias,
            task_type=config.lora_config.task_type,
        )
        logging.info(f"Loading LoRA weights from {checkpoint_path}")
        base_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        logging.info("LoRA weights loaded successfully")
    else:
        logging.warning("Not using LoRA but checkpoint path was provided")

    # 把模型放到设备上
    base_model = base_model.to(device)

    # 优化内存
    base_model = optimize_model_memory(base_model)

    # 构建你的CustomModel
    model = CustomModel(base_model, tokenizer)

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    # # 获取并输出具体的显卡号和accelerator信息
    # if torch.cuda.is_available():
    #     gpu_id = torch.cuda.current_device()
    #     gpu_name = torch.cuda.get_device_name(gpu_id)
    #     logging.info(
    #         f"Model and eval_dataloader prepared with accelerator, device: {device} (GPU {gpu_id}: {gpu_name}), accelerator device: {accelerator.device}, len(eval_dataloader): {len(eval_dataloader)}"
    #     )
    #     # 打印所有样本的ID
    #     for batch in eval_dataloader:
    #         logging.info(f"{accelerator.device} Batch ID: {batch['id']}")
    #         for i in range(len(batch["id"])):
    #             logging.info(f"{accelerator.device} Sample ID: {batch['id'][i]}")

    # else:
    #     logging.info(
    #         f"Model and eval_dataloader prepared with accelerator, device: {device}, accelerator device: {accelerator.device}, len(eval_dataloader): {len(eval_dataloader)}"
    #     )

    # 开始评估
    logging.info("Starting evaluation...")
    evaluate(
        model,
        tokenizer,
        accelerator,
        eval_dataloader,
        device,
        output_dir,
        evaluation_before_grpo=False,
        evaluation_after_grpo=True,
        LLM_EVAL_PROMPT=LLM_EVAL_PROMPT,
    )

    logging.info(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
