import os

from dotenv import load_dotenv
from rich.traceback import install

load_dotenv()
install()

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn, json, datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from utils.utils import (
    load_config,
    optimize_model_memory,
    set_random_seed,
    setup_logging,
)
from model.model import CustomModel
import logging
from peft import LoraConfig, PeftModel
from torch import nn

app = FastAPI()


class Query(BaseModel):
    text: str


setup_logging()
config = load_config("config/config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = f"./checkpoint/Qwen2.5-7B-4GPU/2025-04-12/step-0150"

logging.info(f"Loading model from {config.model.name}")
torch_dtype = getattr(torch, config.model.torch_dtype)
base_model = AutoModelForCausalLM.from_pretrained(config.model.name, torch_dtype=torch_dtype, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
base_model.config.pad_token_id = base_model.config.eos_token_id = tokenizer.eos_token_id

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
base_model = base_model.to(device)
model = CustomModel(base_model, tokenizer)


@app.post("/chat/")
async def chat(query: Query):
    inputs = tokenizer([query.text], return_tensors="pt").to(device)
    inputs_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    output_ids = model.generate(
        input_ids=inputs_ids,
        attention_mask=attention_mask,
        max_new_tokens=1000,
        max_length_for_gather=10000,
        do_sample=False,
        temperature=0.8,
    )
    output_ids = output_ids[0][len(inputs_ids[0]) :]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
    return {"result": outputs}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=12333)
