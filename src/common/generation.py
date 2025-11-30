from typing import Any, List, Tuple

import torch
from transformers import AutoTokenizer
from src.common.config import GenerationConfig

from src.common.mask import *


def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))
    return selected.squeeze(-1)


def compute_log_probabilities(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,
        obtain_logits=True,
    )
    logits = outputs[:, :-1, :]
    ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, ids)


def create_completion_mask(
    completion_ids: torch.Tensor,
    eos_token_id: int,
    observation_start_ids: List[int],
    observation_end_ids: List[int],
) -> torch.Tensor:
    batch_size, seq_len = completion_ids.shape
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((batch_size,), seq_len, dtype=torch.long, device=completion_ids.device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    seq_indices = torch.arange(seq_len, device=completion_ids.device).unsqueeze(0).expand(batch_size, -1)
    completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()
    if not observation_start_ids or not observation_end_ids:
        raise ValueError("Observation start/end ID lists must be non-empty.")
    obs_start_len = len(observation_start_ids)
    obs_end_len = len(observation_end_ids)
    if obs_start_len > seq_len or obs_end_len > seq_len:
        raise ValueError("Observation marker length exceeds sequence length.")
    is_obs_start = torch.zeros_like(completion_ids, dtype=torch.bool)
    is_obs_end = torch.zeros_like(completion_ids, dtype=torch.bool)
    start_tensor = torch.tensor(observation_start_ids, device=completion_ids.device)
    end_tensor = torch.tensor(observation_end_ids, device=completion_ids.device)
    for b in range(batch_size):
        for i in range(seq_len - obs_start_len + 1):
            if torch.all(completion_ids[b, i : i + obs_start_len] == start_tensor):
                is_obs_start[b, i] = True
        for i in range(seq_len - obs_end_len + 1):
            if torch.all(completion_ids[b, i : i + obs_end_len] == end_tensor):
                is_obs_end[b, i] = True
    observation_flag = torch.zeros_like(completion_mask, dtype=torch.int)
    for b in range(batch_size):
        in_obs = False
        for i in range(seq_len):
            if is_obs_start[b, i]:
                in_obs = True
            if in_obs:
                observation_flag[b, i] = 1
            if is_obs_end[b, i]:
                in_obs = False
    final_mask = completion_mask & (1 - observation_flag)
    return final_mask


def generate_completions(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    gen_cfg: GenerationConfig,
    use_SSRL: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    base_num_generations = gen_cfg.num_generations
    if isinstance(base_num_generations, dict):
        mode = base_num_generations.get("mode", "constant")
        if mode == "constant":
            num_generations = int(base_num_generations.get("constant", 1))
        else:
            num_generations = int(base_num_generations.get("constant", 1))
    else:
        num_generations = int(base_num_generations)
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    completion_ids = model(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=gen_cfg.max_new_tokens,
        max_length_for_gather=gen_cfg.max_length_for_gather,
        do_sample=gen_cfg.do_sample,
        temperature=gen_cfg.temperature,
        max_generate_iterations=gen_cfg.max_generate_iterations,
        use_diverse_sampling=gen_cfg.use_diverse_sampling,
        diversity_penalty=gen_cfg.diversity_penalty,
        use_SSRL=use_SSRL,
    )
    start_ids = tokenizer("<observation>").input_ids
    end_ids = tokenizer("</observation>").input_ids
    completion_mask = create_completion_mask(
        completion_ids,
        tokenizer.eos_token_id,
        start_ids,
        end_ids,
    )
    return prompt_ids, prompt_mask, completion_ids, completion_mask
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM
from src.core.model import AgenticRAGModel
from src.utils.utils import optimize_model_memory
def build_agentic_rag_model(
    config,
    device: torch.device,
) -> AgenticRAGModel:
    continue_training = config.training.continue_training
    checkpoint_step = config.training.current_step
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        trust_remote_code=True,
    ).to(device)
    if config.training.use_lora:
        lora_cfg = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )
        if continue_training:
            weights_path = f"checkpoints/{config.experiment.name}/step-{checkpoint_step:04d}"
            base = PeftModel.from_pretrained(base, weights_path, config=lora_cfg, is_trainable=True)
        else:
            base = get_peft_model(base, lora_cfg)
    if config.training.use_quant:
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=config.qlora.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
            load_in_8bit=config.qlora.load_in_8bit,
            llm_int8_threshold=config.qlora.llm_int8_threshold,
        )
        base = load_and_quantize_model(base, bnb_quantization_config=bnb_quantization_config, device_map="auto")
    base = optimize_model_memory(base)
    return AgenticRAGModel(base, tokenizer)


def _unwrap_peft(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "model"):
        model = model.model
    if not isinstance(model, PeftModel):
        raise ValueError("Underlying model is not a PeftModel")
    return model


def save_lora_only_in_zero2(engine, tokenizer, ckpt_dir, accelerator):
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    peft_model = _unwrap_peft(engine)
    lora_params = [p for n, p in peft_model.named_parameters() if "lora" in n]
    enabled = hasattr(engine, "zero_optimization_stage") and engine.zero_optimization_stage() == 2
    if accelerator.is_main_process:
        import logging
        logging.info(f"Found {len(lora_params)} LoRA parameters")
    from deepspeed import zero
    from peft.utils import get_peft_model_state_dict
    with zero.GatheredParameters(lora_params, enabled=enabled):
        lora_state = get_peft_model_state_dict(peft_model)
        if accelerator.is_main_process:
            peft_model.save_pretrained(ckpt_dir)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(ckpt_dir)
