from typing import Any, Callable, Dict, List, Optional, Tuple

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def generate_grad_control_dicts(model, param_importance_path=r"/home/xiaobei/qrh/Agentic-RAG-R1/output_eval/neuron/neuron_active_ratio.json", top_k=8):
    try:
        with open(param_importance_path, 'r', encoding='utf-8') as f:
            param_importance = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"参数重要性文件未找到：{param_importance_path}")
    except json.JSONDecodeError:
        raise ValueError(f"文件格式错误，无法解析JSON：{param_importance_path}")
    except Exception as e:
        raise Exception(f"读取参数重要性文件时出错：{str(e)}")
    all_layers = set()
    for key in param_importance:
        if isinstance(param_importance[key], dict):
            for layer_str in param_importance[key].keys():
                try:
                    layer = int(layer_str)
                    all_layers.add(layer)
                except ValueError:
                    continue
    if not all_layers:
        raise ValueError("无法从参数重要性字典中提取有效的层号信息")
    max_layer = max(all_layers)
    num_layers = max_layer + 1
    tag_keys = ["reasoning_tag", "search_tag", "summary_tag", "answer_tag", "backtrack_tag"]
    args_keys = ["reasoning_arg", "search_arg", "summary_arg", "answer_arg", "backtrack_arg"]
    tag_total_scores = {layer: 0.0 for layer in range(num_layers)}
    args_total_scores = {layer: 0.0 for layer in range(num_layers)}
    for key in tag_keys:
        if key in param_importance and isinstance(param_importance[key], dict):
            layer_scores = param_importance[key]
            for layer_str, score in layer_scores.items():
                try:
                    layer = int(layer_str)
                    if layer in tag_total_scores:
                        tag_total_scores[layer] += score
                except ValueError:
                    continue
    for key in args_keys:
        if key in param_importance and isinstance(param_importance[key], dict):
            layer_scores = param_importance[key]
            for layer_str, score in layer_scores.items():
                try:
                    layer = int(layer_str)
                    if layer in args_total_scores:
                        args_total_scores[layer] += score
                except ValueError:
                    continue
    tag_sorted_layers = sorted(tag_total_scores.items(), key=lambda x: x[1], reverse=True)
    tag_topk_layers = [str(layer) for layer, _ in tag_sorted_layers[:top_k]]
    args_sorted_layers = sorted(args_total_scores.items(), key=lambda x: x[1], reverse=True)
    args_topk_layers = [str(layer) for layer, _ in args_sorted_layers[:top_k]]
    original_grad_states = {name: param.requires_grad for name, param in model.named_parameters()}
    action_train_dict = {}
    for name, param in model.named_parameters():
        if not original_grad_states[name]:
            action_train_dict[name] = False
            continue
        in_tag_topk_layer = any(f"layers.{layer}." in name for layer in tag_topk_layers)
        is_key_proj = "q_proj" in name or "v_proj" in name
        action_train_dict[name] = in_tag_topk_layer and is_key_proj
    args_train_dict = {}
    for name, param in model.named_parameters():
        if not original_grad_states[name]:
            args_train_dict[name] = False
            continue
        in_args_topk_layer = any(f"layers.{layer}." in name for layer in args_topk_layers)
        is_key_proj = "q_proj" in name or "v_proj" in name
        args_train_dict[name] = in_args_topk_layer and is_key_proj
    return action_train_dict, args_train_dict


def create_action_token_mask(
    completion_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    base_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len = completion_ids.shape
    action_mask = torch.zeros_like(completion_ids, dtype=torch.bool)
    action_tokens = ["<search>", "</search>", "<reasoning>", "</reasoning>", "<backtrack>", "</backtrack>", "<summary>", "</summary>"]
    for token in action_tokens:
        token_id_variants = [
            tokenizer.encode(token, add_special_tokens=False),
            tokenizer.encode(" " + token, add_special_tokens=False),
            tokenizer.encode("\n" + token, add_special_tokens=False),
            tokenizer.encode(token + " ", add_special_tokens=False),
            tokenizer.encode(token + "\n", add_special_tokens=False),
            tokenizer.encode(" " + token + "\n", add_special_tokens=False),
            tokenizer.encode("\n" + token + "\n", add_special_tokens=False),
        ]
        for token_ids in token_id_variants:
            if not token_ids:
                continue
            token_len = len(token_ids)
            token_tensor = torch.tensor(token_ids, device=completion_ids.device)
            for b in range(batch_size):
                for i in range(seq_len - token_len + 1):
                    if torch.all(completion_ids[b, i : i + token_len] == token_tensor):
                        action_mask[b, i : i + token_len] = True
    return action_mask & base_mask.bool()


def create_args_content_mask(
    completion_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    base_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len = completion_ids.shape
    action_mask = torch.zeros_like(completion_ids, dtype=torch.bool)
    action_tokens = ["<search>", "</search>", "<reasoning>", "</reasoning>", "<backtrack>", "</backtrack>", "<summary>", "</summary>"]
    for token in action_tokens:
        token_id_variants = [
            tokenizer.encode(token, add_special_tokens=False),
            tokenizer.encode(" " + token, add_special_tokens=False),
            tokenizer.encode("\n" + token, add_special_tokens=False),
            tokenizer.encode(token + " ", add_special_tokens=False),
            tokenizer.encode(token + "\n", add_special_tokens=False),
            tokenizer.encode(" " + token + "\n", add_special_tokens=False),
            tokenizer.encode("\n" + token + "", add_special_tokens=False),
            tokenizer.encode(" " + token + "", add_special_tokens=False),
            tokenizer.encode("\n" + token + "\n", add_special_tokens=False),
        ]
        for token_ids in token_id_variants:
            if not token_ids:
                continue
            token_len = len(token_ids)
            token_tensor = torch.tensor(token_ids, device=completion_ids.device)
            for b in range(batch_size):
                for i in range(seq_len - token_len + 1):
                    if torch.all(completion_ids[b, i : i + token_len] == token_tensor):
                        action_mask[b, i : i + token_len] = True
    args_mask = torch.ones_like(action_mask, dtype=torch.bool) ^ action_mask
    return args_mask & base_mask.bool()
