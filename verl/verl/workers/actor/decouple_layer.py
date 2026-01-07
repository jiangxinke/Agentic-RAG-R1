# -*- coding: utf-8 -*-
"""
Action / Args structured credit assignment utilities.

This module is PPO-agnostic.
It provides:
  1. Action / Args token masks (on full input_ids, restricted by response_mask)
  2. Layer → parameter partition (action layers vs args layers)

NO optimizer / backward logic here.
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Try to import TokenRecognizer
try:
    from verl.spr1_special_tokens.recognition import TokenRecognizer, DEFAULT_ACTION_CONFIG
except ImportError:
    # Fallback or error if not found (though it should be there based on user context)
    TokenRecognizer = Any 
    DEFAULT_ACTION_CONFIG = {}

# ============================================================
# 1. Token-level masks
# ============================================================

def create_action_token_mask(input_ids: torch.LongTensor, recognizer: TokenRecognizer) -> torch.BoolTensor:
    """
    Generate action mask based on TokenRecognizer.
    Action mask includes the start and end tags of actions.
    """
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    action_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
    
    # Process each sequence in the batch
    # We can use recognizer.segment_batch or find_intervals. 
    # find_intervals is more direct if we just want tags.
    # However, segment_batch gives us structured info. 
    # Let's iterate over DEFAULT_ACTION_CONFIG to find all tags.
    
    # We need to handle the case where recognizer is not provided or valid? 
    # The caller must provide a valid recognizer.
    
    input_ids_list = input_ids.tolist()
    
    for b_idx, seq_ids in enumerate(input_ids_list):
        for action_name, (start_token, end_token) in DEFAULT_ACTION_CONFIG.items():
            # Find intervals including boundaries (tags)
            intervals = recognizer.find_intervals(seq_ids, start_token, end_token, include_boundaries=True)
            
            for start, end in intervals:
                # interval is [start, end)
                # "Action" tokens are the tags themselves.
                # Assuming single-token tags (as enforced by TokenRecognizer currently)
                # The start tag is at `start`
                # The end tag is at `end - 1`
                
                if start < seq_len:
                    action_mask[b_idx, start] = True
                if end - 1 < seq_len and end > start:
                    action_mask[b_idx, end - 1] = True
                    
    return action_mask

def create_args_content_mask(input_ids: torch.LongTensor, recognizer: TokenRecognizer) -> torch.BoolTensor:
    """
    Generate args mask based on TokenRecognizer.
    Args mask includes the content BETWEEN the start and end tags.
    """
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    args_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
    
    input_ids_list = input_ids.tolist()
    
    for b_idx, seq_ids in enumerate(input_ids_list):
        for action_name, (start_token, end_token) in DEFAULT_ACTION_CONFIG.items():
            # Find intervals strictly between tags
            intervals = recognizer.find_intervals(seq_ids, start_token, end_token, include_boundaries=False)
            
            for start, end in intervals:
                # interval is [start, end)
                # These are the content tokens.
                if start < end: # valid interval
                     # Clamp to seq_len just in case
                    s = max(0, start)
                    e = min(seq_len, end)
                    args_mask[b_idx, s:e] = True

    return args_mask

def build_action_args_masks_from_model_inputs(
    model_inputs: Dict[str, torch.Tensor],
    recognizer: Optional[TokenRecognizer] = None
) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Build action / args masks directly from PPO-style model_inputs.

    Required keys in model_inputs:
      - input_ids
      - response_mask
      
    Args:
        model_inputs: Dict containing input tensors
        recognizer: TokenRecognizer instance to identify special tokens
    
    Returns:
        action_mask, args_mask
    """
    input_ids = model_inputs["input_ids"]           # [prompt tokens | response tokens]
    response_mask = model_inputs["response_mask"].bool()  # input_ids[:, -response_len:] -> 对应 response_mask

    # print(response_mask.shape, input_ids.shape)

    if recognizer is None:
        # Fallback to zeros or raise error? 
        # Given the user wants to replace mock, let's warn and return zeros if no recognizer
        print("[Warning] No TokenRecognizer provided to build_action_args_masks_from_model_inputs. Returning empty masks.")
        bsz, seq_len = input_ids.shape
        return torch.zeros((bsz, response_mask.shape[1]), dtype=torch.bool, device=input_ids.device), \
               torch.zeros((bsz, response_mask.shape[1]), dtype=torch.bool, device=input_ids.device)

    action_mask = create_action_token_mask(
        input_ids=input_ids,
        recognizer=recognizer
    )

    args_mask = create_args_content_mask(
        input_ids=input_ids,
        recognizer=recognizer
    )

    # 在这个地方返回的时候，需要把action_mask和args_mask给对齐到response_mask的shape
    # Ensure we assume the response is at the end
    action_mask, args_mask = action_mask[:, -response_mask.shape[1]:], args_mask[:, -response_mask.shape[1]:]

    return action_mask, args_mask


# ============================================================
# 2. Layer → parameter partition (model-side, PPO-agnostic)
# ============================================================
def generate_grad_control_dicts(
    model: nn.Module,
    param_importance_path: str = str(Path(__file__).resolve().parent / "neuron" / "neuron_active_ratio.json"),
    top_k: int = 8):
    """
    为Qwen2.5-1.5B-Instruct模型生成action和args的参数集合（FSDP 兼容）
    基于参数重要性字典，汇总tag/args得分后选择Top-K层参数，仅保留对应层的更新权限

    Args:
        model: 目标模型（Qwen2.5-1.5B-Instruct）
        param_importance_path: 参数重要性JSON文件的路径
        top_k: 选择Top-K重要性的层进行训练（默认8层）

    Returns:
        action_params: set of Parameters 对象，用于 action loss
        args_params: set of Parameters 对象，用于 args loss
    """

    # 读取 Top-K 层列表（假设 import_grad_control_dicts 已返回 tag / args Top-K 层列表）
    tag_topk_layers, args_topk_layers = import_grad_control_dicts(model, param_importance_path, top_k)

    print(f"基于tag得分的Top-{top_k}层: {tag_topk_layers}")
    print(f"基于args得分的Top-{top_k}层: {args_topk_layers}")

    action_params: Set[torch.nn.Parameter] = set()
    args_params: Set[torch.nn.Parameter] = set()

    # 获取 FSDP 包装的裸 module
    fsdp_module = getattr(model, "_fsdp_wrapped_module", model)

    # 遍历所有参数
    for name, p in fsdp_module.named_parameters():
        if not p.requires_grad:
            continue

        # 尝试解析层号（一般是 layers.{layer_id}）
        layer_id = None
        for tok in name.split("."):
            if tok.isdigit():
                layer_id = int(tok)
                break
        if layer_id is None:
            continue  # 非 transformer 层，如 embedding / head

        # 判断是否是 action / args Top-K 层
        if str(layer_id) in tag_topk_layers:
            action_params.add(p)
        if str(layer_id) in args_topk_layers:
            args_params.add(p)

    print(f"[generate_grad_control_dicts] Action params count: {len(action_params)}")
    print(f"[generate_grad_control_dicts] Args params count: {len(args_params)}")

    return action_params, args_params

def import_grad_control_dicts(model, param_importance_path=str(Path(__file__).resolve().parent / "neuron" / "neuron_active_ratio.json"), top_k=8):
    """
    为Qwen2.5-1.5B-Instruct模型生成action和args的梯度控制字典
    基于参数重要性字典，汇总tag/args得分后选择Top-K层参数，仅保留原始可训练参数的更新权限
    
    Args:
        model: 目标模型（Qwen2.5-1.5B-Instruct）
        param_importance_path: 参数重要性JSON文件的路径
        top_k: 选择Top-K重要性的层进行训练（默认8层）
    
    Returns:
        action_train_dict: action模式的梯度控制字典（基于tag类得分Top-K层）
        args_train_dict: args模式的梯度控制字典（基于args类得分Top-K层）
    """
    # 读取参数重要性文件
    try:
        with open(param_importance_path, 'r', encoding='utf-8') as f:
            param_importance = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"参数重要性文件未找到：{param_importance_path}")
    except json.JSONDecodeError:
        raise ValueError(f"文件格式错误，无法解析JSON：{param_importance_path}")
    except Exception as e:
        raise Exception(f"读取参数重要性文件时出错：{str(e)}")
    
    # -------------------------- 1. 动态确定层数 --------------------------
    # 从参数重要性字典中提取所有层号并确定最大层号
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
    num_layers = max_layer + 1  # 层号从0开始
    print(f"从参数重要性字典中检测到的层数: {num_layers} (0-{max_layer})")
    
    # -------------------------- 2. 定义tag/args分类并汇总得分 --------------------------
    tag_keys = ["reasoning_tag", "search_tag", "summary_tag", "answer_tag", "backtrack_tag"]
    args_keys = ["reasoning_arg", "search_arg", "summary_arg", "answer_arg", "backtrack_arg"]
    
    # 初始化得分字典（基于实际检测到的层数）
    tag_total_scores = {layer: 0.0 for layer in range(num_layers)}
    args_total_scores = {layer: 0.0 for layer in range(num_layers)}
    
    # 汇总tag类得分
    for key in tag_keys:
        if key in param_importance and isinstance(param_importance[key], dict):
            layer_scores = param_importance[key]
            for layer_str, score in layer_scores.items():
                try:
                    layer = int(layer_str)
                    if layer in tag_total_scores:  # 只处理有效层号
                        tag_total_scores[layer] += score
                except ValueError:
                    continue
    
    # 汇总args类得分
    for key in args_keys:
        if key in param_importance and isinstance(param_importance[key], dict):
            layer_scores = param_importance[key]
            for layer_str, score in layer_scores.items():
                try:
                    layer = int(layer_str)
                    if layer in args_total_scores:  # 只处理有效层号
                        args_total_scores[layer] += score
                except ValueError:
                    continue
    
    # -------------------------- 3. 筛选Top-K重要性的层 --------------------------
    # 筛选tag类Top-K层（按得分降序，取前top_k层）
    tag_sorted_layers = sorted(tag_total_scores.items(), key=lambda x: x[1], reverse=True)
    tag_topk_layers = [str(layer) for layer, _ in tag_sorted_layers[:top_k]]  # 转为字符串匹配参数名
    
    # 筛选args类Top-K层（按得分降序，取前top_k层）
    args_sorted_layers = sorted(args_total_scores.items(), key=lambda x: x[1], reverse=True)
    args_topk_layers = [str(layer) for layer, _ in args_sorted_layers[:top_k]]  # 转为字符串匹配参数名
    
    return tag_topk_layers, args_topk_layers


def build_layer_train_dict(
    model,
    action_top_k=3,
    args_top_k=5,
) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """
    Build parameter-name → trainable dicts for action / args.

    Strategy:
      - select top-k transformer layers
      - other parameters set to False

    Args:
        model:        HF causal LM
        action_top_k: number of top layers for action tokens
        args_top_k:   number of top layers for args tokens

    Returns:
        action_train_dict, args_train_dict
    """
    num_layers = model.config.num_hidden_layers

    action_layers = set(range(num_layers - action_top_k, num_layers))
    args_layers = set(range(num_layers - args_top_k, num_layers))

    action_train_dict: Dict[str, bool] = {}
    args_train_dict: Dict[str, bool] = {}

    for name, _ in model.named_parameters():
        action_train_dict[name] = False
        args_train_dict[name] = False

        for lid in action_layers:
            if f".layers.{lid}." in name:
                action_train_dict[name] = True
                break

        for lid in args_layers:
            if f".layers.{lid}." in name:
                args_train_dict[name] = True
                break

    return action_train_dict, args_train_dict

if __name__ == "__main__":
    generate_grad_control_dicts("2")
