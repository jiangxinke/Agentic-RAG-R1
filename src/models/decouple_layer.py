from typing import Any, Callable, Dict, List, Optional, Tuple

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# TODO: 现在只是Mock了一个写入读取参数layer的代码，后续这里是需要调用的
# def generate_grad_control_dicts(model):
#     """
#     为Qwen2.5-1.5B-Instruct模型生成action和args的梯度控制字典
#     基于原始的requires_grad状态，只能将True改为True/False，不能将False改为True
#     """
#     # 获取所有参数名称和原始梯度状态
#     all_param_names = [name for name, _ in model.named_parameters()]
#     original_grad_states = {name: param.requires_grad for name, param in model.named_parameters()}

#     # 打印原始状态中哪些参数可以更新
#     trainable_params = [name for name, requires_grad in original_grad_states.items() if requires_grad]
#     # print(f"原始可训练参数数量: {len(trainable_params)}")
#     # print("原始可训练参数:")
#     # for name in trainable_params:
#     #     print(f"  {name}")

#     # 找出所有层数（Qwen2.5-1.5B通常有24层）
#     layer_nums = set()
#     for name in trainable_params:  # 只在可训练参数中查找
#         if "layers." in name and "lora" in name:
#             parts = name.split(".")
#             for i, part in enumerate(parts):
#                 if part == "layers" and i + 1 < len(parts):
#                     try:
#                         layer_num = int(parts[i + 1])
#                         layer_nums.add(layer_num)
#                     except ValueError:
#                         continue

#     max_layer = max(layer_nums) if layer_nums else 23
#     # print(f"检测到的层数范围: 0-{max_layer}")

#     # Action字典：只更新前两层的LoRA参数
#     action_train_dict = {}
#     action_trainable_count = 0
#     for name, param in model.named_parameters():
#         if original_grad_states[name]:  # 只有原本可训练的参数才考虑
#             if "lora" in name and any(f"layers.{i}." in name for i in [0, 1]):
#                 if "q_proj" in name or "v_proj" in name:
#                     action_train_dict[name] = True
#                     action_trainable_count += 1
#                 else:
#                     action_train_dict[name] = False
#             else:
#                 action_train_dict[name] = False
#         else:
#             action_train_dict[name] = False  # 保持原始的False状态

#     # Args字典：只更新后两层的LoRA参数
#     args_train_dict = {}
#     args_trainable_count = 0
#     target_layers = [max_layer - 1, max_layer]
#     for name, param in model.named_parameters():
#         if original_grad_states[name]:  # 只有原本可训练的参数才考虑
#             if "lora" in name and any(f"layers.{i}." in name for i in target_layers):
#                 if "q_proj" in name or "v_proj" in name:
#                     args_train_dict[name] = True
#                     args_trainable_count += 1
#                 else:
#                     args_train_dict[name] = False
#             else:
#                 args_train_dict[name] = False
#         else:
#             args_train_dict[name] = False  # 保持原始的False状态

#     # 打印结果对比
#     # print(f"\nAction训练模式 (前两层):")
#     # print(f"  可训练参数数量: {action_trainable_count}")
#     action_true_params = [name for name, val in action_train_dict.items() if val]
#     # for name in action_true_params:
#     #     print(f"  {name}")

#     # print(f"\nArgs训练模式 (后两层):")
#     # print(f"  可训练参数数量: {args_trainable_count}")
#     args_true_params = [name for name, val in args_train_dict.items() if val]
#     # for name in args_true_params:
#     #     print(f"  {name}")

#     return action_train_dict, args_train_dict

def generate_grad_control_dicts(model, param_importance_path=r"/home/xiaobei/qrh/Agentic-RAG-R1/output_eval/neuron/neuron_active_ratio.json", top_k=8):
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
    
    print(f"基于tag得分的Top-{top_k}层: {tag_topk_layers}")
    print(f"基于args得分的Top-{top_k}层: {args_topk_layers}")
    
    # -------------------------- 4. 获取原始参数梯度状态 --------------------------
    original_grad_states = {name: param.requires_grad for name, param in model.named_parameters()}
    print(f"模型总参数数量: {len(original_grad_states)}")
    print(f"原始可训练参数数量: {sum(1 for req in original_grad_states.values() if req)}")
    
    # -------------------------- 5. 生成action梯度控制字典（tag Top-K层） --------------------------
    action_train_dict = {}
    for name, param in model.named_parameters():
        # 基础规则：保持原始不可训练参数为False
        if not original_grad_states[name]:
            action_train_dict[name] = False
            continue
        
        # 对于原始可训练的参数，根据条件决定是否继续训练
        # 检查是否在tag Top-K层
        in_tag_topk_layer = any(f"layers.{layer}." in name for layer in tag_topk_layers)
        
        # 检查是否是LoRA参数（如果存在）
        has_lora = "lora" in name
        
        # 关键投影层检查
        is_key_proj = "q_proj" in name or "v_proj" in name
        
        # 决策逻辑：原始可训练参数中，只保留Top-K层的关键投影层（无论是否有LoRA）
        action_train_dict[name] = in_tag_topk_layer and is_key_proj
    
    # -------------------------- 6. 生成args梯度控制字典（args Top-K层） --------------------------
    args_train_dict = {}
    for name, param in model.named_parameters():
        # 基础规则：保持原始不可训练参数为False
        if not original_grad_states[name]:
            args_train_dict[name] = False
            continue
        
        # 对于原始可训练的参数，根据条件决定是否继续训练
        # 检查是否在args Top-K层
        in_args_topk_layer = any(f"layers.{layer}." in name for layer in args_topk_layers)
        
        # 检查是否是LoRA参数（如果存在）
        has_lora = "lora" in name
        
        # 关键投影层检查
        is_key_proj = "q_proj" in name or "v_proj" in name
        
        # 决策逻辑：原始可训练参数中，只保留Top-K层的关键投影层（无论是否有LoRA）
        args_train_dict[name] = in_args_topk_layer and is_key_proj
    
    # -------------------------- 7. 打印训练参数统计 --------------------------
    action_trainable_count = sum(1 for val in action_train_dict.values() if val)
    args_trainable_count = sum(1 for val in args_train_dict.values() if val)
    
    print(f"\nAction模式可训练参数数量: {action_trainable_count}")
    print(f"Args模式可训练参数数量: {args_trainable_count}")
    
    return action_train_dict, args_train_dict


def create_action_token_mask(
    completion_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    base_mask: torch.Tensor,
) -> torch.Tensor:
    """
    创建只包含action token的mask - 只标记<search>、</search>等标记本身
    """
    batch_size, seq_len = completion_ids.shape
    action_mask = torch.zeros_like(completion_ids, dtype=torch.bool)

    # 只需要标记这些token本身
    action_tokens = [
        "<search>",
        "</search>",
        "<reasoning>",
        "</reasoning>",
        "<backtrack>",
        "</backtrack>",
        "<summary>",
        "</summary>",
    ]

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

    true_count = torch.sum(base_mask).item()
    print("[Action Level] 原始的Completion Mask数量: ", true_count)
    true_count = torch.sum(action_mask).item()
    print("[Action Level] Action的Completion Mask数量: ", true_count)

    return action_mask & base_mask.bool()


def create_args_content_mask(
    completion_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    base_mask: torch.Tensor,
) -> torch.Tensor:
    """
    创建只包含action标记内容的mask（标记之间的内容）
    """
    batch_size, seq_len = completion_ids.shape
    action_mask = torch.zeros_like(completion_ids, dtype=torch.bool)

    # 只需要标记这些token本身
    action_tokens = [
        "<search>",
        "</search>",
        "<reasoning>",
        "</reasoning>",
        "<backtrack>",
        "</backtrack>",
        "<summary>",
        "</summary>",
    ]

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

    true_count = torch.sum(base_mask).item()
    print("[Args Level] 原始的Completion Mask数量: ", true_count)
    true_count = torch.sum(args_mask).item()
    print("[Args Level] Args的Completion Mask数量: ", true_count)

    # 应用基础mask
    return args_mask & base_mask.bool()