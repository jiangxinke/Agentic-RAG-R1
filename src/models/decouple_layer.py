from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# TODO: 现在只是Mock了一个写入读取参数layer的代码，后续这里是需要调用的
def generate_grad_control_dicts(model):
    """
    为Qwen2.5-1.5B-Instruct模型生成action和args的梯度控制字典
    基于原始的requires_grad状态，只能将True改为True/False，不能将False改为True
    """
    # 获取所有参数名称和原始梯度状态
    all_param_names = [name for name, _ in model.named_parameters()]
    original_grad_states = {name: param.requires_grad for name, param in model.named_parameters()}

    # 打印原始状态中哪些参数可以更新
    trainable_params = [name for name, requires_grad in original_grad_states.items() if requires_grad]
    print(f"原始可训练参数数量: {len(trainable_params)}")
    print("原始可训练参数:")
    for name in trainable_params:
        print(f"  {name}")

    # 找出所有层数（Qwen2.5-1.5B通常有24层）
    layer_nums = set()
    for name in trainable_params:  # 只在可训练参数中查找
        if "layers." in name and "lora" in name:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_num = int(parts[i + 1])
                        layer_nums.add(layer_num)
                    except ValueError:
                        continue

    max_layer = max(layer_nums) if layer_nums else 23
    print(f"检测到的层数范围: 0-{max_layer}")

    # Action字典：只更新前两层的LoRA参数
    action_train_dict = {}
    action_trainable_count = 0
    for name, param in model.named_parameters():
        if original_grad_states[name]:  # 只有原本可训练的参数才考虑
            if "lora" in name and any(f"layers.{i}." in name for i in [0, 1]):
                if "q_proj" in name or "v_proj" in name:
                    action_train_dict[name] = True
                    action_trainable_count += 1
                else:
                    action_train_dict[name] = False
            else:
                action_train_dict[name] = False
        else:
            action_train_dict[name] = False  # 保持原始的False状态

    # Args字典：只更新后两层的LoRA参数
    args_train_dict = {}
    args_trainable_count = 0
    target_layers = [max_layer - 1, max_layer]
    for name, param in model.named_parameters():
        if original_grad_states[name]:  # 只有原本可训练的参数才考虑
            if "lora" in name and any(f"layers.{i}." in name for i in target_layers):
                if "q_proj" in name or "v_proj" in name:
                    args_train_dict[name] = True
                    args_trainable_count += 1
                else:
                    args_train_dict[name] = False
            else:
                args_train_dict[name] = False
        else:
            args_train_dict[name] = False  # 保持原始的False状态

    # 打印结果对比
    print(f"\nAction训练模式 (前两层):")
    print(f"  可训练参数数量: {action_trainable_count}")
    action_true_params = [name for name, val in action_train_dict.items() if val]
    for name in action_true_params:
        print(f"  {name}")

    print(f"\nArgs训练模式 (后两层):")
    print(f"  可训练参数数量: {args_trainable_count}")
    args_true_params = [name for name, val in args_train_dict.items() if val]
    for name in args_true_params:
        print(f"  {name}")

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