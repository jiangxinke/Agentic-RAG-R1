import random
import re

from typing import Any, List, Optional, Tuple

import json5
import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    StoppingCriteria,
    StoppingCriteriaList,
)

import src.neuron.parse_tokens as parse_tokens
from src.neuron.neuron_metric import NeuronMetric
from src.utils.Tools import Tools


def apply_masked_spans(
    current_mask: torch.LongTensor,
    masked_spans_per_sample: list[ list[tuple[int,int,int]] ],
) -> torch.LongTensor:
    """
    根据 masked_spans_per_sample 动态屏蔽 attention mask。

    Args:
        current_mask (torch.LongTensor): 当前 attention mask, shape (B, T)
        masked_spans_per_sample (List[List[Tuple[int,int,int]]]): 
            每个 batch 的 span 三元组列表 (prev_action_start, prev_action_end, backtrack_end)

    Returns:
        torch.LongTensor: 更新后的 attention mask
    """
    batch_size, seq_len = current_mask.shape
    new_mask = current_mask.clone()

    for b in range(batch_size):
        for span in masked_spans_per_sample[b]:
            prev_start, prev_end, backtrack_end = span
            # 安全检查，防止越界
            prev_start = max(0, prev_start)
            prev_end = min(prev_end, seq_len)
            if prev_start < prev_end:
                new_mask[b, prev_start:prev_end] = 0  # 屏蔽前一个 action

    return new_mask

def get_masked_spans_from_text(full_seq: str, tokenizer) -> List[Tuple[int, int, int]]:
    # TODO Rihong修改这里
    """
    Mock function: 给定完整文本，返回需要屏蔽的 span 三元组。
    
    这里先返回固定示例，后续你可以改成根据标记解析真实 span。
    
    Returns:
        List of tuples: (prev_action_start, prev_action_end, backtrack_end)
    """
    from src.neuron.action_utils import get_last_two_action_span

    # TODO: 根据 full_text 解析真实 span
    result = get_last_two_action_span(full_seq, tokenizer)
    # 现在先返回示例数据
    return [result]


def expand_to_causal_mask(current_mask: torch.Tensor, dtype=torch.float32):  
    """  
    将 2D padding mask (B, L) 转换为 4D causal attention mask (B, 1, L, L)  
      
    Args:  
        current_mask: 2D tensor (B, L), 1 表示有效位置, 0 表示 padding  
        dtype: 输出 mask 的数据类型(必须是浮点类型)  
      
    Returns:  
        4D causal mask (B, 1, L, L), 0.0 表示允许注意, min_dtype 表示禁止注意  
    """  
    # 形状校验  
    if current_mask.ndim != 2:  
        raise ValueError(f"current_mask must be 2D (B,T), got {current_mask.ndim}D")  
    B, T = current_mask.shape  
      
    # 确保 dtype 是浮点类型  
    if not dtype.is_floating_point:  
        raise TypeError(f"dtype must be a floating point type, got {dtype}")  
      
    # 获取 dtype 的最小值  
    min_dtype = torch.finfo(dtype).min  
      
    # 创建基础 causal mask: 初始化为 min_dtype (全部屏蔽)  
    causal_mask = torch.full(  
        (T, T), fill_value=min_dtype, dtype=dtype, device=current_mask.device  
    )  
      
    # 应用下三角: 对角线及以下设为 0 (允许注意)  
    causal_mask = torch.triu(causal_mask, diagonal=1)  
      
    # 扩展到 batch 维度: (T, T) -> (B, 1, T, T)  
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1)  
      
    # 合并 padding mask  
    causal_mask = causal_mask.clone()  
      
    # 将 padding 位置设为 min_dtype  
    # current_mask 为 0 的位置(padding)应该在所有 query 位置都被屏蔽  
    mask_cond = current_mask[:, None, None, :] == 0  # (B, 1, 1, T)  
    causal_mask = causal_mask.masked_fill(mask_cond, min_dtype)  
      
    return causal_mask  # (B, 1, T, T)

import torch
import matplotlib.pyplot as plt

def expand_to_causal_mask_backtrack(current_mask: torch.Tensor, masked_spans_per_sample, dtype=torch.float32):
    """
    将 2D padding mask (B,L) 转换为 4D causal attention mask (B,1,L,L)
    并根据 masked_spans_per_sample 动态屏蔽 attention。
    masked_spans_per_sample: List[List[Tuple(prev_start, prev_end, backtrack_end)]]
    """
    B, T = current_mask.shape
    min_dtype = torch.finfo(dtype).min

    # 基础下三角 causal mask
    causal_mask = torch.full((T, T), fill_value=min_dtype, dtype=dtype, device=current_mask.device)
    causal_mask = torch.triu(causal_mask, diagonal=1)  # 对角线及以下为0
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1).clone()

    # padding mask
    pad_mask_cond = current_mask[:, None, None, :] == 0  # (B,1,1,T)
    causal_mask = causal_mask.masked_fill(pad_mask_cond, min_dtype)

    # 动态屏蔽 span
    for b in range(B):
        for span in masked_spans_per_sample[b]:
            prev_start, prev_end, backtrack_end = span
            # 左闭右闭转换到 Python 切片=>parellel
            causal_mask[b, 0, prev_start:prev_end+1, backtrack_end+1:] = min_dtype
            causal_mask[b, 0, backtrack_end+1:, prev_start:prev_end+1] = min_dtype
    return causal_mask

def expand_to_causal_mask_parallel(current_mask: torch.Tensor, masked_parallel_spans_per_sample, dtype=torch.float32):
    """
    将 2D padding mask (B,L) 转换为 4D causal attention mask (B,1,L,L)
    并根据 masked_parallel_spans_per_sample 屏蔽不同 span 之间的 attention。

    masked_parallel_spans_per_sample: List[batch] -> List[rollout] -> List[Tuple(start,end)]
    """
    B, T = current_mask.shape
    min_dtype = torch.finfo(dtype).min

    # 基础 causal mask
    causal_mask = torch.full((T, T), fill_value=min_dtype, dtype=dtype, device=current_mask.device)
    causal_mask = torch.triu(causal_mask, diagonal=1)  # 对角线以下为 0
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1).clone()

    # padding mask
    pad_mask_cond = current_mask[:, None, None, :] == 0
    causal_mask = causal_mask.masked_fill(pad_mask_cond, min_dtype)

    # 遍历 batch
    for b in range(B):
        batch_spans = masked_parallel_spans_per_sample[b]  # List of rollout
        for rollout_spans in batch_spans:  # List of (start,end)
            # 两两屏蔽
            for i, span_i in enumerate(rollout_spans):
                start_i, end_i = span_i
                for j, span_j in enumerate(rollout_spans):
                    if i == j:
                        continue  # 不屏蔽自己
                    start_j, end_j = span_j
                    # query in span_i 屏蔽 key in span_j
                    causal_mask[b, 0, start_i:end_i+1, start_j:end_j+1] = min_dtype
                    causal_mask[b, 0, start_j:end_j+1, start_i:end_i+1] = min_dtype
    return causal_mask  # (B,1,T,T)


if __name__ == "__main__":
    # ==== 测试 ====
    B, L = 2, 10
    attention_mask = torch.ones(B, L, dtype=torch.int64)
    masked_spans_per_sample = [
        [(1, 3, 5)],   # batch 0: 屏蔽 [1,3] 对应 query 到 backtrack_end+1 之后 key
        [(2, 4, 6)]    # batch 1: 屏蔽 [2,4] 对应 query 到 backtrack_end+1 之后 key
    ]

    mask = expand_to_causal_mask_backtrack(attention_mask, masked_spans_per_sample, dtype=torch.float32)
    print(mask)