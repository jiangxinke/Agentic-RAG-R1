import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Callable, Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# TODO: 需要写入读取参数layer的代码

def compute_loss_for_layers_simple(  
    model: torch.nn.Module,  
    rollout_data: Dict[str, Any],  
    layers: List[int],  
    mask_type: str  
) -> torch.Tensor:  
    """  
    使用简单的requires_grad控制来实现分层梯度计算  
    """  
    # 1. 先冻结所有参数  
    for p in model.parameters():  
        p.requires_grad_(False)  
      
    # 2. 只解冻目标层的参数  
    for name, param in model.named_parameters():  
        if any(f"layers.{layer}" in name for layer in layers):  
            param.requires_grad_(True)  
      
    # 3. 计算loss（使用对应的mask）  
    loss = compute_masked_loss(model, rollout_data, mask_type)  
      
    # 4. 恢复所有参数的梯度设置  
    for p in model.parameters():  
        p.requires_grad_(True)  
      
    return loss

def create_action_token_mask(  
    completion_ids: torch.Tensor,  
    tokenizer: AutoTokenizer,  
    base_mask: torch.Tensor,  
) -> torch.Tensor:  
    """  
    创建只包含action token的mask - 只标记<search>、</search>等标记本身  
    """  
    batch_size, seq_len = completion_ids.shape  
    action_mask = torch.zeros_like(completion_ids, dtype=torch.int)  
      
    # 只需要标记这些token本身  
    action_tokens = ["<search>", "</search>", "<reasoning>", "</reasoning>",   
                    "<backtrack>", "</backtrack>", "<summary>", "</summary>"]  
      
    for token in action_tokens:  
        token_ids = tokenizer(token).input_ids  
        token_len = len(token_ids)  
        token_tensor = torch.tensor(token_ids, device=completion_ids.device)  
          
        for b in range(batch_size):  
            for i in range(seq_len - token_len + 1):  
                if torch.all(completion_ids[b, i:i + token_len] == token_tensor):  
                    action_mask[b, i:i + token_len] = 1  
      
    return action_mask & base_mask
    
  
def create_args_content_mask(  
    completion_ids: torch.Tensor,  
    tokenizer: AutoTokenizer,  
    base_mask: torch.Tensor,  
) -> torch.Tensor:  
    """  
    创建只包含action标记内容的mask（标记之间的内容）  
    """  
    batch_size, seq_len = completion_ids.shape  
    args_mask = torch.zeros_like(completion_ids, dtype=torch.int)  
      
    # 定义所有action标记  
    action_tags = [  
        ("<search>", "</search>"),  
        ("<reasoning>", "</reasoning>"),  
        ("<backtrack>", "</backtrack>"),  
        ("<summary>", "</summary>"),  
    ]  
      
    for start_tag, end_tag in action_tags:  
        start_ids = tokenizer(start_tag).input_ids  
        end_ids = tokenizer(end_tag).input_ids  
          
        start_len = len(start_ids)  
        end_len = len(end_ids)  
        start_tensor = torch.tensor(start_ids, device=completion_ids.device)  
        end_tensor = torch.tensor(end_ids, device=completion_ids.device)  
          
        for b in range(batch_size):  
            in_tag = False  
            tag_start_pos = None  
              
            for i in range(seq_len - max(start_len, end_len) + 1):  
                # 检查开始标记  
                if i <= seq_len - start_len and torch.all(completion_ids[b, i:i + start_len] == start_tensor):  
                    in_tag = True  
                    tag_start_pos = i + start_len  
                  
                # 检查结束标记  
                if i <= seq_len - end_len and torch.all(completion_ids[b, i:i + end_len] == end_tensor):  
                    if in_tag and tag_start_pos is not None:  
                        # 标记开始和结束标记之间的内容  
                        args_mask[b, tag_start_pos:i] = 1  
                    in_tag = False  
                    tag_start_pos = None  
      
    # 应用基础mask  
    return args_mask & base_mask