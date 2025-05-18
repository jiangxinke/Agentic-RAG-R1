import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class AgenticRAGCritic(nn.Module):
    """
    Critic网络，用于PPO训练中的价值函数估计。
    支持与主模型解耦，输入prompt+completion的token ids和mask，
    输出每个样本的V值。
    支持保存/加载、损失计算、LoRA/量化等扩展。
    """
    def __init__(
        self,
        model_name,
        device,
        torch_dtype="float32",
        loss_type="mse",  # "mse" or "huber"
        lora_config=None,
        quant_config=None,
        **kwargs
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=getattr(torch, torch_dtype),
            trust_remote_code=True,
        ).to(device)
        self.loss_type = loss_type
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # 关键补丁：设置pad_token_id
        if not hasattr(self.model.config, "pad_token_id") or self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def forward(self, input_ids, attention_mask, values=None):
        """
        推理模式：只输入input_ids和attention_mask，返回V值
        训练模式：额外输入values（target），返回loss
        """
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        if values is not None:
            loss = self.loss_fn(logits, values)
            return loss
        return logits

    def save(self, save_path):
        self.model.save_pretrained(save_path)

    def load(self, load_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)

    def get_value(self, input_ids, attention_mask):
        """
        推理接口，返回V值（detach，cpu）
        """
        with torch.no_grad():
            v = self.forward(input_ids, attention_mask)
            return v.detach().cpu() 