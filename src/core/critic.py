from transformers import AutoModelForSequenceClassification
import torch
import torch.nn as nn


class AgenticRAGCritic(nn.Module):
    def __init__(self, model_name, device, torch_dtype="float32", loss_type="mse", lora_config=None, quant_config=None, **kwargs):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=getattr(torch, torch_dtype),
            trust_remote_code=True,
        ).to(device)
        self.loss_type = loss_type
        self.loss_fn = nn.MSELoss() if loss_type == "mse" else nn.SmoothL1Loss()
        if not hasattr(self.model.config, "pad_token_id") or self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def forward(self, input_ids, attention_mask, values=None):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        if values is not None:
            return self.loss_fn(logits, values)
        return logits

    def save(self, save_path):
        self.model.save_pretrained(save_path)

    def load(self, load_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)

    def get_value(self, input_ids, attention_mask):
        with torch.no_grad():
            v = self.forward(input_ids, attention_mask)
            return v.detach().cpu()
