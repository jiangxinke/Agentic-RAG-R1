from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
import torch
from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    def __init__(self, tokenizer, dummy_text="Hello, world!", num_samples=100):
        self.tokenizer = tokenizer
        self.dummy_text = dummy_text
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.dummy_text, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        return item


accelerator = Accelerator()

model_name = "Qwen/Qwen1.5-0.5B-Chat"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
dummy_dataset = DummyDataset(tokenizer, dummy_text="Hello, world!", num_samples=100)
dataloader = DataLoader(dummy_dataset, batch_size=4, shuffle=True)


print("++++" * 100)
policy_state_dict = model.state_dict()
for key, value in policy_state_dict.items():
    if "lora_A" in key or "lora_B" in key:
        print(f"{key}: {value.shape}")
print("++++" * 100)
print("====" * 100)
print("====" * 100)
print("====" * 100)
print("====" * 100)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)


print("++++" * 100)
policy_state_dict = model.state_dict()
for key, value in policy_state_dict.items():
    if "lora_A" in key or "lora_B" in key:
        print(f"{key}: {value.shape}")
print("++++" * 100)

# 训练循环示例
model.train()
num_epochs = 3  # 总共训练 3 个 epoch

# for epoch in range(num_epochs):
#     for step, batch in enumerate(dataloader):
#         # 将输入数据转移到 accelerator.device 已经在 prepare 阶段完成
#         input_ids = batch["input_ids"]
#         attention_mask = batch.get("attention_mask", None)

#         # 使用自回归损失计算
#         outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
#         loss = outputs.loss

#         # 反向传播与优化步骤
#         accelerator.backward(loss)
#         optimizer.step()
#         optimizer.zero_grad()

#         print(f"Epoch {epoch+1}, Step {step}: loss = {loss.item()}")

# print("训练完成！")
