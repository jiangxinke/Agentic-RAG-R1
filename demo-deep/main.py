import torch
from torch.utils.data import Dataset, DataLoader
import deepspeed

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


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


def main():

    # -----------------
    # 1. Load model & tokenizer
    # -----------------
    # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_name = "Qwen/Qwen1.5-0.5B-Chat"
    model_name = "Qwen/Qwen-7B-Chat"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # -----------------
    # 2. Wrap with LoRA
    # -----------------
    lora_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.1, bias="none", target_modules="q_proj,v_proj", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # -----------------
    # 3. Create optimizer (any standard torch optimizer)
    # -----------------
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # -----------------
    # 4. Create DataLoader
    # -----------------
    dummy_dataset = DummyDataset(tokenizer, dummy_text="Hello, world!", num_samples=100)
    dataloader = DataLoader(dummy_dataset, batch_size=4, shuffle=True)

    # -----------------
    # 5. Initialize DeepSpeed
    # -----------------
    # ds_config.json must exist in your working directory
    ds_config = "config/ds_config_zero3.json"

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

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), optimizer=optimizer, config=ds_config
    )

    print("++++" * 100)
    policy_state_dict = model.state_dict()
    for key, value in policy_state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            print(f"{key}: {value.shape}")
    print("++++" * 100)

    # -----------------
    # 6. Training loop
    # -----------------
    num_epochs = 3
    for epoch in range(num_epochs):
        model_engine.train()
        for step, batch in enumerate(dataloader):

            # Move input tensors to device used by DeepSpeed
            for k, v in batch.items():
                batch[k] = v.to(model_engine.local_rank)

            # Forward pass
            outputs = model_engine(**batch, labels=batch["input_ids"])
            loss = outputs.loss

            # Backward + step
            model_engine.backward(loss)
            model_engine.step()

            if step % 10 == 0:
                print(f"Epoch {epoch} | step {step} | loss {loss.item()}")

    # -----------------
    # 7. Save only LoRA weights (PEFT)
    # -----------------
    # If you just want to save LoRA adapter weights:
    model_engine.save_pretrained("my_lora_output")

    # If you also want to save a full DeepSpeed checkpoint, you can do:
    # model_engine.save_checkpoint("my_deepspeed_ckpt")

    print("Training completed.")


if __name__ == "__main__":
    main()
