from __future__ import annotations

import logging
import os
import pdb
import random
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, List, Literal

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optimizer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from reward import format_reward, llm_eval_accuracy
from rich.logging import RichHandler
from rich.traceback import install
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

load_dotenv()
install()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(enable_link_path=False)])


def prepare_dataset_medqa(
    split: Literal["train", "validation", "test"] = "train",
    train_size: int = 10,
    eval_size: int = 10,
    test_size: int = 10,
):
    PROMPT = "请你解决后面的问题：\n"
    dataset = load_dataset("fzkuji/MedQA", "med_qa_zh_4options_bigbio_qa", split=split)
    formatted_dataset = []
    for index, example in enumerate(dataset):
        question, choices, answer_text = example["question"], example["choices"], example["answer"][0]
        try:
            answer_position = choices.index(answer_text)
        except ValueError:
            answer_position = -1
        answer_index = chr(65 + answer_position) if 0 <= answer_position < len(choices) else "N/A"
        options_string = "".join(f"{chr(65+j)}. {choice}\n" for j, choice in enumerate(choices))
        prompt = f"{PROMPT}Question: {question}\nOptions:\n{options_string}"
        formatted_dataset.append(
            {
                "id": index + 1,
                "input_text": prompt,
                "question_and_options": f"{question}\n{options_string}",
                "answer": f"{answer_index}: {answer_text}",
                "answer_text": answer_text,
                "answer_index": answer_index,
            }
        )
    train_dataset, eval_dataset, test_dataset = (
        formatted_dataset[:train_size],
        formatted_dataset[train_size : train_size + eval_size],
        formatted_dataset[train_size + eval_size : train_size + eval_size + test_size],
    )
    return Dataset.from_list(train_dataset), Dataset.from_list(eval_dataset), Dataset.from_list(test_dataset)


def prepare_dataloader_medqa(train_dataset, eval_dataset, test_dataset, train_batch_size=4, eval_batch_size=4, test_batch_size=1):
    return (
        DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True),
        DataLoader(eval_dataset, batch_size=eval_batch_size),
        DataLoader(test_dataset, batch_size=test_batch_size),
    )


def freeze_non_lora_parameters(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(keyword in name for keyword in ("lora_", "loraA", "loraB", "adapter"))
    return model


def extract_lora_state_dict(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if "lora_" in key}


def load_lora_state_dict(model, state_dict):
    model.load_state_dict(state_dict, strict=False)


def load_base_model(model_name, device_map=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, "float16"), device_map=device_map, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_model(model_name, device_map=None, lora_config=None):
    base_model, tokenizer = load_base_model(model_name, device_map)
    lora_config = lora_config or LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    freeze_non_lora_parameters(model)
    return model, tokenizer


def selective_log_softmax(logits, token_ids):
    log_probs = functional.log_softmax(logits, -1)
    return log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def compute_log_probabilities(model, input_token_ids, attention_mask, generation_length):
    output = model(
        input_ids=input_token_ids, attention_mask=attention_mask, logits_to_keep=generation_length + 1, obtain_logits=True
    ).logits  # NOTE 如果在自己定义的模型中实现了 .logits 这里就不用 .logits
    logits = output[:, :-1, :]
    selected_token_ids = input_token_ids[:, -generation_length:]
    return selective_log_softmax(logits[:, -generation_length:, :], selected_token_ids)


def compute_group_relative_advantages(reward, num_generation):
    reward_group = reward.view(-1, num_generation)
    mean, std = reward_group.mean(1), reward_group.std(1)
    advantage = (reward - mean.repeat_interleave(num_generation)) / (std.repeat_interleave(num_generation) + 1e-4)
    return advantage.unsqueeze(1)


@dataclass
class RolloutBatch:
    all_ids: np.ndarray
    all_mask: np.ndarray
    generation_ids: np.ndarray
    generation_mask: np.ndarray
    generation_text: List[str]
    old_log_probabilities: np.ndarray
    meta: List[Dict[str, Any]]


@ray.remote
class ReferenceModelActor:
    def __init__(self, model_path, num_gpus=1):
        self.device = torch.device("cuda")
        self.model, _ = load_model(model_path, device_map="auto")
        self.model.eval().to(self.device)

    def set_lora_weights(self, state_dict):
        load_lora_state_dict(self.model, state_dict)

    def log_probability(self, prompt_token_ids, completion_token_ids):
        with torch.no_grad():
            ids_tensor = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device)
            log_probs = torch.log_softmax(self.model(ids_tensor).logits[:, -1, :], -1)
            token_tensor = torch.tensor(completion_token_ids, dtype=torch.long, device=self.device)
            index = torch.arange(len(completion_token_ids), device=self.device)
            return log_probs[index, token_tensor].cpu().numpy()


@ray.remote
class RolloutWorker:
    def __init__(self, model_path, num_gpus=1, generation_config=None):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model, self.tokenizer = load_model(model_path, device_map="auto")
        self.model.eval().to(self.device)
        self.generation_config = generation_config or GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    def set_lora_weights(self, state_dict):
        load_lora_state_dict(self.model, state_dict)

    def repeat_tensor(self, tensor, repeat_num):
        return tensor.repeat_interleave(repeat_num, 0)

    def generate_rollout_data(self, batch, num_generation=4):
        logging.info(f"in RolloutWorker, start to generate rollout data")
        prompt_texts = batch["input_text"]
        encoded = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.device)
        input_token_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
        repeated_token_ids, repeated_attention_mask = self.repeat_tensor(input_token_ids, num_generation), self.repeat_tensor(
            attention_mask, num_generation
        )
        logging.info(f"in RolloutWorker, success to load repeated_token_ids, repeated_attention_mask")

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=repeated_token_ids,
                attention_mask=repeated_attention_mask,
                generation_config=self.generation_config,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        logging.info(f"in RolloutWorker, success to load generated")
        generation_length = generated.size(1) - repeated_token_ids.size(1)
        completion_token_ids = generated[:, -generation_length:]
        completion_mask = torch.ones_like(completion_token_ids, dtype=torch.int8)
        old_log_probabilities = (
            compute_log_probabilities(
                self.model,
                torch.cat([repeated_token_ids, completion_token_ids], 1),
                torch.cat([repeated_attention_mask, completion_mask], 1),
                generation_length,
            )
            .detach()
            .cpu()
            .numpy()
        )
        logging.info(f"in RolloutWorker, success to load old_log_probabilities")
        meta_list = []
        for question_options, answer in zip(batch["question_and_options"], batch["answer"]):
            meta_list.extend([{"question_and_options": question_options, "answer": answer}] * num_generation)
        generation_text = self.tokenizer.batch_decode(completion_token_ids.cpu().numpy(), skip_special_tokens=True)
        return RolloutBatch(
            all_ids=repeated_token_ids.cpu().numpy(),
            all_mask=repeated_attention_mask.cpu().numpy(),
            generation_ids=completion_token_ids.cpu().numpy(),
            generation_mask=completion_mask.cpu().numpy(),
            generation_text=generation_text,
            old_log_probabilities=old_log_probabilities,
            meta=meta_list,
        )


@ray.remote
class RewardActor:
    """Combines accuracy and formatting rewards."""

    def score(self, metas: List[Dict[str, Any]], completions: List[str]) -> List[float]:
        logging.info(f"in RewardActor, start to score")
        question_and_options = [meta["question_and_options"] for meta in metas]
        answers = [meta["answer"] for meta in metas]
        accuracy_scores = llm_eval_accuracy(question_and_options, completions, answers)  # 0 or 3
        format_scores = format_reward(completions)  # 0‑?  in steps of 0.2 / 0.4
        return [accuracy_score + format_score for accuracy_score, format_score in zip(accuracy_scores, format_scores)]


@ray.remote
class PolicyLearner:
    def __init__(self, reference_model_actor, model_path, num_gpus=1, kl_coefficient=0.2, learning_rate=5e-4, epsilon=0.2):
        self.device = torch.device("cuda")
        self.model, _ = load_model(model_path, device_map="auto")
        self.model.to(self.device)
        self.optimizer = optimizer.Adam((param for param in self.model.parameters() if param.requires_grad), lr=learning_rate)
        self.reference_model_actor = reference_model_actor
        self.kl_coefficient = kl_coefficient
        self.epsilon = epsilon

    def get_lora_weights_cpu(self):
        return extract_lora_state_dict(self.model)

    def update(self, rollout_batch: RolloutBatch, rewards: List[float], num_generation: int):
        batch_size, completion_length = rollout_batch.generation_ids.shape
        all_ids = torch.tensor(rollout_batch.all_ids, dtype=torch.long, device=self.device)
        all_mask = torch.tensor(rollout_batch.all_mask, dtype=torch.long, device=self.device)
        generation_ids = torch.tensor(rollout_batch.generation_ids, dtype=torch.long, device=self.device)
        generation_mask = torch.tensor(rollout_batch.generation_mask, dtype=torch.long, device=self.device)
        old_log_probabilities = torch.tensor(rollout_batch.old_log_probabilities, dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        current_log_probabilities = compute_log_probabilities(
            self.model,
            torch.cat([all_ids, generation_ids], 1),
            torch.cat([all_mask, generation_mask], 1),
            completion_length,
        )
        reference_log_probabilities = torch.tensor(
            ray.get(
                self.reference_model_actor.log_probability.remote(all_ids.cpu().numpy(), generation_ids[:, -1].cpu().numpy())
            ),
            dtype=torch.float32,
            device=self.device,
        )
        ratio = torch.exp(current_log_probabilities - old_log_probabilities)
        # FIXME num_generation 可以修改成从 rollout_batch 中获取
        advantage = compute_group_relative_advantages(reward_tensor, num_generation).to(self.device)
        surrogate_objective_1 = ratio * advantage
        surrogate_objective_2 = torch.clamp(ratio, 1 - self.epsilon, 1 + 1.5 * self.epsilon) * advantage
        # FIXME : 这里需要计算kl散度
        # kl_divergence = current_log_probabilities - reference_log_probabilities
        kl_divergence = torch.zeros_like(surrogate_objective_1)
        # pdb.set_trace()
        loss = (
            -((torch.min(surrogate_objective_1, surrogate_objective_2) - self.kl_coefficient * kl_divergence) * generation_mask)
            .sum(1)
            .mean()
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item()), float(reward_tensor.mean().item())


def main():
    ray.init(ignore_reinit_error=True, log_to_driver=True)
    CONFIG = {
        "model_path": "/data/xiaobei/Common_LLM_Base/Qwen2.5-1.5B-Instruct",
        # "model_path": "/data/xiaobei/Common_LLM_Base/Qwen2.5-7B-Instruct",
        "number_of_workers": 1,
        "number_of_generations": 2,
        "training_steps": 1000,
        "training_size": 16,  # dataset size
        "evaluation_size": 16,  # dataset size
        "testing_size": 16,  # dataset size
        "training_batch_size": 1,  # for all workers, training batch size = number of workers * each worker's batch size
        "kl_coefficient": 0.2,
        "gpu_config": {"reference_model_gpus": 1, "rollout_worker_gpus": 1, "policy_learner_gpus": 1},
    }

    train_dataset, eval_dataset, test_dataset = prepare_dataset_medqa(
        "train", CONFIG["training_size"], CONFIG["evaluation_size"], CONFIG["testing_size"]
    )
    train_dataloader, _, _ = prepare_dataloader_medqa(
        train_dataset, eval_dataset, test_dataset, CONFIG["training_batch_size"], CONFIG["training_batch_size"], 1
    )
    logging.info(f"success to load train_dataloader: {len(train_dataloader)}")

    iterator = cycle(train_dataloader)

    reference_model_actor = ReferenceModelActor.options(num_gpus=CONFIG["gpu_config"]["reference_model_gpus"]).remote(
        CONFIG["model_path"]
    )
    rollout_workers = [
        RolloutWorker.options(num_gpus=CONFIG["gpu_config"]["rollout_worker_gpus"]).remote(CONFIG["model_path"])
        for _ in range(CONFIG["number_of_workers"])
    ]
    reward_actor = RewardActor.remote()
    policy_learner = PolicyLearner.options(num_gpus=CONFIG["gpu_config"]["policy_learner_gpus"]).remote(
        reference_model_actor, CONFIG["model_path"], kl_coefficient=CONFIG["kl_coefficient"]
    )
    logging.info(f"success to load reference_model_actor, rollout_workers, reward_actor, policy_learner")

    try:
        for step in range(CONFIG["training_steps"]):
            logging.info(f"[step {step:04d}]")
            lora_state_dict = ray.get(policy_learner.get_lora_weights_cpu.remote())
            ray.get([worker.set_lora_weights.remote(lora_state_dict) for worker in rollout_workers])

            batch = next(iterator)

            chunk_length = len(batch["id"]) // CONFIG["number_of_workers"]
            chunks = [
                {key: value[i : i + chunk_length] for key, value in batch.items()}
                for i in range(0, len(batch["id"]), chunk_length)
            ]
            logging.info(f"success to load chunks: {len(chunks)}")

            rollout_batches = ray.get(
                [
                    worker.generate_rollout_data.remote(chunk, CONFIG["number_of_generations"])
                    for worker, chunk in zip(rollout_workers, chunks)
                ]
            )
            logging.info(f"success to load rollout_batches: {len(rollout_batches)}")
            # pdb.set_trace()

            completions, meta_list = [], []
            # FIXME 如果是多个 rollout_batches 的话，需要修改
            completions = rollout_batches[0].generation_text
            meta_list = rollout_batches[0].meta

            reward_values = ray.get(reward_actor.score.remote(meta_list, completions))
            logging.info(f"success to load reward_values: {len(reward_values)}")

            loss, average_reward = ray.get(
                policy_learner.update.remote(rollout_batches[0], reward_values, CONFIG["number_of_generations"])
            )
            logging.info(f"success to load loss: {loss}")

            logging.info(f"[step {step:04d}] loss={loss:.4f} | avgR={average_reward:.2f}")
            if step % 10 == 9:
                reference_model_actor.set_lora_weights.remote(lora_state_dict)

    except KeyboardInterrupt:
        logging.warning("Interrupted.")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
