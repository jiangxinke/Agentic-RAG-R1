from datasets import load_dataset
from src.utils.utils import load_config

config = load_config("src/config/config.yaml")
# Robustly read SSRL flag from config (support both use_SSRL and use_ssrl)
use_ssrl = getattr(config.training, "use_SSRL", getattr(config.training, "use_ssrl", False))
if use_ssrl:
    from src.data.prompt import SYSTEM_PROMPT_TOOLS_SSRL as SYSTEM_PROMPT
else:
    from src.data.prompt import SYSTEM_PROMPT_TOOLS_BACKTRACK as SYSTEM_PROMPT

from src.data.prompt import build_prompt, build_system_tools

from datasets import load_dataset, Dataset, DownloadMode
import os
import numpy as np


def prepare_dataset(split="train", name="gsm8k", eval_size=10):
    if name == "gsm8k":
        return prepare_dataset_gsm8k(split, eval_size)
    elif name == "medmcqa":
        return prepare_dataset_medmcqa(split, eval_size)
    elif name == "medqa":
        return prepare_dataset_medqa(split, eval_size)
    elif name == "asearcher":
        return prepare_dataset_asearcher(split, eval_size)
    elif name.startswith("nq_hotpotqa_train_multi_"):
        return prepare_dataset_nq_hotpotqa(split, name, eval_size)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def prepare_dataset_gsm8k(split="train", eval_size=10):
    """Load and prepare the GSM8K dataset for training with string prompts."""
    data = load_dataset("openai/gsm8k", "main")[split]
    formatted_data = []

    for example in data:
        # Convert list of messages to a single string prompt.
        prompt_str = build_prompt(
            [
                {"role": "system", "content": build_system_tools(SYSTEM_PROMPT)},
                {"role": "user", "content": example["question"]},
            ]
        )
        formatted_example = {
            "prompt": prompt_str,  # Now a string rather than a list.
            "answer": extract_answer_from_dataset(example["answer"]),
        }
        formatted_data.append(formatted_example)

    return formatted_data


def prepare_dataset_medmcqa(split="train"):
    # 加载医学数据集（以 medmcqa 为例）
    data = load_dataset("medmcqa", split=split)
    formatted_data = []

    for example in data:
        # 构造 prompt，假设 SYSTEM_PROMPT 是一个医学相关的提示
        question = f"""Question: {example["question"]}
            Options:
            A. {example["opa"]}
            B. {example["opb"]}
            C. {example["opc"]}
            D. {example["opd"]}"""

        prompt_str = "\n".join(
            [
                build_system_tools(SYSTEM_PROMPT).strip(),
                f"""Question: {example["question"]}
            Options:
            A. {example["opa"]}
            B. {example["opb"]}
            C. {example["opc"]}
            D. {example["opd"]}""",
            ]
        )
        # 提取正确答案（假设答案在 "correct_answer" 字段中）
        # 构造格式化数据
        correct_answer_index = example["cop"]
        options = [example["opa"], example["opb"], example["opc"], example["opd"]]
        correct_answer = options[correct_answer_index]

        formatted_example = {
            "prompt": prompt_str,
            "question": question,
            "answer": str(correct_answer),  # 将答案转换为字符串
        }
        formatted_data.append(formatted_example)

    return formatted_data


def prepare_dataset_medqa(split="train", eval_size=10):
    """Load and format MedQA (Chinese, 4 options) dataset.

    Adds robust error handling and clearer guidance if loading fails.
    """
    import logging, os

    try:
        # Prefer passing split directly to avoid unnecessary builder calls
        # Use a fresh local cache to avoid legacy dataset_info incompatibilities
        cache_dir = os.path.join(".hf_cache", "medqa")
        data = load_dataset(
            "fzkuji/MedQA",
            "med_qa_zh_4options_bigbio_qa",
            split=split,
            cache_dir=cache_dir,
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
        )
    except Exception as e:
        logging.error(
            "Failed to load dataset 'fzkuji/MedQA' with config 'med_qa_zh_4options_bigbio_qa' and split '%s': %s",
            split,
            e,
        )
        logging.info(
            "HF_ENDPOINT=%s. If empty, consider setting 'HF_ENDPOINT=https://hf-mirror.com' for faster access in China.",
            os.environ.get("HF_ENDPOINT", "<not set>"),
        )
        logging.warning("Falling back to 'medmcqa' dataset for training.")
        # 回退到 medmcqa，保持返回 (train_dataset, eval_dataset) 的格式
        fallback_data = prepare_dataset_medmcqa(split=split)
        eval_data = fallback_data[:eval_size]
        train_data = fallback_data[eval_size:]
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
        return train_dataset, eval_dataset

    formatted_data = []

    for idx, example in enumerate(data):
        question = example["question"]
        choices = example["choices"]
        answer = example["answer"][0]

        # 拼接选项文本 A/B/C/D
        options_text = ""
        for j, choice in enumerate(choices):
            option_letter = chr(65 + j)  # 'A' == 65
            options_text += f"{option_letter}. {choice}\n"

        prompt_str = "\n".join(
            [
                build_system_tools(SYSTEM_PROMPT).strip(),
                f"""Question: {question}
            Options:
            {options_text}""",
            ]
        )

        formatted_data.append(
            {
                "id": idx + 1,
                "prompt": prompt_str,
                "question": question + "\n" + options_text,
                "answer": str(answer),
            }
        )

    eval_data = formatted_data[:eval_size]
    train_data = formatted_data[eval_size:]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    return train_dataset, eval_dataset



def prepare_dataset_asearcher(split="train", eval_size=10):
    # Path
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "ASearcher")
    train_file = os.path.join(data_dir, "train.parquet")
    
    # Load
    # If split is 'train', we load train.parquet. 
    # For evaluation, we can split the train set or load one of the test sets.
    # Given the user said "Asearcher as a dataset... training set name is Asearcher", 
    # and usually we validate on a held-out set. 
    # I will simply split the train.parquet for now as is common in this repo's examples (medmcqa, gsm8k).
    
    dataset = load_dataset("parquet", data_files={"train": train_file})["train"]
    
    formatted_data = []
    for idx, example in enumerate(dataset):
        prompt_str = build_prompt([
            {"role": "system", "content": build_system_tools(SYSTEM_PROMPT)},
            {"role": "user", "content": example["question"]}
        ])
        formatted_data.append({
            "prompt": prompt_str,
            "answer": example["answer"],
            "question": example["question"]
        })
        
    # Split
    eval_data = formatted_data[:eval_size]
    train_data = formatted_data[eval_size:]
    
    return Dataset.from_list(train_data), Dataset.from_list(eval_data)


def prepare_dataset_nq_hotpotqa(split="train", name="nq_hotpotqa_train_multi_2", eval_size=10):
    # Construct path
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", name)
    train_file = os.path.join(data_dir, "train.parquet")
    test_file = os.path.join(data_dir, "test.parquet")
    
    # Load
    data_files = {"train": train_file}
    if os.path.exists(test_file):
        data_files["test"] = test_file
        
    dataset = load_dataset("parquet", data_files=data_files)
    
    def format_example(example):
        prompt_msgs = example["prompt"]
        if isinstance(prompt_msgs, np.ndarray):
            prompt_msgs = prompt_msgs.tolist()
            
        if isinstance(prompt_msgs, list) and len(prompt_msgs) > 0:
            prompt_content = prompt_msgs[0]['content']
        else:
            prompt_content = str(prompt_msgs)

        # Answer
        reward_model = example["reward_model"]
        ground_truth = reward_model.get('ground_truth', {})
        targets = ground_truth.get('target', [])
        
        final_answers = []
        # targets might be a numpy array of arrays if loaded via datasets/arrow
        if isinstance(targets, np.ndarray):
            targets = targets.tolist()
            
        for target_list in targets:
            # target_list might be an array or list
            if isinstance(target_list, np.ndarray):
                target_list = target_list.tolist()
                
            if len(target_list) > 0:
                final_answers.append(str(target_list[0]))
            else:
                final_answers.append("")
        
        answer_str = "; ".join(final_answers)
        
        return {
            "prompt": prompt_content,
            "answer": answer_str,
            "question": "" 
        }

    train_data = [format_example(ex) for ex in dataset["train"]]
    
    if "test" in dataset:
        eval_data = [format_example(ex) for ex in dataset["test"]]
        if eval_size > 0:
            eval_data = eval_data[:eval_size]
    else:
        eval_data = train_data[:eval_size]
        train_data = train_data[eval_size:]
        
    return Dataset.from_list(train_data), Dataset.from_list(eval_data)


if __name__ == "__main__":
    data = prepare_dataset_medqa(split="train")
