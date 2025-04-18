from datasets import load_dataset

from src.data.prompt import SYSTEM_PROMPT_TOOLS as SYSTEM_PROMPT
from src.data.prompt import build_prompt, build_system_tools

from datasets import load_dataset, Dataset


def prepare_dataset(split="train", name="gsm8k", eval_size=10):
    if name == "gsm8k":
        return prepare_dataset_gsm8k(split, eval_size)
    elif name == "medmcqa":
        return prepare_dataset_medmcqa(split, eval_size)
    elif name == "medqa":
        return prepare_dataset_medqa(split, eval_size)
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
    # med_qa_zh_4options_bigbio_qa_train 这个 subset
    data = load_dataset("fzkuji/MedQA", "med_qa_zh_4options_bigbio_qa")[split]

    formatted_data = []

    for idx, example in enumerate(data):
        question = example["question"]
        choices = example["choices"]
        answer = example["answer"][0]

        # 将选项拼接成 A [0] B [1] ... 的格式
        options_text = ""
        for j, choice in enumerate(choices):  # 使用j而不是i作为循环变量，避免覆盖外层循环的i
            option_letter = chr(65 + j)  # 65 是 ASCII 中 'A' 的编码
            options_text += f"{option_letter}. {choice}\n"

        prompt_str = "\n".join(
            [
                build_system_tools(SYSTEM_PROMPT).strip(),
                f"""Question: {question}f
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
    train_data = formatted_data[eval_size:]  # fixme here

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    # for i, item in enumerate(eval_dataset):
    #     print(item["id"])

    return train_dataset, eval_dataset


if __name__ == "__main__":
    data = prepare_dataset_medqa(split="train")
