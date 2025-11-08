import re
import torch
from transformers import AutoTokenizer

def compute_token_spans_by_offsets(output_tokens, tokenizer):
    # 与方案A一致：统一关闭清理，获取稳定的整段文本
    text = tokenizer.decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # 为确保与 output_tokens 语义一致，这里不添加特殊符号
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )

    # offsets: List[Tuple[start_char, end_char]]，与 encoded.input_ids 对齐
    offsets = encoded["offset_mapping"]
    spans = [(s, e) for (s, e) in offsets]
    return text, spans


def locate_action_token_spans_fast(output_tokens, tokenizer):
    output_text, token_spans = compute_token_spans_by_offsets(output_tokens, tokenizer)

    pattern = re.compile(r'<(reasoning|search|observation|summary|backtrack|answer)>(.*?)</\1>', re.DOTALL)
    matches = list(pattern.finditer(output_text))

    # 仍以“最后一次出现”为准
    actions = {}
    for m in matches:
        name = m.group(1)
        raw = m.group(2)
        start, end = m.start(2), m.end(2)
        stripped_lead = len(raw) - len(raw.lstrip())
        stripped_tail = len(raw) - len(raw.rstrip())
        arg_span = (start + stripped_lead, end - stripped_tail)
        tag_span = (m.start(0), m.end(0))
        actions[name] = {"tag_span": tag_span, "arg_span": arg_span}

    action_token_spans = {"tag": {}, "arg": {}}
    for name, spans_dict in actions.items():
        ts, te = spans_dict["tag_span"]
        as_, ae = spans_dict["arg_span"]

        tag_token_idxs = [i for i, (s, e) in enumerate(token_spans) if e > ts and s <= te - 1]
        arg_token_idxs = [i for i, (s, e) in enumerate(token_spans) if e > as_ and s <= ae - 1]

        if tag_token_idxs:
            action_token_spans["tag"].setdefault(name, []).append((tag_token_idxs[0], tag_token_idxs[-1] + 1))
        if arg_token_idxs:
            action_token_spans["arg"].setdefault(name, []).append((arg_token_idxs[0], arg_token_idxs[-1] + 1))

    return action_token_spans


def flatten_any_level(nested_list):
    result = []
    for item in nested_list:
        # 仅对list类型递归铺平，保留元组、数字等其他类型
        if isinstance(item, list):
            result.extend(flatten_any_level(item))
        else:
            result.append(item)
    return result

def get_last_two_action_span(output_tokens: torch.Tensor, tokenizer: AutoTokenizer):
    num_tokens = len(output_tokens)
    
    # actions = locate_action_token_spans(output_tokens, tokenizer)
    actions = locate_action_token_spans_fast(output_tokens, tokenizer)
    print(actions)
    tag_spans_list = list(actions['tag'].values())
    spans_list = sorted(flatten_any_level(tag_spans_list))
    # print(spans_list)

    if len(spans_list) == 0:
        return (0, 0, 0)
    elif len(spans_list) == 1:
        return (0, 0, num_tokens)
    else:
        return (spans_list[-2][0], spans_list[-1][0] - 1, num_tokens)


if __name__ == "__main__":
    # Example usage
    model_path = "/data/xiaobei/Common_LLM_Base/Qwen2.5-3B-Instruct/Qwen/Qwen2___5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    example_output = """
    How to find the capital of France?
<think> I need to search for the capital of France. </think>
<search> capital France </search>
<observation> The capital of France is Paris. </observation>
<answer> The capital of France is Paris. </answer>"""
    output_tokens = tokenizer.encode(example_output, return_tensors="pt")[0]
    
    result = get_last_two_action_span(output_tokens, tokenizer)
    print(result)