import torch
from transformers import AutoTokenizer

from src.neuron.parse_tokens import locate_action_token_spans

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
    
    actions = locate_action_token_spans(output_tokens, tokenizer)
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

    example_output = "<search> three four </search> <answer> fourth fifth </answer>"
#     example_output = """
#     How to find the capital of France?
# <think> I need to search for the capital of France. </think>
# <search> capital France </search>
# <observation> The capital of France is Paris. </observation>
# <answer> The capital of France is Paris. </answer>"""
    output_tokens = tokenizer.encode(example_output, return_tensors="pt")[0]
    
    result = get_last_two_action_span(output_tokens, tokenizer)
    print(result)