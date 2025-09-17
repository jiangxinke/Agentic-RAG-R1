import re
import torch
from transformers import AutoTokenizer

def parse_actions(output_tokens: torch.Tensor, tokenizer: AutoTokenizer):
    """
    Parse the model output tokens to extract actions locations enclosed in specific tags.

    This function decodes the output tokens into a string and uses regular expressions
    to find all occurrences of actions enclosed within <action> and </action> tags.

    Args:
        output_tokens (torch.Tensor): The tensor containing output token IDs from the model.
        tokenizer (Any): The tokenizer used to decode the token IDs into a string.

    Returns:
        List[str]: A list of action strings extracted from the output.
    """
    # Step 1: Decode token sequence to string
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    
    # Step 2: Find all action spans
    # Action pattern:  <action>...</action>
    pattern = re.compile(r'<(\w+?)>(.*?)</\1>', re.DOTALL)
    actions = []
    for match in pattern.finditer(output_text):
        action_name = match.group(1)
        tag_span = (match.start(1) - 1, match.end(1) + 1)  # Tag start/end char idx, include <>
        
        raw_action_args = match.group(2)
        # Find the start/end of the stripped content within the original match
        arg_span = (match.start(2), match.end(2))  # Content start/end char idx
        # Adjust to the stripped version
        leading_spaces = len(raw_action_args) - len(raw_action_args.lstrip())
        trailing_spaces = len(raw_action_args) - len(raw_action_args.rstrip())
        arg_span = (arg_span[0] + leading_spaces, arg_span[1] - trailing_spaces)

        actions.append((action_name, tag_span, arg_span))
        # print(f"Found action '{action_name}' with tag chars [{tag_span[0]}, {tag_span[1]}) and arg chars [{arg_span[0]}, {arg_span[1]})")
        
    # Step 3: For each token, get its corresponding (start,end) char position in output_text
    tokens = output_tokens.tolist()
    running = 0
    token_spans = []
    for token_id in tokens:
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        # Some tokenizers prepend spaces inconsistently, so .strip() might corrupt alignment
        # Try to find the next occurrence from current position
        if len(token_str) == 0:
            token_spans.append((running, running))
            continue
        idx = output_text.find(token_str, running)
        if idx == -1:
            # fallback (should almost never happen)
            idx = running
        token_spans.append((idx, idx+len(token_str)))
        # print(f"Token '{token_str}' at chars [{idx}, {idx+len(token_str)})")
        running = idx + len(token_str)
    
    # Step 4: For each action's arg span, find token idxs that cover it
    action_token_spans = {'tag': {}, 'arg': {}}
    for act_name, (tag_start, tag_end), (arg_start, arg_end) in actions:
        # Find tokens whose span overlaps with [arg_start, arg_end)
        token_idxs = [i for i, (tok_start, tok_end) in enumerate(token_spans)
                      if (tok_end > tag_start and tok_start <= tag_end - 1)]
        if token_idxs:
            action_token_spans["tag"].setdefault(act_name, []).append((token_idxs[0], token_idxs[-1]+1))  # python-style [start, end)
        
        # Find tokens whose span overlaps with [arg_start, arg_end)
        token_idxs = [i for i, (tok_start, tok_end) in enumerate(token_spans)
                      if (tok_end > arg_start and tok_start <= arg_end - 1)]
        if token_idxs:
            action_token_spans["arg"].setdefault(act_name, []).append((token_idxs[0], token_idxs[-1]+1))  # python-style [start, end)
    
    # For example, output: {'reasoning': [(2, 15)], 'search': [(18, 27)], ...}
    return action_token_spans


if __name__ == "__main__":

    # Example usage
    model_path = "/data/xiaobei/Common_LLM_Base/Qwen2.5-3B-Instruct/Qwen/Qwen2___5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    example_output = """<think> I need to search for the capital of France. </think>
<search> capital France </search>
<observation> The capital of France is Paris. </observation>
<answer> The capital of France is Paris. </answer>"""
    output_tokens = tokenizer.encode(example_output, return_tensors="pt")[0]
    actions = parse_actions(output_tokens, tokenizer)
    print("Extracted actions tags:", actions['tag'])
    print("Extracted actions args:", actions['arg'])
    print()

    token_strs = [tokenizer.decode([tok], skip_special_tokens=True) for tok in output_tokens]
    print("Output tokens: ", token_strs)
    print()

    for act, spans in actions['tag'].items():
        for start, end in spans:
            print(f"Action '{act}' tag tokens: ", tokenizer.decode(output_tokens[start:end], skip_special_tokens=True))
    for act, spans in actions['arg'].items():
        for start, end in spans:
            print(f"Action '{act}' args tokens: ", tokenizer.decode(output_tokens[start:end], skip_special_tokens=True))