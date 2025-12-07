import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/data/xiaobei/Common_LLM_Base/Qwen2.5-3B-Instruct/Qwen/Qwen2___5-3B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# ======= Prompt ========
prompt = """
<system instruction>
When the user asks a question, the assistant should actively solve it. The assistant may think, search, reflect, and then produce a final answer. Use the following structured tags to organize reasoning and search steps. Precise formatting matters — follow the rules below.

Tags (semantic roles)
1. <reasoning> ... </reasoning>
   - Use to record the assistant's internal reasoning, step-by-step analysis, or intermediate thoughts that explain how the assistant reached a conclusion.
2. <search> ... </search>
   - Use when the assistant must perform external or uncertain information retrieval.
   - The content must follow this exact format:
     "<search> [Wiki_RAG]: keyword_1 keyword_2 ... </search>"
   - After sending the search tag, the system/tool will return results wrapped in an "<observation> ... </observation>" block.
3. <backtrack> ... </backtrack>
   - Use when previous reasoning or conclusions need correction or revision. Explain what changed and why.
4. <summary> ... </summary>
   - Use to give concise recaps of prior content or conclusions.
5. <answer> ... </answer>
   - Provide the final answer to the user's question.
   - This tag must appear exactly once and must be placed at the very end of the response.

Strict rules and formatting constraints
1. Only the <answer> tag is required to appear exactly once — and it must appear only at the end of the assistant's response.
2. All other tags (<reasoning>, <search>, <backtrack>, <summary>) may appear multiple times in any order, as needed.
3. Maintain exact tag spelling and angle-bracket punctuation. Tags are case-sensitive.
4. When using <search>, adhere to the required syntax: use the literal prefix "[Wiki_RAG]" followed by space-separated keywords. Do not deviate from this format.
5. If a <search> tag is used, expect a follow-up "<observation> ... </observation>" from the system and incorporate that observation into subsequent reasoning or the final answer.
6. Keep reasoning clear and focused — long internal chains of thought may be split across multiple <reasoning> blocks if appropriate.

Tool availability
You have the following tool(s) available to assist your search work:
{tool_descs}

Behavioral guidance
- Be concise, truthful, and helpful.
- When you backtrack, explicitly state what you changed and why.
- The final <answer> should be a clear, stand-alone response that a user could read without needing to see the intermediate tags (though including a brief summary of the reasoning is allowed if it helps clarity).
- Avoid leaking internal-only control signals or non-human-readable tokens outside the structured tags.

</system instruction>

<query>
Question: 下列对腺病毒生物学性状的描述中，正确的是（  ）。
Options:
A. 双股DNA(dsDNA)无包膜病毒
B. dsRNA无包膜病毒
C. 单股负链RNA(-ssRNA)无包膜病毒
D. -ssRNA有包膜病毒
E. dsDNA有包膜病毒
</query>

<reasoning>
首先，我需要确定腺病毒的生物学性状。腺病毒属于DNA病毒，所以选项B和C、D可能不正确，因为它们涉及RNA。然后，腺病毒是否有包膜呢？我记得腺病毒是无包膜的，所以选项A和E中的E有包膜可能错误，而A是dsDNA无包膜。所以正确选项可能是A。
</reasoning>

<search>
"""

enc = tokenizer([prompt], return_tensors="pt", padding=True)
input_ids = enc.input_ids.to(device)
attention_mask = enc.attention_mask.to(device)

num_samples = 3  # 你希望不一样的数量

outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,
    do_sample=True,         # 必须采样
    temperature=1.5,        # 高随机性！（关键）
    top_p=0.95,             # 扩大采样空间
    top_k=0,                # 取消 top-k 限制，可随机到更稀有的 token
    repetition_penalty=1.25,# 强惩罚重复
    no_repeat_ngram_size=6, # 强去重（提升差异度）
    num_return_sequences=num_samples,
    typical_p=0.8,          # 高频 token 不再主导
)

for i, out in enumerate(outputs):
    print(f"\n======== Sample {i+1} ========")
    print(tokenizer.decode(out, skip_special_tokens=True))
