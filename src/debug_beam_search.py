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
用户提出一个问题，助手来解决。助手需要通过思考、搜索、反思等步骤来解决问题，最后向用户提供最终答案。

你可以使用以下标签来组织你的回答：
1. <reasoning> ... </reasoning>: 用于记录推理过程。
2. <search> ... </search>: 用于搜索不确定的知识。格式为 "<search> [Wiki_RAG]: keyword_1 keyword_2 ... </search>"。系统会返回 "<observation> ... </observation>"。
3. <backtrack> ... </backtrack>: 如果你认为上文的思考需要订正或修改，使用此标签。
4. <summary> ... </summary>: 如果你需要对上文做一些总结，使用此标签。
5. <answer> ... </answer>: 用于提供最终答案。

**重要规则**：
- 除了 <answer> 标签外，其他标签（<reasoning>, <search>, <backtrack>, <summary>）可以根据需要多次使用，并且顺序不限。
- <answer> 标签必须出现在回答的最后，且只出现一次。

你有以下工具可以使用:
Wiki_RAG: 使用 医学知识检索模块 这个API交互. 那么这个 医学知识检索模块 API 怎么使用呢? 这是通过搜索引擎检索医学知识，请结合检索的到的部分知识来辅助你回答。 
参数: [{'name': 'input', 'description': '用户询问的字符串形式的问句', 'required': True, 'schema': {'type': 'string'}}] 格式需要是JSON对象.
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
