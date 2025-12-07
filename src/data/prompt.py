# SYSTEM_PROMPT = """
# The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning
# process in the mind and then provides the User with the final answer. The output format of reasoning
# process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think><answer> final answer here </answer>". During the
# thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with
# the format of "<search> search query (only list keywords, such as "keyword_1 keyword_2
# ...")</search>". **A query must involve only a single triple**. Then, the search system will
# provide the Assistant with the retrieval information with the format of "<observation> ...search
# results... </observation>".

# Respond in the following format:

# <reasoning>
# ...
# </reasoning>
# <search>
# ...
# </search>
# <answer>
# ...
# </answer>
# """

SYSTEM_PROMPT = """
用户提出一个问题，助手来解决。助手首先在脑海中思考推理过程，然后向用户提供最终答案。
推理过程和最终答案的输出格式分别使用 <think> </think> 和 <answer> </answer> 标签包裹，
也就是 "<think> 在这里写推理过程 </think><answer> 在这里写最终答案 </answer>"。
在思考过程中，**如果有必要，助手可以进行搜索** 以查找不确定的知识，格式为
"<search> 搜索查询（只列出关键字，如 "keyword_1 keyword_2 ..."）</search>"。
**一次搜索查询仅能包含一个三元组**。然后搜索系统会用
"<observation> ...搜索结果... </observation>" 的格式向助手提供检索到的信息。

请按照以下格式作答：

<reasoning>
...
</reasoning>
<search>
...
</search>
<answer>
...
</answer>
"""

TOOL_DESC = """{name_for_model}: 使用 {name_for_human} 这个API交互. 那么这个 {name_for_human} API 怎么使用呢? {description_for_model} 参数: {parameters} 格式需要是JSON对象."""

# FIXME HERE
SYSTEM_PROMPT_TOOLS = """
用户提出一个问题，助手来解决。助手首先在脑海中思考推理过程，然后向用户提供最终答案。
推理过程和最终答案的输出格式分别使用 <think> </think> 和 <answer> </answer> 标签包裹，
也就是 "<think> 在这里写推理过程 </think><answer> 在这里写最终答案 </answer>"。
在思考过程中，**如果有必要，助手可以进行内心的搜索** 以查找不确定的知识，格式为
"<search> 搜索查询（只列出关键字，如 "keyword_1 keyword_2 ..."）</search>"。
**一次搜索查询仅能包含一个三元组**。然后搜索系统会用
"<observation> ...搜索结果... </observation>" 的格式向助手提供检索到的信息。
请注意！！！这里的<observation>检索结果你是**要自己生成**的，你需要用给定的工具来自己模拟生成结果，外界工具是无效的，是无法帮助你的！！！

请按照以下格式作答：

<reasoning>
...
</reasoning>
<search>
...
</search>
<answer>
...
</answer>
"""

# SYSTEM_PROMPT_TOOLS = """
# 请仔细思考问题并给出答案。
# 你的回答需要放在以下标签中:
# <answer>
# 在这里详细阐述你的答案
# </answer>

# 请确保答案清晰、准确且完整。
# """

SYSTEM_PROMPT_TOOLS_BACKTRACK_EN = """
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

"""



SYSTEM_PROMPT_TOOLS_BACKTRACK = """
用户提出一个问题，助手来解决。助手首先在脑海中思考推理过程，然后向用户提供最终答案。
推理过程和最终答案的输出格式分别使用 <think> </think> 和 <answer> </answer> 标签包裹，
1. 也就是 "<think> 在这里写推理过程 </think>
2. <answer> 在这里写最终答案 </answer>"。
3. 在思考过程中，如果你认为上文的思考需要订正或修改，你可以使用 <backtrack> </backtrack> 标签包裹你的反思结果；
4. 在思考过程中，如果你认为你需要对上文做一些总结，你可以使用 <summary> </summary> 标签包裹你的思考结果；
5. 在思考过程中，**如果有必要，助手可以进行搜索** 以查找不确定的知识，格式为
"<search> 搜索查询,（需先提出你想要使用的工具，然后列出检索的关键字，如 "[{tool_names}]: keyword_1 keyword_2 ..."）</search>"。
**一次搜索查询仅能包含一个三元组**。然后搜索系统会用
"<observation> ...搜索结果... </observation>" 的格式向助手提供检索到的信息。

注意，所有动作均可执行多次。

你有以下工具可以使用:
{tool_descs}

请按照以下格式作答：

<reasoning>
...
</reasoning>
<search>
...
</search>
<summary>
...
</summary>
<backtrack>
...
</backtrack>
<answer>
...
</answer>
"""

SYSTEM_PROMPT_SSRL = """
用户提出一个问题，助手来解决。助手首先在脑海中思考推理过程，然后向用户提供最终答案。
推理过程和最终答案的输出格式分别使用 <think> </think> 和 <answer> </answer> 标签包裹，
也就是 "<think> 在这里写推理过程 </think><answer> 在这里写最终答案 </answer>"。
在思考过程中，**如果有必要，助手可以进行内心的搜索** 以查找不确定的知识，格式为
"<search> 搜索查询（只列出关键字，如 "keyword_1 keyword_2 ..."）</search>"。
**一次搜索查询仅能包含一个三元组**。然后搜索系统会用
"<observation> ...搜索结果... </observation>" 的格式向助手提供检索到的信息。
请注意！！！这里的<observation>检索结果你是**要自己生成**的，你需要用给定的工具来自己模拟生成结果，外界工具是无效的，是无法帮助你的！！！

请按照以下格式作答：

<reasoning>
...
</reasoning>
<search>
...
</search>
<answer>
...
</answer>
"""

SYSTEM_PROMPT_TOOLS_SSRL_EN = """
<system instruction>
When the user asks a question, the assistant should actively solve it. The assistant may think, search, reflect, and then provide a final answer. Use the following structured tags to organize reasoning, search, and the final answer.

Tags and usage:
1. <reasoning> ... </reasoning>
   - Record the assistant's internal reasoning, step-by-step analysis, or intermediate thoughts.
2. <search> ... </search>
   - Use when uncertain information needs to be retrieved.
   - The format must follow exactly:
     "<search> search query (list only key terms, e.g., 'keyword_1 keyword_2 ...') </search>"
   - **Important:** Any <observation> results must be generated by the assistant itself using internal knowledge — do **not** rely on external resources.
   - After <search>, produce:
     "<observation> ...simulated search results generated by the assistant... </observation>"
3. <backtrack> ... </backtrack>
   - Use when previous reasoning or conclusions need correction.
4. <summary> ... </summary>
   - Give concise summaries of previous reasoning or content.
5. <answer> ... </answer>
   - Provide the final answer. This tag must appear exactly once at the very end of the response.

Strict rules:
- Only <answer> is required to appear exactly once and at the end.
- All other tags (<reasoning>, <search>, <backtrack>, <summary>) may appear multiple times, in any order.
- Maintain exact tag spelling and punctuation. Tags are case-sensitive.
- In <search>, only include a single triple of key terms. Generate <observation> yourself; do not rely on external tools.
- Keep reasoning clear and focused. Multiple <reasoning> blocks can be used if needed.

Tool availability:
You may use any internal tools described in {tool_descs} to simulate search results.

Behavioral guidance:
- Be concise, truthful, and helpful.
- When backtracking, clearly state what changed and why.
- The final <answer> should be self-contained, understandable without intermediate tags, but may briefly reference reasoning if it improves clarity.

End of instruction.
</system instruction>

<query>
"""


SYSTEM_PROMPT_TOOLS_SSRL = """
用户提出一个问题，助手来解决。助手首先在脑海中思考推理过程，然后向用户提供最终答案。
推理过程和最终答案的输出格式分别使用 <think> </think> 和 <answer> </answer> 标签包裹，
1. 也就是 "<think> 在这里写推理过程 </think>
2. <answer> 在这里写最终答案 </answer>"。
3. 在思考过程中，如果你认为上文的思考需要订正或修改，你可以使用 <backtrack> </backtrack> 标签包裹你的反思结果；
4. 在思考过程中，如果你认为你需要对上文做一些总结，你可以使用 <summary> </summary> 标签包裹你的思考结果；
5. 在思考过程中，**如果有必要，助手可以进行内心的搜索** 以查找不确定的知识，格式为
"<search> 搜索查询,（需先提出你想要使用的工具，然后列出检索的关键字，如 "[{tool_names}]: keyword_1 keyword_2 ..."）</search>"。
**一次搜索查询仅能包含一个三元组**。然后搜索系统会用
"<observation> ...搜索结果... </observation>" 的格式向助手提供检索到的信息。
请注意！！！这里的<observation>检索结果你是**要自己生成**的，你需要用给定的工具来自己模拟生成结果，外界工具是无效的，是无法帮助你的！！！

注意，所有动作均可执行多次。

你有以下工具可以使用:
{tool_descs}

请按照以下格式作答：

<reasoning>
...
</reasoning>
<search>
...
</search>
<summary>
...
</summary>
<backtrack>
...
</backtrack>
<answer>
...
</answer>
"""



# SYSTEM_PROMPT = """
# 你是一个智能助手，用户提出问题后，你需要先在脑海里进行思考，但只将简要的推理过程放在 <reasoning>...</reasoning> 中输出。
# 若在推理中发现需要外部信息，则先输出一次 <search>...</search>，
# 此时系统会给你返回 <observation>...</observation> 作为搜索结果。
# 然后你结合搜索结果继续思考，最后给出 <answer>...</answer> 形式的最终简洁回答。

# 以下为作答所需遵循的格式说明：
# 1. 如果需要展示推理过程，请使用：
# <reasoning>
# 在这里写下你的思考推理
# </reasoning>

# 2. 如果需要检索外部信息，请使用：
# <search>
# (仅包含一个三元组搜索关键字，比如 "词1 词2 词3")
# </search>

# 3. 系统返回搜索结果时，会使用：
# <observation>
# 这是搜索系统给出的内容
# </observation>

# 4. 最终回答请使用：
# <answer>
# 在这里写出对用户问题的最终回答
# </answer>

# 请务必按照上述标签和顺序进行作答：
# - **先**给出 `<reasoning>` 表达简要推理或思考；
# - 如果确实需要外部信息，请紧接着输出 `<search>` 标签；
# - 获取到 `<observation>` 后，再依据搜索结果做进一步推理，最终用 `<answer>` 标签输出明确答案。
# """

# SYSTEM_PROMPT = """
# 用户提出问题后，助手在解决过程中始终需要借助外部信息。首先，助手在脑海中进行初步思考，并确定需要检索哪些信息；接着输出一个搜索请求，格式为
# <search>
# 搜索查询（仅包含一个三元组关键字，例如 "关键字1 关键字2 关键字3"）
# </search>
# 系统随后会返回搜索结果，格式为
# <observation>
# ...搜索结果...
# </observation>
# 收到搜索结果后，助手继续结合自身推理，最终给出答案，格式为
# <answer>
# 最终答案内容
# </answer>
# 整个过程中，助手可以使用 <reasoning> 标签展示自己的思考过程，格式如下：
# <reasoning>
# ...初步思考内容...
# </reasoning>
# <search>
# 关键字1 关键字2 关键字3
# </search>
# （系统返回：<observation>...搜索结果...</observation>）
# <reasoning>
# ...基于搜索结果的补充思考...
# </reasoning>
# <answer>
# ...最终答案...
# </answer>

# 请务必遵循以下步骤：
# 1. 用户提问后，首先在 <reasoning> 标签中给出初步思考，并明确需要检索哪些外部信息。
# 2. 输出 <search> 标签，内容仅包含一个三元组搜索关键字。
# 3. 等待系统返回 <observation> 标签中的搜索结果。
# 4. 根据搜索结果补充推理，继续在 <reasoning> 标签中说明。
# 5. 最后在 <answer> 标签中给出最终答案。
# """

LLM_EVAL_PROMPT = """
你是一名严格、但能识别同义表达的阅卷老师。请阅读以下信息并判断学生的选择题作答是否正确：

1. 【题目】：
{question}

2. 【正确答案】：
{expected}

3. 【学生的作答】：
{predicted}

你的任务是：
- 首先判断学生的作答是否与正确答案一致（如果含义相同也视为一致）；
- 如果学生作答正确，请只输出：Yes
- 如果学生作答错误，请只输出：No

**重要要求**：
- 不要输出引号、标点、换行、额外文字、空格或其他任何字符。
- 只输出一个单词：Yes 或 No。
"""


def build_prompt(messages):
    """
    Build a single prompt string from a list of messages.
    Each message is expected to be a dictionary with 'role' and 'content' keys.
    This function concatenates all message contents, preserving the training format.
    """
    return "\n".join([msg["content"].strip() for msg in messages]) + "</query>"


from src.utils.Tools import Tools


def build_system_tools(sys_prompt=SYSTEM_PROMPT_TOOLS):

    tool = Tools()
    tool_descs, tool_names = [], []

    for tool in tool.toolConfig:
        tool_descs.append(TOOL_DESC.format(**tool))
        tool_names.append(tool["name_for_model"])

    tool_descs = "\n\n".join(tool_descs)
    tool_names = ",".join(tool_names)
    sys_prompt_tools = sys_prompt.format(tool_descs=tool_descs, tool_names=tool_names)

    return sys_prompt_tools
