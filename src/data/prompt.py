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
    return "\n".join([msg["content"].strip() for msg in messages])


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
