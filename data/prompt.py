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
