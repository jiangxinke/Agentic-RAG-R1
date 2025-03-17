SYSTEM_PROMPT = """
The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning
process in the mind and then provides the User with the final answer. The output format of reasoning
process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think><answer> final answer here </answer>". During the
thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with
the format of "<search> search query (only list keywords, such as "keyword_1 keyword_2
...")</search>". **A query must involve only a single triple**. Then, the search system will
provide the Assistant with the retrieval information with the format of "<observation> ...search
results... </observation>".

Respond in the following format:

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

def build_prompt(messages):
    """
    Build a single prompt string from a list of messages.
    Each message is expected to be a dictionary with 'role' and 'content' keys.
    This function concatenates all message contents, preserving the training format.
    """
    return "\n".join([msg["content"].strip() for msg in messages])
