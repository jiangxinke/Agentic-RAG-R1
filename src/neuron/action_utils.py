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

    example_output = """
    用户提出一个问题，助手来解决。助手首先在脑海中思考推理过程，然后向用户提供最终答案。
推理过程和最终答案的输出格式分别使用 <think> </think> 和 <answer> </answer> 标签包裹，
1. 也就是 "<think> 在这里写推理过程 </think>
2. <answer> 在这里写最终答案 </answer>"。
3. 在思考过程中，如果你认为上文的思考需要订正或修改，你可以使用 <backtrack> </backtrack> 标签包裹你的反思结果；
4. 在思考过程中，如果你认为你需要对上文做一些总结，你可以使用 <summary> </summary> 标签包裹你的思考结果；
5. 在思考过程中，**如果有必要，助手可以进行搜索** 以查找不确定的知识，格式为
"<search> 搜索查询,（需先提出你想要使用的工具，然后列出检索的关键字，如 "[Web_RAG]: keyword_1 keyword_2 ..."）</search>"。
**一次搜索查询仅能包含一个三元组**。然后搜索系统会用
"<observation> ...搜索结果... </observation>" 的格式向助手提供检索到的信息。

注意，所有动作均可执行多次。

你有以下工具可以使用:
Web_RAG: 使用 医学知识检索模块 这个API交互. 那么这个 医学知识检索模块 API 怎么使用呢? 这是通过搜索引擎检索医学知识，请结合检索的到的部分知识来辅助你回答。 参数: [{'name': 'input', 'description': '用户询问的字符串形式的问句', 'required': True, 'schema': {'type': 'string'}}] 格式需要是JSON对象.

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
Question: 女性，37岁，因短肠综合征入院。入院后经颈内静脉插管行全胃肠外营养支持。3周后突然出现寒战、高热，无咳嗽、咳痰。体检，腹部无压痛和反跳痛。1．最有可能的诊断是（　　）。f
            Options:
            A. 导管性脓毒症
B. 气胸
C. 高渗性非酮性昏迷
D. 导管折断
<reasoning>
女性患者因短肠综合征入院，并且进行了全胃肠外营养支持治疗，3周后出现了寒战、高热的症状。根据症状和治疗方法，首先应该考虑导管相关的问题。全胃肠外营养支持通常通过中心静脉插管来进行，所以导管性并发症是需要考虑的主要问题之一。接下来需要进一步分析选项，判断哪一种可能性更大。
</reasoning>
<search>
{
  "input": "37岁女性患者 导管性脓毒症 全胃肠外营养支持 寒战 高热"
}
</search>
"""
#     example_output = """
#     How to find the capital of France?
# <think> I need to search for the capital of France. </think>
# <search> capital France </search>
# <observation> The capital of France is Paris. </observation>
# <answer> The capital of France is Paris. </answer>"""
    output_tokens = tokenizer.encode(example_output, return_tensors="pt")[0]
    
    result = get_last_two_action_span(output_tokens, tokenizer)
    print(result)