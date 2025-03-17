from transformers import PreTrainedModel, StoppingCriteria, StoppingCriteriaList
import torch
from torch.nn.utils.rnn import pad_sequence
import re

# 统一采用新代码中的搜索器（例如 EnglishWebSearcher）
from web_search import * 

class ThinkTagStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, think_token="</search>"):
        self.tokenizer = tokenizer
        self.think_token = think_token
        self.think_token_id = tokenizer.encode(think_token, add_special_tokens=False)[0]

    def __call__(self, input_ids, scores):
        last_token = input_ids[0][-1]
        if last_token == self.think_token_id:
            return True
        return False

class CustomModel(PreTrainedModel):
    def __init__(self, model, tokenizer, searcher=None):
        """
        构造函数增加 searcher 参数，默认为 EnglishWebSearcher。
        """
        super().__init__(model.config)
        self.model = model
        self.tokenizer = tokenizer
        # self.searcher = searcher if searcher is not None else EnglishWebSearcher()

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def generate_with_think_interruption(
        self,
        prompt_ids,
        attention_mask=None,
        max_new_tokens=None,
        do_sample=True,
        temperature=1.0,
        pad_token_id=None,
        eos_token_id=None,
        max_iterations=3,
    ):
        """
        多轮生成过程：
          1. 初始输入 prompt_ids 与 attention_mask（若未提供，则全部为有效 token）。
          2. 每轮调用 model.generate 时，传入当前的 input_ids 和 attention_mask。
          3. 生成的输出剔除 prompt 部分后进行解码，得到 new_text。
          4. 如果 new_text 中检测到 "</search>" 且当前迭代未达到 max_iterations - 1，
             同时注意未出现 eos token（通过将 eos_token_id 解码为字符串进行检查），
             则提取 "<search>...</search>" 之间的查询，调用 searcher 进行查询，
             将返回的搜索结果以 "<search result>...</search result>" 拼接到 new_text 后，
             然后将历史 token 与新文本重新编码后拼接，构成新的输入。
          5. 更新后的文本重新通过 tokenizer 生成 input_ids 和 attention_mask（均使用 left padding），
             用于下一轮生成。
          6. 如果未触发检索，则认为当前样本生成完成，将该样本输出保存。
          7. 最终使用 pad_sequence 对所有输出进行对齐，并调用 _postprocess_responses 进行文本后处理。
        """
        stopping_criteria = StoppingCriteriaList([ThinkTagStoppingCriteria(self.tokenizer)])
        batch_size = prompt_ids.size(0)
        device = prompt_ids.device

        # 初始化 current_input_ids 与 attention_mask
        current_input_ids = prompt_ids.clone()
        current_attention_mask = (attention_mask.clone() if attention_mask is not None 
                                  else torch.ones_like(prompt_ids))
        
        all_outputs = [None] * batch_size
        should_generate = torch.ones(batch_size, dtype=torch.bool, device=device)

        # 如果 eos_token_id 不为空，将其解码为字符串用于后续判断
        eos_token_str = (self.tokenizer.decode([eos_token_id], skip_special_tokens=False)
                         if eos_token_id is not None else "")

        for iteration in range(max_iterations):
            if should_generate.any():
                outputs = self.model.generate(
                    current_input_ids,
                    attention_mask=current_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    stopping_criteria=stopping_criteria,
                )
                
                prompt_list = []
                true_indices = torch.nonzero(should_generate).squeeze(dim=1)
                for i in range(outputs.shape[0]):
                    j = true_indices[i]  # 映射回原始 batch 中的索引
                    sample_output = outputs[i]
                    # 截取新生成的部分（排除 prompt 部分）
                    new_tokens = sample_output[current_input_ids.size(1):]
                    new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                    # print(new_text)

                    # 只有当检测到 "</search>"、未检测到 eos token，
                    # 且当前迭代还未达到 max_iterations - 1 时才执行检索
                    if "</search>" in new_text and "<search>" in new_text and iteration < max_iterations - 1 :
                        print(f"样本 {j} 检测到 </search>")
                        # 截取 <search> ... </search> 之间的内容
                        start_index = new_text.find("<search>") + len("<search>")
                        end_index = new_text.find("</search>")
                        query = new_text[start_index:end_index].strip()
                        # 进行 Web 搜索
                        try:
                            search_result = str(web_search(query, count=3))
                        except Exception as e:
                            search_result = "检索当前query报错，请重新执行"

                        # **截断 new_text，仅保留 </search> 之前的内容**
                        new_text = new_text[:end_index + len("</search>")]

                        # **拼接 observation**
                        new_text += f"\n<observation>{search_result}</observation>\n"

                        # 重新编码：将历史 token 与新生成的文本拼接
                        history_ids = current_input_ids[i]
                        new_input_ids = self.tokenizer.encode(new_text, return_tensors="pt").to(device)
                        updated_input_ids = torch.cat([history_ids, new_input_ids[0]], dim=-1)
                        
                        prompt_list.append(updated_input_ids)

                    elif (eos_token_str == "" or eos_token_str not in new_text[-1]) and iteration < max_iterations - 1 :
                        # 重新编码：将历史 token 与新生成的文本拼接, 继续生成
                        history_ids = current_input_ids[i]
                        new_input_ids = self.tokenizer.encode(new_text, return_tensors="pt").to(device)
                        updated_input_ids = torch.cat([history_ids, new_input_ids[0]], dim=-1)
                        
                        prompt_list.append(updated_input_ids)
                
                    else:
                        # 如果检测到 eos token 或未出现检索标记，则认为生成完成
                        should_generate[j] = False
                        all_outputs[j] = outputs[i]
                        
                if should_generate.any():
                    # 对更新后的 prompt_list 重新进行 batch 化：调用 tokenizer 生成 input_ids 与 attention_mask，
                    # 注意 padding_side 保持为 "left" 与初始一致
                    new_prompt_list = [self.tokenizer.decode(ids, skip_special_tokens=False)
                                       for ids in prompt_list]
                    inputs = self.tokenizer(new_prompt_list, return_tensors="pt", padding=True, padding_side="left")
                    current_input_ids = inputs["input_ids"].to(device)
                    current_attention_mask = inputs["attention_mask"].to(device)
            else:
                break 

        # 对所有输出进行对齐，填充值使用 eos_token_id
        padded_outputs = pad_sequence(all_outputs, batch_first=True, padding_value=eos_token_id)
        # 后处理：确保每个生成文本按照要求截断和补全
        # final_outputs = self._postprocess_responses(padded_outputs)

        # 取第一个样本并打印格式
        for item_output in padded_outputs:
            print("****"*20)
            print("First output:", self.tokenizer.decode(item_output, skip_special_tokens=True))
            print("****"*20)

        return padded_outputs
