from transformers import PreTrainedModel, StoppingCriteria, StoppingCriteriaList
import torch

from utils.web_search import * 

class ThinkTagStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, think_token="</search>"):
        self.tokenizer = tokenizer
        self.think_token = think_token
        self.think_token_id = tokenizer.encode(think_token, add_special_tokens=False)[0]

    def __call__(self, input_ids, scores):
        # print(input_ids.shape)
        last_token = input_ids[0][-1]
        # print(f"Last token ID: {last_token}, Think token ID: {self.think_token_id}")
        if last_token == self.think_token_id:
            # print("Stopping generation.")
            return True
        return False
    
    # def __call__(self, input_ids, scores):
    #     # TODO at least two batch
    #     # any one stop
    #     # print(input_ids.shape)  # 打印 input_ids 的形状
    #     # 检查 batch 中是否有任意一个样本的最后一个 token 是 `</search>`
    #     any_stop = any(sample[-1] == self.think_token_id for sample in input_ids)
    #     if any_stop:
    #         print("Stopping generation.")
    #         return True  # 停止生成
    #     return False  # 继续生成

    # def __call__(self, input_ids, scores):
    #     # all stop
    #     print(input_ids.shape)  # 打印 input_ids 的形状
    #     for i, sample in enumerate(input_ids):
    #         last_token = sample[-1]
    #         print(f"Sample {i}: Last token ID: {last_token}, Think token ID: {self.think_token_id}")
    #     any_stop = any(sample[-1] == self.think_token_id for sample in input_ids)
    #     if any_stop:
    #         print("Stopping generation.")
    #         return True
    #     return False

class CustomModel(PreTrainedModel):
    def __init__(self, model, tokenizer):
        super().__init__(model.config)
        self.model = model
        self.tokenizer = tokenizer

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
        stopping_criteria = StoppingCriteriaList([ThinkTagStoppingCriteria(self.tokenizer)])

        batch_size = prompt_ids.size(0)
        device = prompt_ids.device

        # 初始化每个样本的输入和掩码
        current_input_ids = prompt_ids.clone()
        current_attention_mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(prompt_ids)
        
        # 保存所有生成的 token
        all_outputs = [None] * batch_size
        should_generate = torch.ones(batch_size, dtype=torch.bool, device=device)

        # 初始化 ignore_mask
        ignore_mask = torch.zeros_like(current_input_ids, dtype=torch.bool)

        for iteration in range(max_iterations):
            # 生成当前活跃的样本
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
                
                # 如果 should_generate 为 True，更新对应位置
                # 检查每个样本是否生成了 </reasoning>
                prompt_list = []
                true_indices = torch.nonzero(should_generate).squeeze(dim=1)

                for i in range(outputs.shape[0]):
                    j = true_indices[i]     # 构建绝对映射
                    sample_output = outputs[i]      # FIXME 第二次，其实是没有batch_size个样本的，这里需要修改一下
                    new_tokens = sample_output[current_input_ids.size(1):]
                    new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                    # print(new_text)

                    if "</search>" in new_text and iteration < max_iterations - 1:
                        print(f"样本 {j} 检测到 </search>")
                        
                        # 找到 <search> 和 </search> 之间的内容作为查询语句
                        start_index = new_text.find("<search>") + len("<search>")
                        end_index = new_text.find("</search>")
                        query = new_text[start_index:end_index].strip()
                        
                        # 使用提取的查询语句进行搜索
                        try:
                            custom_segment = str(web_search(query, count=3))
                        except:
                            custom_segment = "检索当前query报错，请重新执行"
                        
                        # 将搜索结果插入到文本中
                        new_text = new_text + f"<search result>{custom_segment}</search result>"

                        # 重新编码为新的 input_ids（包括历史上下文）
                        history_ids = current_input_ids[i]  # 历史 token    # 需要内层j和外层i的映射
                        new_input_ids = self.tokenizer.encode(new_text, return_tensors="pt").to(device)
                        # 拼接历史 token 和新 token
                        updated_input_ids = torch.cat([history_ids, new_input_ids[0]], dim=-1)
                        prompt_list.append(updated_input_ids)
                    # elif "</search>" in new_text and iteration < max_iterations - 1:  # TODO <EOS> token
                    #     print(f"样本 {j} 未检测到 <eos>")
                        
                    #     # 找到 <search> 和 </search> 之间的内容作为查询语句
                    #     start_index = new_text.find("<search>") + len("<search>")
                    #     end_index = new_text.find("</search>")
                    #     query = new_text[start_index:end_index].strip()
                        
                    #     # 使用提取的查询语句进行搜索
                    #     try:
                    #         custom_segment = str(web_search(query, count=3))
                    #     except:
                    #         custom_segment = "检索当前query报错，请重新执行"
                        
                    #     # 将搜索结果插入到文本中
                    #     new_text = new_text + f"<search result>{custom_segment}</search result>"

                    #     # 重新编码为新的 input_ids（包括历史上下文）
                    #     history_ids = current_input_ids[i]  # 历史 token    # 需要内层j和外层i的映射
                    #     new_input_ids = self.tokenizer.encode(new_text, return_tensors="pt").to(device)
                    #     # 拼接历史 token 和新 token
                    #     updated_input_ids = torch.cat([history_ids, new_input_ids[0]], dim=-1)
                    #     prompt_list.append(updated_input_ids)
                    else:
                        # 如果未生成 </reasoning>，直接使用当前输出
                        # sample_output = torch.cat([sample_output, torch.tensor([eos_token_id], device=device)], dim=-1)
                        # prompt_list.append(sample_output)
                        # 这里不需要进入下一次循环了
                        should_generate[j] = False
                        all_outputs[j] = outputs[i]

                if should_generate.any():  
                    new_prompt_list = []
                    # 将 prompt_list 中的 token 进行填充对齐
                    for prompt in prompt_list:
                        new_prompt_list.append(self.tokenizer.decode(prompt, skip_special_tokens=False))

                    inputs = self.tokenizer(new_prompt_list, return_tensors="pt", padding=True, padding_side="left")
                    current_input_ids = inputs["input_ids"].to(device)      # Shape: (batch_size, prompt_seq_len)
                    current_attention_mask = inputs["attention_mask"].to(device)  # Shape: (batch_size, prompt_seq_len)

                # # 标记 `"xxxx"` 的位置    # TODO add ignore mask
                # custom_segment_ids = self.tokenizer.encode("xxxx", add_special_tokens=False)
                # custom_segment_start = all_outputs.size(1)
                # custom_segment_end = custom_segment_start + len(custom_segment_ids)
                # ignore_mask[:, custom_segment_start:custom_segment_end] = True

            else:
                break 

        from torch.nn.utils.rnn import pad_sequence
        padded_reversed = pad_sequence(all_outputs, batch_first=True, padding_value=eos_token_id)

        return padded_reversed  #, ignore_mask
    