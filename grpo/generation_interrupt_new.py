import logging
from transformers import PreTrainedModel, StoppingCriteria, StoppingCriteriaList
import torch
from torch.nn.utils.rnn import pad_sequence
import re

# 统一采用新代码中的搜索器（例如 EnglishWebSearcher）
from utils.web_search import web_search


# TODO any
class ThinkTagStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, think_token="</search>"):
        super().__init__()
        # 目标序列
        self.target_ids_1 = [522, 1836, 29]
        self.target_ids_2 = [522, 1836, 397]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 只检查 input_ids[0]
        sample = input_ids[0]
        # 确保序列长度至少有 3
        if sample.size(0) >= 3:
            # 取末尾 3 个 token
            tail_slice = sample[-3:]
            # 转成 Python list 后比对
            if tail_slice.tolist() == self.target_ids_1 or tail_slice.tolist() == self.target_ids_2:
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
        self.max_length_for_gather = 4000  # 不能太短了，太短了全是eos token
        # self.searcher = searcher if searcher is not None else EnglishWebSearcher()

    # def forward(self, input_ids, attention_mask=None):
    #     return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        logits_to_keep=None,
        obtain_logits=False,
        **kwargs,
    ):
        """
        这里是 DataParallel 并行时会自动调用的入口。
        """
        import pdb

        pdb.set_trace()
        # 可在此直接写生成逻辑，或调用自定义的 generate_with_think_interruption()
        if not obtain_logits:
            generated_output = self.generate_with_think_interruption(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            return generated_output
        else:  # 第二次采样，只是获得logits
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep + 1,
            ).logits

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.7,
        pad_token_id=None,
        eos_token_id=None,
    ):
        generated_output = self.generate_with_think_interruption(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        return generated_output

    def padding_and_truncate(self, all_outputs, device):
        # 2) 初始化一个列表来收集处理后的张量
        final_outputs = []
        for out_tensor in all_outputs:
            # 如果太长，先截断
            if out_tensor.size(0) > self.max_length_for_gather:
                out_tensor = out_tensor[: self.max_length_for_gather]

            # 构造一个 shape=[max_length_for_gather] 的张量，用于容纳最终序列
            padded_tensor = torch.full(
                (self.max_length_for_gather,),
                fill_value=self.tokenizer.eos_token_id,
                dtype=torch.long,
                device=device,
            )

            # 将实际内容拷贝到 padded_tensor 的前 out_tensor.size(0) 部分
            padded_tensor[: out_tensor.size(0)] = out_tensor

            # 把这个处理好的固定长度序列放进 final_outputs
            final_outputs.append(padded_tensor)

        # 3) 堆叠成为 [batch_size, max_length_for_gather] 的 2D 张量
        padded_outputs = torch.stack(final_outputs, dim=0)
        return padded_outputs

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
        多轮生成逻辑：用文本字符串来检测 <search> / </search>、<answer> / </answer> 等标签。
        1) 初始输入 prompt_ids 与 attention_mask
        2) 调用 self.model.generate
        3) 解码到文本后判断:
           - 若检测到 '</answer>' => 截断到此并终止
           - 若检测到 '<search>' & '</search>' => 提取查询，做搜索，用字符串插入 <observation> ... </observation>
        4) 重新编码作为下一轮输入
        5) 重复直至 max_iterations 或检测到 eos/answer_end
        """
        device = prompt_ids.device
        batch_size = prompt_ids.size(0)

        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        current_input_ids = prompt_ids.clone()
        current_attention_mask = attention_mask.clone()

        # 标记还需要继续生成的样本
        should_generate = torch.ones(batch_size, dtype=torch.bool, device=device)
        all_outputs = [None] * batch_size

        stopping_criteria = StoppingCriteriaList([ThinkTagStoppingCriteria(self.tokenizer)])

        for iteration in range(max_iterations):
            if should_generate.any():
                active_indices = torch.nonzero(should_generate).squeeze(dim=1)
                import pdb

                pdb.set_trace()
                # # 只对还在生成的样本进行 generate
                # if iteration == 0:
                #     for i in current_input_ids:
                #         print("*" * 100)
                #         print(self.tokenizer.decode(i, skip_special_tokens=False))
                #         print("*" * 100)
                outputs = self.model.generate(
                    current_input_ids,
                    attention_mask=current_attention_mask,
                    # max_new_tokens=max_new_tokens,
                    max_new_tokens=200,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    stopping_criteria=stopping_criteria,
                )
                import pdb

                pdb.set_trace()

                new_prompt_list = []

                for i in range(outputs.size(0)):
                    batch_idx = active_indices[i]
                    sample_output = outputs[i]

                    # 截取新生成的部分
                    old_len = current_input_ids.size(1) - 1
                    new_tokens = sample_output[old_len:]

                    # 解码本轮新增文本
                    new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                    # logging.info(f"[Iteration {iteration}] New text => {new_text}")

                    # print("new_tokens:" + "*"*100)
                    # for i in new_tokens:
                    #     print(i)
                    # print("new_tokens:" + "*"*100)

                    # ------- 1) 检查是否出现 '</answer>' -------
                    # 如果出现，就截断到 '</answer>' 并结束
                    if "</answer>" in new_text:
                        # 找到第一个 </answer> 的位置
                        end_idx = new_text.index("</answer>") + len("</answer>")
                        truncated_text = new_text[:end_idx]

                        # 拼回历史
                        history_text = self.tokenizer.decode(sample_output[:old_len], skip_special_tokens=False)
                        final_text = history_text + truncated_text

                        # 再编码成 token 用于保存
                        final_ids = self.tokenizer.encode(final_text, return_tensors="pt")[0].to(device)

                        all_outputs[batch_idx] = final_ids
                        should_generate[batch_idx] = False
                        continue

                    # ------- 2) 若没出现 '</answer>'，再检查 <search> ... </search> -------
                    if "<search>" in new_text and "</search>" in new_text and (iteration < max_iterations - 1):
                        # 找到 <search> ... </search>
                        start_idx = new_text.index("<search>") + len("<search>")
                        end_idx = new_text.index("</search>")
                        query = new_text[start_idx:end_idx].strip()

                        # 调用搜索
                        try:
                            search_result = web_search(query, count=1)
                        except Exception as e:
                            search_result = f"检索出错: {str(e)}"

                        # 在文本层面截断到 </search> 后，插入 <observation> ... </observation>
                        # 先保留到 </search>（含）
                        search_end_str = "</search>"
                        sub_text = new_text[: new_text.index(search_end_str) + len(search_end_str)]

                        # 拼接 observation
                        appended_text = f"{sub_text}\n<observation>\n{search_result}\n</observation>\n"

                        # 拼回历史文本
                        history_text = self.tokenizer.decode(sample_output[:old_len], skip_special_tokens=False)
                        updated_text = history_text + appended_text

                        # 重新编码
                        updated_ids = self.tokenizer.encode(updated_text, return_tensors="pt")[0].to(device)
                        new_prompt_list.append(updated_ids)
                    else:
                        # ------- 3) 如果没出现 <search> ... </search> 或已经是最后一次，则继续生成或结束 -------
                        # 检查是否出现 eos
                        # （如果 eos_token_id 在 new_tokens 里，说明模型可能终止了）
                        eos_found = False
                        if eos_token_id is not None:
                            eos_found = (new_tokens == eos_token_id).any().item()

                        if iteration < max_iterations - 1 and not eos_found:
                            # 拼回历史，继续下一轮
                            history_ids = sample_output[:old_len]
                            updated_input_ids = torch.cat([history_ids, new_tokens], dim=-1)
                            new_prompt_list.append(updated_input_ids)
                        else:
                            # 已出现 eos 或不再进行下一轮 => 生成完成
                            all_outputs[batch_idx] = sample_output
                            should_generate[batch_idx] = False

                # 若仍有样本要下一轮生成
                if should_generate.any():
                    # 重新将 new_prompt_list 组装
                    prompt_texts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in new_prompt_list]
                    inputs = self.tokenizer(
                        prompt_texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                    )
                    current_input_ids = inputs["input_ids"].to(device)
                    current_attention_mask = inputs["attention_mask"].to(device)
                else:
                    break
            else:
                break
        for i in all_outputs:
            print("*" * 100)
            print(self.tokenizer.decode(i, skip_special_tokens=False))
            print("*" * 100)

        # 对所有输出做对齐
        padded_outputs = self.padding_and_truncate(all_outputs, device)
        return padded_outputs
