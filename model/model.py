import logging
import pdb
import re

import json5
import torch
from transformers import PreTrainedModel, StoppingCriteria, StoppingCriteriaList

from utils.Tools import Tools


class ThinkTagStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, think_token="</search>"):
        super().__init__()
        # Target sequences
        self.target_ids_1 = [522, 1836, 29]  # "</search>"
        self.target_ids_2 = [522, 1836, 397]  # "</search>\n"

    # Any one stop
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] < 3:
            return False

        last_tokens = input_ids[:, -3:]  # shape: [batch_size, 3]

        device = input_ids.device
        t1 = torch.tensor(self.target_ids_1, device=device)
        t2 = torch.tensor(self.target_ids_2, device=device)

        # Compare separately to see if any sequence's last three tokens match target_ids_1 or target_ids_2
        match_t1 = (last_tokens == t1).all(dim=1).any()
        match_t2 = (last_tokens == t2).all(dim=1).any()

        return match_t1 or match_t2


class CustomModel(PreTrainedModel):
    def __init__(self, model, tokenizer):
        super().__init__(model.config)

        self.model = model
        self.tokenizer = tokenizer
        self.tool = Tools()

    def forward(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=1000,
        max_length_for_gather=2000,
        do_sample=True,
        temperature=0.8,
        logits_to_keep=None,
        obtain_logits=False,
        max_generate_iterations=8,
        **kwargs,
    ):
        """
        Entry function that is automatically called during DataParallel processing.
        """
        if not obtain_logits:
            # Text generation mode
            return self.generate_with_think_interruption(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                max_length_for_gather=max_length_for_gather,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_generate_iterations=max_generate_iterations,
                **kwargs,
            )
        else:
            # Logits retrieval mode
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep + 1,
            ).logits

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=1000,
        max_length_for_gather=2000,
        do_sample=True,
        temperature=0.8,
        pad_token_id=None,
        eos_token_id=None,
        max_generate_iterations=8,
        **kwargs,
    ):
        return self.forward(
            input_ids,
            attention_mask=attention_mask,
            obtain_logits=False,
            max_new_tokens=max_new_tokens,
            max_length_for_gather=max_length_for_gather,
            do_sample=do_sample,
            temperature=temperature,
            max_generate_iterations=max_generate_iterations,
            **kwargs,
        )

    def call_plugin(self, plugin_name: str, plugin_args: str):
        try:
            plugin_args = json5.loads(plugin_args)
            plugin_args = {"input": plugin_args}
        except Exception as e:
            plugin_args = {"input": str(plugin_args)}

        def format_result(result):
            if isinstance(result, list):
                return "\n".join(str(item) for item in result)
            return str(result)

        try:
            if plugin_name == "google_search":
                result = self.tool.Web_RAG(**plugin_args)
                return "\nObservation:" + format_result(result)
            elif plugin_name == "HypothesisOutput":
                result = self.tool.HypothesisOutput(**plugin_args)
                return "\nObservation:" + format_result(result)
            elif plugin_name == "MedicalNER":
                result = self.tool.MedicalNER(**plugin_args)
                return "\nObservation:" + format_result(result)
            elif plugin_name == "DOC_RAG":
                result = self.tool.DOC_RAG(**plugin_args)
                return "\nObservation:" + format_result(result)
            elif plugin_name == "KG_RAG":
                result = self.tool.KG_RAG(**plugin_args)
                return "\nObservation:" + format_result(result)
            elif plugin_name == "Baike_RAG":
                result = self.tool.Baike_RAG(**plugin_args)
                return "\nObservation:" + format_result(result)
            elif plugin_name == "Web_RAG":
                result = self.tool.Web_RAG(**plugin_args)
                return "\nObservation:" + format_result(result)
            elif plugin_name == "Wiki_RAG":
                result = self.tool.Wiki_RAG(**plugin_args)
                return "\nObservation:" + format_result(result)
            elif plugin_name == "KnowledgeOrganize":
                result = self.tool.KnowledgeOrganize(**plugin_args)
                return "\nObservation:" + format_result(result)
            elif plugin_name == "Filter":
                result = self.tool.Filter(**plugin_args)
                return "\nObservation:" + format_result(result)
            else:
                return "\nObservation:" + f"Plugin {plugin_name} not found"
        except Exception as e:
            return "\nObservation:" + str(e)

    def parse_latest_plugin_call(self, text):
        match = re.match(r'\[(.*?)\]:\s*(?:"(.*?)"|(.*))', text)
        if match:
            plugin_name = match.group(1)  # 提取 TAG（如 DOC, WEB, KG）
            plugin_args = match.group(2) if match.group(2) is not None else match.group(3)  # 处理带引号和不带引号的情况
        else:
            plugin_name = "Web_RAG"  # default
            plugin_args = text

        plugin_name = re.sub(r"[^a-zA-Z_]", "", plugin_name)  # re-format

        return plugin_name, plugin_args.strip() if plugin_args else ""

    def padding_and_truncate(self, all_outputs, device, max_length_for_gather):
        decoded_outputs = []
        for out_tensor in all_outputs:
            if out_tensor is None:
                decoded_outputs.append("")
            else:
                decoded_text = self.tokenizer.decode(out_tensor, skip_special_tokens=True)
                decoded_outputs.append(decoded_text)

        encoded_outputs = self.tokenizer(
            decoded_outputs,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length_for_gather,
            truncation=True,
        )

        padded_outputs = encoded_outputs.input_ids.to(device)

        for i, text in enumerate(decoded_outputs):
            if not text:
                padded_outputs[i] = torch.full(
                    (max_length_for_gather,),
                    fill_value=self.tokenizer.eos_token_id,
                    dtype=torch.long,
                    device=device,
                )

        # 检查当前输入中是否有前导的 eos token（所有样本在同一位置都是 eos token）
        # 如果有，则跳过这些位置，减少不必要的计算
        leading_eos = []
        for pos in range(padded_outputs.size(1)):
            if (padded_outputs[:, pos] == self.tokenizer.eos_token_id).all():
                leading_eos.append(pos)
            else:
                break

        # 如果找到了前导的 eos token，则调整输入来跳过这些位置
        if leading_eos and len(leading_eos) > 0:
            skip_len = leading_eos[-1] + 1
            padded_outputs = padded_outputs[:, skip_len:]
        
        # 如果全是eos，只保留一个位置
        if padded_outputs.size(1) == 0:
            padded_outputs = torch.full(
                (padded_outputs.size(0), 1),
                fill_value=self.tokenizer.eos_token_id,
                dtype=torch.long,
                device=device
            )

        return padded_outputs

    def generate_with_think_interruption(
        self,
        prompt_ids,
        attention_mask,
        max_new_tokens,
        max_length_for_gather,
        do_sample,
        temperature,
        pad_token_id,
        eos_token_id,
        max_generate_iterations,
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

        for iteration in range(max_generate_iterations):
            # pdb.set_trace()
            # 检查当前输入中是否有前导的 eos token（所有样本在同一位置都是 eos token）
            # 如果有，则跳过这些位置，减少不必要的计算
            leading_eos = []
            for pos in range(current_input_ids.size(1)):
                if (current_input_ids[:, pos] == eos_token_id).all():
                    leading_eos.append(pos)
                else:
                    break

            # 如果找到了前导的 eos token，则调整输入和注意力掩码来跳过这些位置
            if leading_eos and len(leading_eos) > 0:
                skip_len = leading_eos[-1] + 1
                current_input_ids = current_input_ids[:, skip_len:]
                current_attention_mask = current_attention_mask[:, skip_len:]

            if should_generate.any():
                active_indices = torch.nonzero(should_generate).squeeze(dim=1)

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
                    if "<search>" in new_text and "</search>" in new_text and (iteration < max_generate_iterations - 1):
                        # 找到 <search> ... </search>
                        start_idx = new_text.index("<search>") + len("<search>")
                        end_idx = new_text.index("</search>")
                        query = new_text[start_idx:end_idx].strip()

                        # NOTE tools calling
                        try:
                            plugin_name, plugin_args = self.parse_latest_plugin_call(query)
                            search_result = self.call_plugin(plugin_name, plugin_args)  # TODO FIXME here
                            print(f"tools calling: {plugin_name}, {search_result[:100]}")
                        except Exception as e:
                            print(f"tools calling error: {e}")
                            search_result = f"tools calling error: {str(e)}"

                        # 在文本层面截断到 </search> 后，插入 <observation> ... </observation>
                        # 先保留到 </search>（含）
                        search_end_str = "</search>"
                        sub_text = new_text[: new_text.index(search_end_str) + len(search_end_str)]

                        # 拼接 observation
                        appended_text = f"{sub_text}\n<observation>\n{search_result}\n</observation>\n"

                        # 拼回历史文本
                        history_text = self.tokenizer.decode(sample_output[:old_len], skip_special_tokens=True)
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

                        if iteration < max_generate_iterations - 1 and not eos_found:
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

        # 对所有输出做对齐
        padded_outputs = self.padding_and_truncate(all_outputs, device, max_length_for_gather)
        return padded_outputs
