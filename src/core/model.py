from typing import Any, List, Optional, Tuple

import json5
import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    StoppingCriteria,
    StoppingCriteriaList,
)

import src.neuron.parse_tokens as parse_tokens
from src.neuron.neuron_metric import NeuronMetric
from src.common.mask import *


class HammingDiversityLogitsProcessor(LogitsProcessor):
    def __init__(self, beams_history, lambda_penalty=1.0, top_k=32):
        super().__init__()
        self.beams_history = beams_history
        self.lambda_penalty = lambda_penalty
        self.top_k = top_k

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape
        for beam_idx in range(batch_size):
            history = input_ids[beam_idx].tolist()
            if self.top_k is None or self.top_k == 0 or self.top_k >= vocab_size:
                token_indices = range(vocab_size)
            else:
                _, topk_indices = torch.topk(scores[beam_idx], self.top_k)
                token_indices = topk_indices.tolist()
            for token_id in token_indices:
                penalty = 0.0
                candidate_seq = history + [token_id]
                for other_idx, other_history in enumerate(self.beams_history):
                    if other_idx == beam_idx:
                        continue
                    min_len = min(len(candidate_seq), len(other_history))
                    for i in range(min_len):
                        if candidate_seq[i] == other_history[i]:
                            penalty += (i + 1) / min_len
                scores[beam_idx, token_id] -= self.lambda_penalty * penalty
        return scores


class SearchTagStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: Any, stop_action_token: List[str] = ["<search>", "</search>", "</backtrack>", "</summary>"]) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.target_ids = []
        for tok in stop_action_token:
            ids = tokenizer.encode(tok)
            if ids:
                self.target_ids.append(ids)
            ids2 = tokenizer.encode(tok + "\n")
            if ids2:
                self.target_ids.append(ids2)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        device = input_ids.device
        seq_len = input_ids.size(1)
        seq_len_range = [3, 4]
        for seq_len_range_itm in seq_len_range:
            if seq_len >= seq_len_range_itm:
                last_tokens = input_ids[:, -seq_len_range_itm:]
                for ids in self.target_ids:
                    if not ids or len(ids) < seq_len_range_itm:
                        continue
                    if len(ids) >= seq_len_range_itm:
                        t = torch.tensor(ids[-seq_len_range_itm:], device=device)
                        if (last_tokens == t).all(dim=1).any():
                            return True
        return False


class AgenticRAGModel(PreTrainedModel):
    def __init__(self, model: PreTrainedModel, tokenizer: Any, tool_registry: Any = None, **kwargs: Any) -> None:
        super().__init__(model.config)
        self.model = model
        self.tokenizer = tokenizer
        self.masked_spans_per_sample = []
        self.masked_parallel_spans_per_sample = []
        self.masked_parellel_spans_per_sample = self.masked_parallel_spans_per_sample
        self._tool_registry = tool_registry

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int = 1000,
        max_length_for_gather: int = 2000,
        do_sample: bool = True,
        temperature: float = 0.8,
        logits_to_keep: Optional[int] = None,
        obtain_logits: bool = False,
        max_generate_iterations: int = 8,
        use_diverse_sampling: bool = False,
        diversity_penalty: float = 1.0,
        calculate_param_importance: bool = False,
        use_SSRL: bool = False,
        enable_2D_attention_mask: bool = True,
        **kwargs: Any,
    ) -> torch.LongTensor:
        if not obtain_logits:
            if use_SSRL:
                if not enable_2D_attention_mask:
                    return self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        enable_2D_attention_mask=enable_2D_attention_mask,
                    )
                return self.generate_with_think_interruption(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    max_length_for_gather=max_length_for_gather,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_generate_iterations=max_generate_iterations,
                    use_diverse_sampling=use_diverse_sampling,
                    diversity_penalty=diversity_penalty,
                    calculate_param_importance=calculate_param_importance,
                    enable_2D_attention_mask=enable_2D_attention_mask,
                    use_SSRL=True,
                    **kwargs,
                )
            return self.generate_with_think_interruption(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                max_length_for_gather=max_length_for_gather,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_generate_iterations=max_generate_iterations,
                use_diverse_sampling=use_diverse_sampling,
                diversity_penalty=diversity_penalty,
                calculate_param_importance=calculate_param_importance,
                enable_2D_attention_mask=enable_2D_attention_mask,
                **kwargs,
            )
        if not enable_2D_attention_mask:
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=(logits_to_keep or 0) + 1,
            ).logits
            return logits
        current_casual_mask = expand_to_causal_mask_backtrack(attention_mask, self.masked_spans_per_sample, dtype=self.dtype)
        if not use_SSRL:
            current_casual_mask_parellel = expand_to_causal_mask_parallel(attention_mask, self.masked_parallel_spans_per_sample, dtype=self.dtype)
            current_casual_mask = (current_casual_mask_parellel + current_casual_mask) / 2
        logits = self.model(
            input_ids=input_ids,
            attention_mask=current_casual_mask,
            logits_to_keep=(logits_to_keep or 0) + 1,
        ).logits
        return logits

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int = 1000,
        max_length_for_gather: int = 2000,
        do_sample: bool = True,
        temperature: float = 0.8,
        max_generate_iterations: int = 8,
        use_diverse_sampling: bool = False,
        diversity_penalty: float = 1.0,
        calculate_param_importance: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor:
        return self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            obtain_logits=False,
            max_new_tokens=max_new_tokens,
            max_length_for_gather=max_length_for_gather,
            do_sample=do_sample,
            temperature=temperature,
            max_generate_iterations=max_generate_iterations,
            use_diverse_sampling=use_diverse_sampling,
            diversity_penalty=diversity_penalty,
            calculate_param_importance=calculate_param_importance,
            **kwargs,
        )

    def call_plugin(self, plugin_name: str, plugin_args: str) -> str:
        try:
            args = json5.loads(plugin_args)
            payload = {"input": args}
        except Exception:
            payload = {"input": plugin_args}
        result_payload = {"plugin": plugin_name, "ok": False, "data": None, "error": None}
        try:
            tool = None
            if self._tool_registry is not None:
                tool = self._tool_registry.get(plugin_name)
            if tool is None:
                from src.utils.Tools import Tools
                t = Tools()
                if hasattr(t, plugin_name):
                    result = getattr(t, plugin_name)(**payload)
                    result_payload.update({"ok": True, "data": result})
                else:
                    result_payload["error"] = f"Plugin {plugin_name} not found"
            else:
                result = tool.invoke(payload)
                result_payload.update({"ok": True, "data": result})
        except Exception as exc:
            result_payload["error"] = str(exc)
        import json as _json
        return f"\nObservation:{_json.dumps(result_payload, ensure_ascii=False)}"

    # 省略 generate_with_think_interruption 的实现，复用原 src/models/model.py 版本
    def generate_with_think_interruption(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        max_new_tokens: int,
        max_length_for_gather: int,
        do_sample: bool,
        temperature: float,
        pad_token_id: int,
        eos_token_id: int,
        max_generate_iterations: int,
        use_diverse_sampling: bool = False,
        diversity_penalty: float = 1.0,
        calculate_param_importance: bool = False,
        enable_2D_attention_mask: bool = True,
        use_SSRL: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor:
        device = input_ids.device
        batch_size = input_ids.size(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        should_gen = torch.ones(batch_size, dtype=torch.bool, device=device)
        outputs: List[Optional[torch.LongTensor]] = [None] * batch_size
        criteria = StoppingCriteriaList([SearchTagStoppingCriteria(self.tokenizer)])
        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()
        beams_history = [[] for _ in range(batch_size)]
        for _ in range(max_generate_iterations):
            skip_len = 0
            for pos in range(current_ids.size(1)):
                if (current_ids[:, pos] == eos_token_id).all():
                    skip_len += 1
                else:
                    break
            if skip_len:
                current_ids = current_ids[:, skip_len:]
                current_mask = current_mask[:, skip_len:]
            if not should_gen.any():
                break
            active = torch.nonzero(should_gen).squeeze(1)
            logits_processor = None
            if use_diverse_sampling:
                logits_processor = LogitsProcessorList(
                    [HammingDiversityLogitsProcessor(beams_history, lambda_penalty=diversity_penalty)]
                )
            gen_out_dict = self.model.generate(
                input_ids=current_ids,
                attention_mask=current_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                stopping_criteria=criteria,
                return_dict_in_generate=True,
                logits_processor=logits_processor,
            )
            gen_out = gen_out_dict.sequences
            next_prompts = []
            for idx, seq in enumerate(gen_out):
                b = active[idx].item()
                old_len = current_ids.size(1) - 1
                new_tokens = seq[old_len:]
                beams_history[b].extend(new_tokens.tolist())
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                if "</answer>" in text:
                    end = text.index("</answer>") + len("</answer>")
                    prev = self.tokenizer.decode(seq[:old_len], skip_special_tokens=False)
                    final = prev + text[:end]
                    outputs[b] = torch.tensor(self.tokenizer.encode(final), device=device)
                    should_gen[b] = False
                    continue
                if "<search>" in text and "</search>" in text and (_ < max_generate_iterations - 1):
                    part = text
                    s = part.index("<search>") + len("<search>")
                    e = part.index("</search>")
                    query = part[s:e].strip()
                    obs = self.call_plugin(*self.parse_latest_plugin_call(query))
                    sub = part[: e + len("</search>")]
                    merged = self.tokenizer.decode(seq[:old_len], skip_special_tokens=True)
                    merged += sub + obs + "\n"
                    next_prompts.append(torch.tensor(self.tokenizer.encode(merged), device=device))
                    continue
                eos_found = eos_token_id in new_tokens.tolist()
                if not eos_found and (_ < max_generate_iterations - 1):
                    continue_ids = torch.cat([seq[:old_len], new_tokens], dim=0)
                    next_prompts.append(continue_ids)
                else:
                    outputs[b] = seq
                    should_gen[b] = False
            if next_prompts:
                texts = [self.tokenizer.decode(t, skip_special_tokens=False) for t in next_prompts]
                enc = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side="left")
                current_ids = enc.input_ids.to(device)
                current_mask = enc.attention_mask.to(device)
        final_output = self.prompt_left_generation_right_padding(input_ids, outputs, device, max_length_for_gather)
        return final_output

    def parse_latest_plugin_call(self, text: str) -> Tuple[str, str]:
        import re as _re
        pattern = r'\[(.*?)\]:\s*(?:"(.*?)"|(.*))'
        match = _re.match(pattern, text)
        if match:
            name = match.group(1)
            args = match.group(2) or match.group(3) or ""
        else:
            name, args = "Web_RAG", text
        name = _re.sub(r"[^a-zA-Z_]", "", name)
        return name, args.strip()

    def padding_and_truncate(self, all_outputs: List[Optional[torch.LongTensor]], device: torch.device, max_length_for_gather: int) -> torch.LongTensor:
        decoded: List[str] = []
        for seq in all_outputs:
            decoded.append("") if seq is None else decoded.append(self.tokenizer.decode(seq, skip_special_tokens=True))
        enc = self.tokenizer(decoded, return_tensors="pt", padding="max_length", max_length=max_length_for_gather, truncation=True)
        padded = enc.input_ids.to(device)
        for i, txt in enumerate(decoded):
            if not txt:
                padded[i] = torch.full((max_length_for_gather,), self.tokenizer.eos_token_id, dtype=torch.long, device=device)
        skip = 0
        for pos in range(padded.size(1)):
            if (padded[:, pos] == self.tokenizer.eos_token_id).all():
                skip += 1
            else:
                break
        padded = padded[:, skip:] if skip < padded.size(1) else padded[:, :1]
        return padded

    def prompt_left_generation_right_padding(self, input_ids: torch.LongTensor, outputs: List[Optional[torch.LongTensor]], device: torch.device, max_length_for_gather: int) -> torch.LongTensor:
        batch_size = input_ids.size(0)
        input_contents = []
        generation_parts = []
        max_gen_len = 0
        for i in range(batch_size):
            input_seq = input_ids[i]
            non_pad_mask = input_seq != self.tokenizer.eos_token_id
            non_pad_len = non_pad_mask.sum().item()
            input_content = input_seq[-non_pad_len:]
            input_contents.append(input_content)
            if outputs[i] is None:
                generation_parts.append(None)
                continue
            output_seq = outputs[i]
            input_len = len(input_content)
            output_len = len(output_seq)
            input_end_pos = output_len
            non_pad_mask = output_seq != self.tokenizer.eos_token_id
            first_non_pad = torch.nonzero(non_pad_mask, as_tuple=True)[0]
            if len(first_non_pad) > 0:
                start_pos = first_non_pad[0].item()
                if start_pos + input_len <= output_len and torch.equal(output_seq[start_pos : start_pos + input_len], input_content):
                    input_end_pos = start_pos + input_len
            gen_part = output_seq[input_end_pos:]
            generation_parts.append(gen_part)
            max_gen_len = max(max_gen_len, len(gen_part))
        padded_outputs = torch.full((batch_size, max_gen_len), self.tokenizer.eos_token_id, dtype=torch.long, device=device)
        for i, gen in enumerate(generation_parts):
            if gen is None:
                continue
            padded_outputs[i, : len(gen)] = gen
        combined = torch.cat([input_ids, padded_outputs], dim=1)
        return combined
