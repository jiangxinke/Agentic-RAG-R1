
import random
import re

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
from src.utils.Tools import Tools
from src.models.mask_utils import *


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
    """ 
    StoppingCriteria that halts generation when specific end-of-search tags appear. 
    This criteria checks the last three generated token IDs against predefined sequences corresponding
    to '</search>' and '</search>\n', and backtrack, summary. When any sequence in the batch matches, generation stops. 
     
     Args: tokenizer (Any): Tokenizer used for decoding (provides token IDs). 
     think_token (str): The text tag marking the end of a search thought. 
     
     Raises: None 
     
     """

    def __init__(
        self,
        tokenizer: Any,
        stop_action_token: List[str] = ["<search>", "</search>", "</backtrack>", "</summary>"],
    ) -> None:
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

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:

        device = input_ids.device
        seq_len = input_ids.size(1)

        seq_len_range = [3, 4]       # 适配不同的stop token size
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
    """
    Retrieval-Augmented Generation model with iterative 'think' interruptions.

    Wraps a base causal LM to interleave generation with external tool calls.

    Args:
        model (PreTrainedModel): The underlying language model.
        tokenizer (Any): Corresponding tokenizer.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(model.config)
        self.model = model
        self.tokenizer = tokenizer
        self.tool = Tools()
        self.masked_spans_per_sample = []       # 三元组的list
        self.masked_parellel_spans_per_sample = []  # 并行的search start/end list

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
        use_KV_Cache: bool = False,
        use_diverse_sampling: bool = False,
        diversity_penalty: float = 1.0,
        calculate_param_importance: bool = False,
        use_SSRL: bool = False,
        enable_2D_attention_mask: bool = True,
        **kwargs: Any,
    ) -> torch.LongTensor:
        """
        Forward pass: either generates tokens with thought interruptions or returns logits.

        Args:
            input_ids (torch.LongTensor): Input token IDs.
            attention_mask (torch.LongTensor): Attention mask.
            max_new_tokens (int): Max tokens per generation call.
            max_length_for_gather (int): Max length for final alignment.
            do_sample (bool): Whether to sample or greedy.
            temperature (float): Sampling temperature.
            logits_to_keep (Optional[int]): If logging probabilities, number of tokens.
            obtain_logits (bool): Switch to logits-only mode.
            max_generate_iterations (int): Iterations for think interruptions.
            use_KV_Cache (bool): Whether to use KV cache.
            **kwargs: Extra args forwarded to generate.

        Returns:
            torch.LongTensor: Generated token IDs or logits.

        Raises:
            RuntimeError: If model generation fails.
        """
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
                if enable_2D_attention_mask:     # TODO，把attention mask给传递出来
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
                        use_SSRL=True
                        **kwargs,
                    )
            else:
                if use_KV_Cache:        # TODO jxk 这个地方还没有支持2D mask => 先不着急吧本次
                    return self.generate_with_think_interruption_KV_Cache(
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
                else:      # TODO，把attention mask给传递出来
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

        else:
            # 不需要做打断的时候，就用原始的获取logits即可
            if not enable_2D_attention_mask:
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=(logits_to_keep or 0) + 1,
                ).logits
                return logits
            # 如果需要打断，则获取4D attention mask
            else:       
                # Step 1:
                print(f"[model.py] make 4D mask from 2D mask: {self.masked_spans_per_sample}")
                current_casual_mask = expand_to_causal_mask_backtrack(attention_mask, self.masked_spans_per_sample, dtype=self.dtype)
                # Step 2:
                ## 并行屏蔽 span，jiaran你构造这样的形式：
                ## self.masked_parellel_spans_per_sample是一个batch list：
                ## self.masked_parellel_spans_per_sample是一个batch[0]又是一个list，代表一个rollout的多次检索
                ## self.masked_parellel_spans_per_sample是一个batch[0][0]代表第1次检索
                ## self.masked_parellel_spans_per_sample是一个batch[0][0]=[(a,b),(c,d),(e,f)]把彼此两两之间的attn设置为0就行

                ### self.masked_parellel_spans_per_sample[0] ： 第0个rollout
                #### self.masked_parellel_spans_per_sample[0][6] ： 第0个rollout第6次并行search
                ##### self.masked_parellel_spans_per_sample[0][6] = [(a,b),(c,d),(e,f)]： b=c-1, e=d+1: 这个地方是绝对id
                ###### ：a<search>[path 1]: 我要查红楼梦贾宝玉 <\end path 1>b  ;  [path 2] 我查查林黛玉 <\end path 2>; .....
                ######                     [0][6][1]                                [0][6][2]

                ######: query：红楼梦的贾宝玉性格怎么样：
                ######## GRPO： rollout 0， rollout 1， ....， rollout 3
                ######### rollout 0: 思考0--search0--summary0--反思0--思考1--...--search6--回答
                ########### search6: path[1] 第36回贾宝玉做了什么.. path[2] 第36回林黛玉做了什么.. 

                if not use_SSRL:
                    current_casual_mask_parellel = expand_to_causal_mask_parallel(attention_mask, self.masked_parellel_spans_per_sample, dtype=self.dtype)
                    current_casual_mask = (current_casual_mask_parellel + current_casual_mask)/2

                # 无任何变化的从2D=>4D normal
                # current_casual_mask = expand_to_causal_mask(attention_mask, dtype=self.dtype)
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
        use_KV_Cache: bool = False,
        use_diverse_sampling: bool = False,
        diversity_penalty: float = 1.0,
        calculate_param_importance: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor:
        """
        Alias for forward in generation mode.

        See `forward` for parameter details.
        """
        return self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            obtain_logits=False,
            max_new_tokens=max_new_tokens,
            max_length_for_gather=max_length_for_gather,
            do_sample=do_sample,
            temperature=temperature,
            max_generate_iterations=max_generate_iterations,
            use_KV_Cache=use_KV_Cache,
            use_diverse_sampling=use_diverse_sampling,
            diversity_penalty=diversity_penalty,
            calculate_param_importance=calculate_param_importance,
            **kwargs,
        )

    def call_plugin(
        self,
        plugin_name: str,
        plugin_args: str,
    ) -> str:
        """
        Invoke an external tool plugin and format its output.

        Args:
            plugin_name (str): Name of tool method in `Tools`.
            plugin_args (str): JSON5 or raw string arguments.

        Returns:
            str: Formatted observation string.

        Raises:
            AttributeError: If plugin not found.
            Exception: On tool execution failure.
        """
        try:
            args = json5.loads(plugin_args)
            kwargs = {"input": args}
        except Exception:
            kwargs = {"input": plugin_args}

        if not hasattr(self.tool, plugin_name):
            raise AttributeError(f"Plugin {plugin_name} not found")

        result = getattr(self.tool, plugin_name)(**kwargs)
        if isinstance(result, list):
            result_str = "\n".join(str(item) for item in result)
        else:
            result_str = str(result)
        return f"\nObservation:{result_str}"

    def parse_latest_plugin_call(
        self,
        text: str,
    ) -> Tuple[str, str]:
        """
        Extract plugin name and arguments from model-generated text.

        Args:
            text (str): The generated text containing plugin call pattern.

        Returns:
            Tuple[str, str]: Cleaned plugin name and argument payload.
        """
        pattern = r'\[(.*?)\]:\s*(?:"(.*?)"|(.*))'
        match = re.match(pattern, text)
        if match:
            name = match.group(1)
            args = match.group(2) or match.group(3) or ""
        else:
            name, args = "Web_RAG", text
        name = re.sub(r"[^a-zA-Z_]", "", name)
        return name, args.strip()

    def padding_and_truncate(
        self,
        all_outputs: List[Optional[torch.LongTensor]],
        device: torch.device,
        max_length_for_gather: int,
    ) -> torch.LongTensor:
        """
        Align and pad generated outputs for batch postprocessing.

        Decodes token lists, re-encodes with fixed max length, then truncates leading
        EOS-only columns to reduce compute, ensuring at least one token remains.

        Args:
            all_outputs (List[Optional[torch.LongTensor]]): List of token sequences.
            device (torch.device): Target device for output tensor.
            max_length_for_gather (int): Max sequence length.

        Returns:
            torch.LongTensor: Tensor of shape (batch, <= max_length_for_gather).
        """
        decoded: List[str] = []
        for seq in all_outputs:
            if seq is None:
                decoded.append("")
            else:
                decoded.append(self.tokenizer.decode(seq, skip_special_tokens=True))

        enc = self.tokenizer(
            decoded,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length_for_gather,
            truncation=True,
        )
        padded = enc.input_ids.to(device)

        # Replace empty strings with EOS padding
        for i, txt in enumerate(decoded):
            if not txt:
                padded[i] = torch.full(
                    (max_length_for_gather,),
                    self.tokenizer.eos_token_id,
                    dtype=torch.long,
                    device=device,
                )

        # Remove leading all-EOS columns
        skip = 0
        for pos in range(padded.size(1)):
            if (padded[:, pos] == self.tokenizer.eos_token_id).all():
                skip += 1
            else:
                break
        padded = padded[:, skip:] if skip < padded.size(1) else padded[:, :1]
        return padded

    def prompt_left_generation_right_padding(
        self,
        input_ids: torch.LongTensor,
        outputs: List[Optional[torch.LongTensor]],
        device: torch.device,
        max_length_for_gather: int,
    ) -> torch.LongTensor:
        """
        Align input and generated parts with left padding for input and right padding for generation.

        Args:
            input_ids (torch.LongTensor): Input token IDs with left padding.
            outputs (List[Optional[torch.LongTensor]]): List of generated sequences.
            device (torch.device): Target device for output tensor.
            max_length_for_gather (int): Maximum sequence length.

        Returns:
            torch.LongTensor: Combined tensor with left-padded input and right-padded generation.
        """
        # pdb.set_trace()
        batch_size = input_ids.size(0)
        input_contents = []
        generation_parts = []
        max_gen_len = 0

        # extract input contents and generation parts
        for i in range(batch_size):
            input_seq = input_ids[i]
            non_pad_mask = input_seq != self.tokenizer.eos_token_id
            non_pad_len = non_pad_mask.sum().item()
            input_content = input_seq[-non_pad_len:]  # Get the actual input content
            input_contents.append(input_content)

            if outputs[i] is None:
                generation_parts.append(None)
                continue

            output_seq = outputs[i]
            input_len = len(input_content)
            output_len = len(output_seq)
            input_end_pos = output_len  # Default to end if not found

            non_pad_mask = output_seq != self.tokenizer.eos_token_id
            first_non_pad = torch.nonzero(non_pad_mask, as_tuple=True)[0]
            if len(first_non_pad) > 0:
                start_pos = first_non_pad[0].item()
                if start_pos + input_len <= output_len and torch.equal(
                    output_seq[start_pos : start_pos + input_len], input_content
                ):
                    input_end_pos = start_pos + input_len

            gen_part = output_seq[input_end_pos:]
            generation_parts.append(gen_part)
            max_gen_len = max(max_gen_len, len(gen_part))

        max_gen_len = min(max_gen_len, max_length_for_gather)

        # combine input and padded generation parts
        final_outputs = []
        for i in range(batch_size):
            input_content = input_contents[i]
            gen_part = generation_parts[i]

            if gen_part is None:
                # If no generation, just use input with right padding
                padded = torch.full(
                    (max_gen_len + len(input_ids[i]),),
                    self.tokenizer.eos_token_id,
                    dtype=torch.long,
                    device=device,
                )
                padded[: len(input_ids[i])] = input_ids[i]  # Use original input_ids with left padding
            else:
                # Right pad the generation part
                padded_gen = torch.full(
                    (max_gen_len,),
                    self.tokenizer.eos_token_id,
                    dtype=torch.long,
                    device=device,
                )
                gen_len = min(len(gen_part), max_gen_len)
                padded_gen[:gen_len] = gen_part[:gen_len]

                # Combine original input_ids with padded generation
                padded = torch.cat([input_ids[i], padded_gen])

            final_outputs.append(padded)

        # pdb.set_trace()
        return torch.stack(final_outputs)

    def generate_with_think_interruption(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
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
    ) -> torch.LongTensor:
        """
        Perform iterative generation with tool calls based on markers in text.

        The loop:
          1. Generate tokens.
          2. If '</answer>' appears, finalize that sample.
          3. If '<search>...</search>' appears, call tool, insert '<observation>' block, and continue.
          4. If <backtrack>/<summary> appears, reset the attention mask.
          5. Otherwise, continue generating until EOS or max iterations.

        Args:
            input_ids (torch.LongTensor): Starting tokens.
            attention_mask (Optional[torch.LongTensor]): Attention mask.
            max_new_tokens (int): Tokens per generation iteration.
            max_length_for_gather (int): Final gather length.
            do_sample (bool): Sampling flag.
            temperature (float): Sampling temperature.
            pad_token_id (int): Padding token ID.
            eos_token_id (int): End-of-sequence token ID.
            max_generate_iterations (int): Max loop iterations.

        Returns:
            torch.LongTensor: Padded output IDs for all samples.

        Raises:
            RuntimeError: On generation or tool-calling failures.
        """
        # pdb.set_trace()

        # FIXME diversity_penalty dynamic
        diversity_penalty = diversity_penalty if diversity_penalty is not None else random.uniform(0.5, 1.0)

        device = input_ids.device
        batch_size = input_ids.size(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        should_gen = torch.ones(batch_size, dtype=torch.bool, device=device)
        outputs: List[Optional[torch.LongTensor]] = [None] * batch_size
        criteria = StoppingCriteriaList([SearchTagStoppingCriteria(self.tokenizer)])

        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()

        # jxk 2D mask
        self.masked_spans_per_sample: List[List[Tuple[int, int, int]]] = [[] for _ in range(batch_size)]
        self.masked_parellel_spans_per_sample: List[List[List[Tuple[int, int]]]] = [[] for _ in range(batch_size)]

        beams_history = [[] for _ in range(batch_size)]
        for _ in range(max_generate_iterations):
            # Skip leading EOS columns
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

            current_mask = apply_masked_spans(current_mask, self.masked_spans_per_sample)
            # 插入对attentiin mask的修改：
            print(f"[model.py] current_ids.shape={current_ids.shape}, current_mask.shape={current_mask.shape}")
            gen_out = self.model.generate(
                current_ids,
                attention_mask=current_mask,   
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                stopping_criteria=criteria,
                logits_processor=logits_processor,
                return_dict_in_generate=calculate_param_importance,
                output_hidden_states=calculate_param_importance,
            )

            if calculate_param_importance:
                sequences = gen_out.sequences
                output_hidden_states = gen_out.hidden_states  # Tuple of (gen_seq_len, num_layers + 1, (batch_size, 1, hidden_size))
                neuron_metric = NeuronMetric()
            else:
                sequences = gen_out

            next_prompts = []

            # NOTE 以下对所有的 rollout 进行处理（beam search 和 2D mask）
            for idx, seq in enumerate(sequences):
                if idx >= len(active):
                    continue
                b = active[idx].item() # eun：只对 active 的 id 进行处理？这里应该判断不是 active 就 continue？
                old_len = current_ids.size(1)
                new_tokens = seq[old_len:]
                beams_history[b].extend(new_tokens.tolist())
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

                if calculate_param_importance:
                    # Calculate neuron importance for the newly generated tokens
                    token_spans = parse_tokens.locate_action_token_spans(new_tokens, self.tokenizer)
                    parse_tokens.accumulate_neuron_importance(token_spans["tag"], "tag", output_hidden_states, neuron_metric)
                    parse_tokens.accumulate_neuron_importance(token_spans["arg"], "arg", output_hidden_states, neuron_metric)

                # 1) Answer end
                if "</answer>" in text:
                    end = text.index("</answer>") + len("</answer>")
                    prev = self.tokenizer.decode(seq[:old_len], skip_special_tokens=False)
                    final = prev + text[:end]
                    outputs[b] = torch.tensor(self.tokenizer.encode(final), device=device)
                    should_gen[b] = False
                    continue

                # 2.0) for parellel search  # NOTE not support SSRL
                if "<search>" in text and (_ < max_generate_iterations - 1) and not use_SSRL:
                    print("[model.py] detect <search> (immediate beam trigger)")
                    part = text
                    s = part.index("<search>") + len("<search>")
                    
                    prefix = [self.tokenizer.decode(seq, skip_special_tokens=False)]
                    enc = self.tokenizer(prefix, return_tensors="pt", padding=True, padding_side="left")
                    prefix_ids = enc.input_ids.to(device)
                    prefix_mask = enc.attention_mask.to(device)

                    num_beams = 3
                    num_return_sequences = 3
                    
                    prefix_len = prefix_ids.size(1)
                    # prefix_text = self.tokenizer.decode(prefix_ids[0], skip_special_tokens=False).strip()
                    print("[model.py] detect <search> 当前beam search采样的前缀为：", prefix[0].strip())

                    # FIXME 这里有问题，这里的prefix_ids需要检查一下
                    beam_out = self.model.generate(
                        prefix_ids.to(device), # FIXED: 加上生成的内容
                        attention_mask=prefix_mask.to(device),
                        max_new_tokens=max_new_tokens, # default 100
                        num_beams=num_beams,
                        # stopping_criteria=criteria,
                        num_return_sequences=num_return_sequences,
                        do_sample=False,
                    )

                    search_lines, obs_lines = [], []
                    span_positions = []  # 用来存放每个 path 的 (start, end)

                    # step 1: decode and collect search/obs pairs
                    for idx_beam in range(beam_out.size(0)): # (0,1,2)
                        full_seq = beam_out[idx_beam]
                        gen_part = full_seq[prefix_len:]  # 只取新生成部分
                        key_text = self.tokenizer.decode(gen_part, skip_special_tokens=True).strip()

                        path_name = f"path{idx_beam+1}"

                        try:
                            pname, pargs = self.parse_latest_plugin_call(key_text)
                            obs = self.call_plugin(pname, pargs)
                        except Exception as exc:
                            obs = f"Error: {exc}"
                        
                        if '</search>' in key_text:
                            key_text = key_text[:key_text.index("</search>")]
                        # 搜索和观测文本块
                        search_lines.append(f"[{path_name}] {key_text} [/{path_name}]")
                        obs_lines.append(f"[{path_name}] {obs} [/{path_name}]")

                    # step 2: 拼接 search/obs 块
                    final_search_block = "<search>\n" + "\n".join(search_lines) + "\n</search>"
                    final_obs_block = "<observation>\n" + "\n".join(obs_lines) + "\n</observation>"

                    merged_text = self.tokenizer.decode(seq[:prefix_len], skip_special_tokens=True)
                    merged_text += "\n" + final_search_block + "\n" + final_obs_block

                    # step 3: 计算 token 位置（仅针对 search key）
                    # 为了计算绝对位置，重新对 merged_text 分词一次
                    merged_ids = self.tokenizer.encode(merged_text, add_special_tokens=False)
                    merged_str = self.tokenizer.decode(merged_ids, skip_special_tokens=False)

                    # 现在我们重新对 <search> 块部分进行 token 对齐
                    # 简单方法：对每个 [pathX] key [/pathX] 单独 encode，然后用偏移量计算
                    offset = 0
                    prefix_ids_len = len(self.tokenizer.encode(
                        self.tokenizer.decode(seq[:prefix_len], skip_special_tokens=True),
                        add_special_tokens=False
                    ))

                    current_offset = prefix_ids_len + 1 # merged_text 中 search 部分的起点 token idx（在这里加1，算入换行符

                    for idx_beam, line in enumerate(search_lines):
                        # 对这一行单独编码（包括标签），计算长度
                        line_ids = self.tokenizer.encode(line, add_special_tokens=False)
                        line_len = len(line_ids)

                        # 我们只想要 key 的内部区间
                        # 定位 key_text 的 token 起止，相对 line_ids 的位置
                        left_tag = f"[path{idx_beam+1}]"
                        right_tag = f"[/{'path'+str(idx_beam+1)}]"
                        left_ids = self.tokenizer.encode(left_tag, add_special_tokens=False)
                        right_ids = self.tokenizer.encode(right_tag, add_special_tokens=False)
                        # EUNINFO: 这里应该 encode 的是 line 而不是 key_text
                        key_ids = self.tokenizer.encode(key_text, add_special_tokens=False)
                        key_start = current_offset # + len(left_ids) # +1 for space （EUNINFO：这里不用加1
                        key_end = key_start + line_len # 这里算入了 [/path1]，但是并没有算 [path]，原来加的是 len(key_ids)

                        span_positions.append((key_start, key_end)) # 加入的区间应该是左闭右开？
                        current_offset += line_len  # 加换行符 （EUNINFO: 这里不用加1

                    # step 4: 存入 masked_parellel_spans_per_sample
                    if not hasattr(self, "masked_parellel_spans_per_sample"):
                        self.masked_parellel_spans_per_sample = [[] for _ in range(batch_size)]

                    self.masked_parellel_spans_per_sample[b].append(span_positions)

                    print(f"[model.py] masked spans for sample {b}:", span_positions)

                    # step 5: 将 merged_text 编码回 tensor 继续生成
                    next_prompts.append(torch.tensor(self.tokenizer.encode(merged_text), device=device))
                    continue
                
                # TODO: 以下部分是以前的非并发版本，加上非 parallel search 的判断。
                # 2) Search and observation
                if "<search>" in text and "</search>" in text and (_ < max_generate_iterations - 1) and not use_SSRL and False: # 需要加 not parallel_search
                    print("detect search")
                    part = text
                    s = part.index("<search>") + len("<search>")
                    e = part.index("</search>")
                    query = part[s:e].strip()
                    try:
                        pname, pargs = self.parse_latest_plugin_call(query)
                        obs = self.call_plugin(pname, pargs)
                    except Exception as exc:
                        obs = f"<observation>Error: {exc}"  # preserve flow
                    sub = part[: e + len("</search>")]
                    merged = self.tokenizer.decode(seq[:old_len], skip_special_tokens=True)
                    merged += sub + obs + "\n"
                    next_prompts.append(torch.tensor(self.tokenizer.encode(merged), device=device))
                    continue
                
                # 3） backtrack or summary actions
                ## 遇到这两个动作的时候，将该动作前的上一个动作，和该动作之后的所有动作的attention设置为False
                if ("</backtrack>" in text) or ("</summary>" in text): 
                    print("[model.py] deteck </backtrack> or </summary>")
                    # 当前完整序列
                    full_seq = torch.cat([seq[:old_len], new_tokens], dim=0)
                    full_text = self.tokenizer.decode(full_seq, skip_special_tokens=False)
                    print(f"[model.py] [zzx debug] full_text = {full_text}")    
                    '''
                    <<reasoning>>
                    ...
                    </reasoning>
                    <search>
                    [Web_RAG]: 慢性胃炎 幽门螺杆菌感染 �服药 �疗程
                    </search>
                    '''
                    spans = get_masked_spans_from_text(full_seq, self.tokenizer) 
                    self.masked_spans_per_sample[b].extend(spans)
                    print(f"[model.py] <back> or <sum> masked_spans_per_sample[{b}]: {self.masked_spans_per_sample[b]}")
                    continue

                # 3) Continue or finish
                eos_found = eos_token_id in new_tokens.tolist()
                if not eos_found and (_ < max_generate_iterations - 1):
                    continue_ids = torch.cat([seq[:old_len], new_tokens], dim=0)
                    next_prompts.append(continue_ids)
                else:
                    outputs[b] = seq
                    should_gen[b] = False

            # Prepare next round
            if next_prompts:
                texts = [self.tokenizer.decode(t, skip_special_tokens=False) for t in next_prompts]
                enc = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side="left")
                current_ids = enc.input_ids.to(device)
                current_mask = enc.attention_mask.to(device)
                # print(current_mask.shape)

        final_output = self.prompt_left_generation_right_padding(input_ids, outputs, device, max_length_for_gather)
        
        if calculate_param_importance:
            seq = final_output[0].tolist()
            input_len = input_ids.shape[1]
            response_text = self.tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            
            return response_text, neuron_metric
        else:
            return final_output
    
    # 不着急更新
    # def generate_with_think_interruption_KV_Cache(
    #     self,
    #     input_ids: torch.LongTensor,
    #     attention_mask: Optional[torch.LongTensor],
    #     max_new_tokens: int,
    #     max_length_for_gather: int,
    #     do_sample: bool,
    #     temperature: float,
    #     pad_token_id: int,
    #     eos_token_id: int,
    #     max_generate_iterations: int,
    #     use_diverse_sampling: bool = False,
    #     diversity_penalty: float = 1.0,
    #     calculate_param_importance: bool = False,
    # ) -> torch.LongTensor:
    #     """
    #     Perform iterative generation with tool calls based on markers in text. Use KV Cache to accelerate it.

    #     The loop:
    #       1. Generate tokens.
    #       2. If '</answer>' appears, finalize that sample.
    #       3. If '<search>...</search>' appears, call tool, insert '<observation>' block, and continue.
    #       4. Otherwise, continue generating until EOS or max iterations.

    #     Args:
    #         input_ids (torch.LongTensor): Starting tokens.
    #         attention_mask (Optional[torch.LongTensor]): Attention mask.
    #         max_new_tokens (int): Tokens per generation iteration.
    #         max_length_for_gather (int): Final gather length.
    #         do_sample (bool): Sampling flag.
    #         temperature (float): Sampling temperature.
    #         pad_token_id (int): Padding token ID.
    #         eos_token_id (int): End-of-sequence token ID.
    #         max_generate_iterations (int): Max loop iterations.
    #         use_diverse_sampling (bool): Use diverse sampling.
    #         diversity_penalty (float): Diversity penalty.
    #     Returns:
    #         torch.LongTensor: Padded output IDs for all samples.

    #     Raises:
    #         RuntimeError: On generation or tool-calling failures.
    #     """
    #     assert not calculate_param_importance, "Not Implemented"

    #     device = input_ids.device
    #     batch_size = input_ids.size(0)
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids)

    #     should_gen = torch.ones(batch_size, dtype=torch.bool, device=device)
    #     outputs: List[Optional[torch.LongTensor]] = [None] * batch_size
    #     criteria = StoppingCriteriaList([SearchTagStoppingCriteria(self.tokenizer)])

    #     current_ids = input_ids.clone()
    #     current_mask = attention_mask.clone()

    #     past_key_values: Optional[List[Tuple[torch.Tensor]]] = None

    #     beams_history = [[] for _ in range(batch_size)]
    #     for _ in range(max_generate_iterations):
    #         # Skip leading EOS columns
    #         skip_len = 0
    #         for pos in range(current_ids.size(1)):
    #             if (current_ids[:, pos] == eos_token_id).all():
    #                 skip_len += 1
    #             else:
    #                 break
    #         if skip_len:
    #             current_ids = current_ids[:, skip_len:]
    #             current_mask = current_mask[:, skip_len:]

    #         if not should_gen.any():
    #             break

    #         active = torch.nonzero(should_gen).squeeze(1)

    #         # Drop past_key_values entries for finished samples
    #         if past_key_values is not None:
    #             past_key_values = tuple(tuple(tensor[active] for tensor in layer) for layer in past_key_values)

    #         logits_processor = None
    #         if use_diverse_sampling:
    #             logits_processor = LogitsProcessorList(
    #                 [HammingDiversityLogitsProcessor(beams_history, lambda_penalty=diversity_penalty)]
    #             )

    #         gen_out_dict = self.model.generate(
    #             input_ids=current_ids,
    #             attention_mask=current_mask,
    #             max_new_tokens=max_new_tokens,
    #             do_sample=do_sample,
    #             temperature=temperature,
    #             pad_token_id=pad_token_id,
    #             eos_token_id=eos_token_id,
    #             stopping_criteria=criteria,
    #             use_cache=True,
    #             past_key_values=past_key_values,
    #             return_dict_in_generate=True,
    #             logits_processor=logits_processor,
    #         )

    #         past_key_values = gen_out_dict.past_key_values
    #         gen_out = gen_out_dict.sequences

    #         next_prompts = []

    #         for idx, seq in enumerate(gen_out):
    #             b = active[idx].item()
    #             old_len = current_ids.size(1) - 1
    #             new_tokens = seq[old_len:]
    #             beams_history[b].extend(new_tokens.tolist())
    #             text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

    #             # 1) Answer end
    #             if "</answer>" in text:
    #                 end = text.index("</answer>") + len("</answer>")
    #                 prev = self.tokenizer.decode(seq[:old_len], skip_special_tokens=False)
    #                 final = prev + text[:end]
    #                 outputs[b] = torch.tensor(self.tokenizer.encode(final), device=device)
    #                 should_gen[b] = False
    #                 continue

    #             # 2) Search and observation
    #             if "<search>" in text and "</search>" in text and (_ < max_generate_iterations - 1):
    #                 part = text
    #                 s = part.index("<search>") + len("<search>")
    #                 e = part.index("</search>")
    #                 query = part[s:e].strip()
    #                 try:
    #                     pname, pargs = self.parse_latest_plugin_call(query)
    #                     obs = self.call_plugin(pname, pargs)
    #                 except Exception as exc:
    #                     obs = f"<observation>Error: {exc}"  # preserve flow
    #                 sub = part[: e + len("</search>")]
    #                 merged = self.tokenizer.decode(seq[:old_len], skip_special_tokens=True)
    #                 merged += sub + obs + "\n"
    #                 next_prompts.append(torch.tensor(self.tokenizer.encode(merged), device=device))

    #                 continue

    #             # 3) Continue or finish
    #             eos_found = eos_token_id in new_tokens.tolist()
    #             if not eos_found and (_ < max_generate_iterations - 1):
    #                 continue_ids = torch.cat([seq[:old_len], new_tokens], dim=0)
    #                 next_prompts.append(continue_ids)
    #             else:
    #                 outputs[b] = seq
    #                 should_gen[b] = False

    #         # Prepare next round
    #         if next_prompts:
    #             texts = [self.tokenizer.decode(t, skip_special_tokens=False) for t in next_prompts]
    #             enc = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side="left")
    #             current_ids = enc.input_ids.to(device)
    #             current_mask = enc.attention_mask.to(device)
    #         else:
    #             past_key_values = None  # reset if no prompt continuation

    #     final_output = self.prompt_left_generation_right_padding(input_ids, outputs, device, max_length_for_gather)
    #     return final_output