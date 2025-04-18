import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import json5
import torch
from transformers import PreTrainedModel, StoppingCriteria, StoppingCriteriaList

from src.utils.Tools import Tools


class SearchTagStoppingCriteria(StoppingCriteria):
    """
    StoppingCriteria that halts generation when specific end-of-search tags appear.

    This criteria checks the last three generated token IDs against predefined
    sequences corresponding to '</search>' and '</search>\n'. When any sequence
    in the batch matches, generation stops.

    Args:
        tokenizer (Any): Tokenizer used for decoding (provides token IDs).
        think_token (str): The text tag marking the end of a search thought.

    Raises:
        None
    """

    def __init__(
        self,
        tokenizer: Any,
        think_token: str = "</search>",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.target_ids_1: List[int] = tokenizer.encode(think_token)
        self.target_ids_2: List[int] = tokenizer.encode(think_token + "\n")

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        """
        Check if generation should stop based on recently generated tokens.

        Args:
            input_ids (torch.LongTensor): Generated token IDs of shape (batch, seq_len).
            scores (torch.FloatTensor): Model scores (unused here).
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if any sequence ends with a stop tag, else False.
        """
        if input_ids.size(1) < 3:
            return False

        last_tokens = input_ids[:, -3:]
        device = input_ids.device
        t1 = torch.tensor(self.target_ids_1, device=device)
        t2 = torch.tensor(self.target_ids_2, device=device)

        return (last_tokens == t1).all(dim=1).any() or (last_tokens == t2).all(dim=1).any()


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
    ) -> None:
        super().__init__(model.config)
        self.model = model
        self.tokenizer = tokenizer
        self.tool = Tools()

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
            **kwargs: Extra args forwarded to generate.

        Returns:
            torch.LongTensor: Generated token IDs or logits.

        Raises:
            RuntimeError: If model generation fails.
        """
        if not obtain_logits:
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
                **kwargs,
            )
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        max_generate_iterations: int = 8,
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
    ) -> torch.LongTensor:
        """
        Perform iterative generation with tool calls based on markers in text.

        The loop:
          1. Generate tokens.
          2. If '</answer>' appears, finalize that sample.
          3. If '<search>...</search>' appears, call tool, insert '<observation>' block, and continue.
          4. Otherwise, continue generating until EOS or max iterations.

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
        device = input_ids.device
        batch_size = input_ids.size(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        should_gen = torch.ones(batch_size, dtype=torch.bool, device=device)
        outputs: List[Optional[torch.LongTensor]] = [None] * batch_size
        criteria = StoppingCriteriaList([SearchTagStoppingCriteria(self.tokenizer)])

        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()

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
            gen_out = self.model.generate(
                current_ids,
                attention_mask=current_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                stopping_criteria=criteria,
            )
            next_prompts = []

            for idx, seq in enumerate(gen_out):
                b = active[idx].item()
                old_len = current_ids.size(1) - 1
                new_tokens = seq[old_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

                # 1) Answer end
                if "</answer>" in text:
                    end = text.index("</answer>") + len("</answer>")
                    prev = self.tokenizer.decode(seq[:old_len], skip_special_tokens=False)
                    final = prev + text[:end]
                    outputs[b] = torch.tensor(self.tokenizer.encode(final), device=device)
                    should_gen[b] = False
                    continue

                # 2) Search and observation
                if "<search>" in text and "</search>" in text and (_ < max_generate_iterations - 1):
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

        return self.padding_and_truncate(outputs, device, max_length_for_gather)
