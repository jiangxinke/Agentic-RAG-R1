class AgenticRAGModel:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer

    # ------------------ helper: find subsequence start ------------------
    def _find_subsequence_start(self, full_row: torch.LongTensor, subseq: torch.LongTensor) -> Optional[int]:
        if subseq.numel() == 0 or full_row.numel() == 0 or subseq.numel() > full_row.numel():
            return None
        n = full_row.size(0)
        m = subseq.size(0)
        for i in range(n - m + 1):
            if torch.equal(full_row[i : i + m], subseq):
                return i
        return None

    # ------------------ helper: reapply masked spans ------------------
    def _reapply_masked_spans(
        self,
        enc_input_ids: torch.LongTensor,
        base_mask: torch.LongTensor,
        next_prompt_samples: List[int],
        next_prompts_token_tensors: List[torch.LongTensor],
        masked_spans_per_sample: List[List[Tuple[int,int]]],
        enable_2D_attention_mask: bool,
    ):
        device = enc_input_ids.device
        rows, seq_len = enc_input_ids.size(0), enc_input_ids.size(1)
        if not enable_2D_attention_mask:
            for row_idx in range(rows):
                b = next_prompt_samples[row_idx]
                seq_tensor = next_prompts_token_tensors[row_idx].to(device)
                start_pos = self._find_subsequence_start(enc_input_ids[row_idx], seq_tensor)
                if start_pos is None:
                    non_pad_mask = enc_input_ids[row_idx] != self.tokenizer.eos_token_id
                    nonzeros = torch.nonzero(non_pad_mask, as_tuple=True)[0]
                    if len(nonzeros) == 0:
                        continue
                    start_pos = nonzeros[0].item()
                spans = masked_spans_per_sample[b]
                for (s,e) in spans:
                    s_clamped = max(0,s)
                    e_clamped = max(s_clamped,e)
                    apply_s = start_pos + s_clamped
                    apply_e = min(start_pos + e_clamped, seq_len)
                    if apply_s >= apply_e:
                        continue
                    base_mask[row_idx, apply_s:apply_e] = 0
            return base_mask

        attention_mask_2d = base_mask.unsqueeze(1).repeat(1, seq_len, 1).clone()
        for row_idx in range(rows):
            b = next_prompt_samples[row_idx]
            seq_tensor = next_prompts_token_tensors[row_idx].to(device)
            start_pos = self._find_subsequence_start(enc_input_ids[row_idx], seq_tensor)
            if start_pos is None:
                non_pad_mask = enc_input_ids[row_idx] != self.tokenizer.eos_token_id
                nonzeros = torch.nonzero(non_pad_mask, as_tuple=True)[0]
                if len(nonzeros) == 0:
                    continue
                start_pos = nonzeros[0].item()
            spans = masked_spans_per_sample[b]
            for (s,e) in spans:
                s_clamped = max(0,s)
                e_clamped = max(s_clamped,e)
                key_s = start_pos + s_clamped
                key_e = min(start_pos + e_clamped, seq_len)
                if key_s >= key_e:
                    continue
                for qpos in range(key_e, seq_len):
                    attention_mask_2d[row_idx, qpos, key_s:key_e] = 0
        return attention_mask_2d

    # ------------------ generate without KV ------------------
    def generate_with_think_interruption(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        max_new_tokens: int = 32,
        do_sample: bool = True,
        temperature: float = 1.0,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        max_generate_iterations: int = 10,
        enable_2D_attention_mask: bool = True,
    ) -> torch.LongTensor:
        device = input_ids.device
        batch_size = input_ids.size(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).to(device)

        should_gen = torch.ones(batch_size, dtype=torch.bool, device=device)
        outputs: List[Optional[torch.LongTensor]] = [None] * batch_size
        current_ids = input_ids.clone().to(device)
        current_mask_1d = attention_mask.clone().to(device)
        masked_spans_per_sample: List[List[Tuple[int,int]]] = [[] for _ in range(batch_size)]

        for it in range(max_generate_iterations):
            if not should_gen.any():
                break
            active = torch.nonzero(should_gen).squeeze(1)
            current_attention_arg = current_mask_1d.unsqueeze(1).repeat(1, current_ids.size(1), 1) if enable_2D_attention_mask else current_mask_1d
            gen_out = self.model.generate(
                current_ids,
                attention_mask=current_attention_arg,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            next_prompts = []
            next_prompts_samples = []
            next_prompts_token_tensors = []

            for idx, seq in enumerate(gen_out):
                b = active[idx].item()
                old_len = current_ids.size(1)
                new_tokens = seq[old_len:]
                combined_seq = torch.cat([seq[:old_len], new_tokens], dim=0)
                text_new = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

                if ("</backtrack>" in text_new) or ("</summary>" in text_new):
                    token_spans = parse_tokens.locate_action_token_spans(combined_seq, self.tokenizer)
                    # reference spans: any action (think, observation, search)
                    ref_spans = []
                    for k,v in token_spans.items():
                        ref_spans.extend(v)
                    if ref_spans:
                        last_ref = ref_spans[-1]
                        ref_start, ref_end = last_ref[0], last_ref[1]
                        masked_spans_per_sample[b].append((ref_start, ref_end))
                    next_prompts.append(combined_seq)
                    next_prompts_samples.append(b)
                    next_prompts_token_tensors.append(combined_seq.clone())
                    continue

                # continue normally
                next_prompts.append(combined_seq)
                next_prompts_samples.append(b)
                next_prompts_token_tensors.append(combined_seq.clone())

            if next_prompts:
                enc = self.tokenizer(
                    [self.tokenizer.decode(t, skip_special_tokens=False) for t in next_prompts],
                    return_tensors="pt", padding=True, padding_side="left"
                )
                enc_input_ids = enc.input_ids.to(device)
                enc_attention_mask_1d = enc.attention_mask.to(device)
                if enable_2D_attention_mask:
                    enc_attention_mask = self._reapply_masked_spans(
                        enc_input_ids, enc_attention_mask_1d, next_prompts_samples, next_prompts_token_tensors,
                        masked_spans_per_sample, enable_2D_attention_mask=True
                    )
                else:
                    enc_attention_mask = self._reapply_masked_spans(
                        enc_input_ids, enc_attention_mask_1d, next_prompts_samples, next_prompts_token_tensors,
                        masked_spans_per_sample, enable_2D_attention_mask=False
                    )
                current_ids = enc_input_ids
                current_mask_1d = enc_attention_mask_1d
            else:
                break
        return current_ids

    # ------------------ generate with KV ------------------
    def generate_with_think_interruption_kv(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        max_new_tokens: int = 32,
        do_sample: bool = True,
        temperature: float = 1.0,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        max_generate_iterations: int = 10,
        enable_2D_attention_mask: bool = True,
    ) -> torch.LongTensor:
        device = input_ids.device
        batch_size = input_ids.size(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).to(device)

        should_gen = torch.ones(batch_size, dtype=torch.bool, device=device)
        outputs: List[Optional[torch.LongTensor]] = [None] * batch_size
        current_ids = input_ids.clone().to(device)
        current_mask_1d = attention_mask.clone().to(device)
        masked_spans_per_sample: List[List[Tuple[int,int]]] = [[] for _ in range(batch_size)]
        past_key_values = None

        for it in range(max_generate_iterations):
            if not should_gen.any():
                break
            active = torch.nonzero(should_gen).squeeze(1)
            current_attention_arg = current_mask_1d.unsqueeze(1).repeat(1, current_ids.size(1), 1) if enable_2D_attention_mask else current_mask_1d
            gen_out = self.model.generate(
                current_ids,
                attention_mask=current_attention_arg,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                past_key_values=past_key_values,
            )
            next_prompts = []
            next_prompts_samples = []
            next_prompts_token_tensors = []

            for idx, seq in enumerate(gen_out):
                b = active[idx].item()
                old_len = current_ids.size(1)
                new_tokens = seq[old_len:]
                combined_seq = torch.cat([seq[:old_len], new_tokens], dim=0)
                text_new = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

                if ("</backtrack>" in text_new) or ("</summary>" in text_new):
                    token_spans = parse_tokens.locate_action_token_spans(combined_seq, self.tokenizer)
                    ref_spans = []
                    for k,v in token_spans.items():
                        ref_spans.extend(v)
                    if ref_spans:
                        last_ref = ref_spans[-1]
                        ref_start, ref_end = last_ref[0], last_ref[1]
                        masked_spans_per_sample[b].append((ref_start, ref_end))
                        # KV cache handling: zero out past_key_values corresponding to this span
                        if past_key_values is not None:
                            for layer_idx in range(len(past_key_values)):
                                k, v = past_key_values[layer_idx]
                                k[:, :, ref_start:ref_end, :] = 0
                                v[:, :, ref_start:ref_end, :] = 0
                    next_prompts.append(combined_seq)
                    next_prompts_samples.append(b)
                    next_prompts_token_tensors.append(combined_seq.clone())
                    continue

                next_prompts.append(combined_seq)
                next_prompts_samples.append(b)
                next_prompts_token_tensors.append(combined_seq.clone())

            if next_prompts:
                enc = self.tokenizer(
                    [self.tokenizer.decode(t, skip_special_tokens=False) for t in next_prompts],
                    return_tensors="pt", padding=True, padding_side="left"
                )
                enc_input_ids = enc.input_ids.to(device)
                enc_attention_mask_1d = enc.attention_mask.to(device)
                if enable_2D_attention_mask:
                    enc_attention_mask = self._reapply_masked_spans(
                        enc_input_ids, enc_attention_mask_1d, next_prompts_samples, next_prompts_token_tensors,
                        masked_spans_per_sample, enable_2D_attention_mask=True
                    )
                else:
                    enc_attention_mask = self._reapply_masked_spans(
                        enc_input_ids, enc_attention_mask_1d, next_prompts_samples, next_prompts_token_tensors,
                        masked_spans_per_sample, enable_2D_attention_mask=False
                    )
                current_ids = enc_input_ids
                current_mask_1d = enc_attention_mask_1d
        return current_ids
