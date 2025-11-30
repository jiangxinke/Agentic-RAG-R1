import torch


def expand_to_causal_mask(current_mask: torch.Tensor, dtype=torch.float32):
    if current_mask.ndim != 2:
        raise ValueError(f"current_mask must be 2D (B,T), got {current_mask.ndim}D")
    B, T = current_mask.shape
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be a floating point type, got {dtype}")
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((T, T), fill_value=min_dtype, dtype=dtype, device=current_mask.device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1)
    causal_mask = causal_mask.clone()
    mask_cond = current_mask[:, None, None, :] == 0
    causal_mask = causal_mask.masked_fill(mask_cond, min_dtype)
    return causal_mask


def expand_to_causal_mask_backtrack(current_mask: torch.Tensor, masked_spans_per_sample, dtype=torch.float32):
    B, T = current_mask.shape
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((T, T), fill_value=min_dtype, dtype=dtype, device=current_mask.device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1).clone()
    pad_mask_cond = current_mask[:, None, None, :] == 0
    causal_mask = causal_mask.masked_fill(pad_mask_cond, min_dtype)
    for b in range(B):
        for span in masked_spans_per_sample[b]:
            prev_start, prev_end, backtrack_end = span
            causal_mask[b, 0, prev_start:prev_end+1, backtrack_end+1:] = min_dtype
            causal_mask[b, 0, backtrack_end+1:, prev_start:prev_end+1] = min_dtype
    return causal_mask


def expand_to_causal_mask_parallel(current_mask: torch.Tensor, masked_parallel_spans_per_sample, dtype=torch.float32):
    B, T = current_mask.shape
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((T, T), fill_value=min_dtype, dtype=dtype, device=current_mask.device)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1).clone()
    pad_mask_cond = current_mask[:, None, None, :] == 0
    causal_mask = causal_mask.masked_fill(pad_mask_cond, min_dtype)
    for b in range(B):
        batch_spans = masked_parallel_spans_per_sample[b]
        for rollout_spans in batch_spans:
            for i, span_i in enumerate(rollout_spans):
                start_i, end_i = span_i
                for j, span_j in enumerate(rollout_spans):
                    if i == j:
                        continue
                    start_j, end_j = span_j
                    causal_mask[b, 0, start_i:end_i+1, start_j:end_j+1] = min_dtype
                    causal_mask[b, 0, start_j:end_j+1, start_i:end_i+1] = min_dtype
    return causal_mask


__all__ = [
    "expand_to_causal_mask_backtrack",
    "expand_to_causal_mask_parallel",
    "expand_to_causal_mask",
]
