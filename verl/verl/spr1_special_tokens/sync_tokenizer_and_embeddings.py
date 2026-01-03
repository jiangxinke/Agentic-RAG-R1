import argparse
import json
import shutil
import struct
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer


# =========================
# utils
# =========================

def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=[
            "<think>", "</think>",
            "<reflect>", "</reflect>",
            "<tool_call>", "</tool_call>",
            "<answer>", "</answer>",
        ],
    )
    parser.add_argument("--overwrite", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text())


def _dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _prepare_output_dir(src_dir: Path, dst_dir: Path, overwrite: bool) -> None:
    if dst_dir.exists():
        if not overwrite and any(dst_dir.iterdir()):
            raise ValueError(f"output-path exists and not empty: {dst_dir}")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def _sync_added_tokens(out_dir: Path, requested_tokens: List[str]):
    toks = _dedupe_preserve_order([t for t in requested_tokens if t])
    tokenizer = AutoTokenizer.from_pretrained(str(out_dir), use_fast=True, local_files_only=True)
    to_add = [t for t in toks if t not in tokenizer.get_vocab()]
    if to_add:
        tokenizer.add_tokens(to_add)
        tokenizer.save_pretrained(str(out_dir))
    return tokenizer, tokenizer.get_vocab()


# =========================
# safetensors helpers
# =========================

def _read_safetensors_header(path: Path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len), object_pairs_hook=OrderedDict)
    return header_len, header


def _dtype_nbytes(dtype: str) -> int:
    if dtype in ("BF16", "F16"):
        return 2
    if dtype in ("F32",):
        return 4
    raise ValueError(dtype)


def _stream_copy_range(src_f, dst_f, offset: int, length: int):
    src_f.seek(offset)
    remaining = length
    while remaining > 0:
        buf = src_f.read(min(64 * 1024 * 1024, remaining))
        if not buf:
            raise IOError("unexpected EOF")
        dst_f.write(buf)
        remaining -= len(buf)


def _bf16_from_f32_bytes(f32_bytes: bytes) -> bytes:
    import numpy as np
    arr = np.frombuffer(f32_bytes, dtype="<f4")
    u32 = arr.view("<u4")
    return (u32 >> 16).astype("<u2").tobytes()


def _rand_row(dtype: str, hidden: int, seed: int, scale: float) -> bytes:
    import numpy as np
    rng = np.random.default_rng(seed)
    f32 = rng.normal(0.0, scale, size=(hidden,)).astype("<f4")
    if dtype == "BF16":
        return _bf16_from_f32_bytes(f32.tobytes())
    if dtype == "F16":
        return f32.astype("<f2").tobytes()
    return f32.tobytes()


def _read_rows(path: Path, tensor: str, rows: list[int]):
    header_len, header = _read_safetensors_header(path)
    info = header[tensor]
    dtype = info["dtype"]
    hidden = info["shape"][1]
    row_bytes = hidden * _dtype_nbytes(dtype)
    base = 8 + header_len + info["data_offsets"][0]

    out = []
    with open(path, "rb") as f:
        for r in rows:
            f.seek(base + r * row_bytes)
            out.append(f.read(row_bytes))
    return dtype, hidden, out


def _mean_rows(dtype: str, rows: list[bytes]) -> bytes:
    import numpy as np
    if dtype == "F32":
        arr = np.stack([np.frombuffer(b, "<f4") for b in rows])
        return arr.mean(0).astype("<f4").tobytes()
    if dtype == "F16":
        arr = np.stack([np.frombuffer(b, "<f2").astype("<f4") for b in rows])
        return arr.mean(0).astype("<f2").tobytes()
    if dtype == "BF16":
        arr = []
        for b in rows:
            u16 = np.frombuffer(b, "<u2")
            arr.append(((u16.astype("<u4")) << 16).view("<f4"))
        mean = np.stack(arr).mean(0)
        return _bf16_from_f32_bytes(mean.astype("<f4").tobytes())
    raise ValueError(dtype)


# =========================
# semantic seed rules
# =========================

SEMANTIC_SEEDS = {
    "<think>": ["think", "reason", "action"],
    "</think>": ["think", "reason", "action", "end"],

    "<reflect>": ["reflect", "summary", "action"],
    "</reflect>": ["reflect", "summary", "action", "end"],

    "<tool_call>": ["tool", "search", "action"],
    "</tool_call>": ["tool", "search", "action", "end"],

    "<answer>": ["answer", "conclusion", "action"],
    "</answer>": ["answer", "conclusion", "action", "end"],
}


# =========================
# embedding rewrite
# =========================

def _rewrite_safetensors_with_vocab_append(
    src: Path,
    dst: Path,
    tensor: str,
    old_vocab: int,
    new_vocab: int,
    initializer_range: float,
    seed: int,
    token_id_map: dict[str, int],
):
    header_len, header = _read_safetensors_header(src)
    info = header[tensor]
    dtype = info["dtype"]
    hidden = info["shape"][1]

    append = new_vocab - old_vocab
    rows_bytes = []

    for i in range(append):
        tid = old_vocab + i
        token = next(k for k, v in token_id_map.items() if v == tid)

        seeds = SEMANTIC_SEEDS.get(token)
        if seeds:
            seed_ids = [token_id_map[s] for s in seeds if s in token_id_map]
            if seed_ids:
                _, _, rows = _read_rows(src, tensor, seed_ids)
                rows_bytes.append(_mean_rows(dtype, rows))
                continue

        rows_bytes.append(_rand_row(dtype, hidden, seed + tid, initializer_range))

    extra = b"".join(rows_bytes)

    new_header = OrderedDict()
    sizes = {}

    for k, v in header.items():
        if k == "__metadata__":
            new_header[k] = v
            continue
        if k == tensor:
            sizes[k] = new_vocab * hidden * _dtype_nbytes(dtype)
            nv = dict(v)
            nv["shape"] = [new_vocab, hidden]
            new_header[k] = nv
        else:
            sizes[k] = v["data_offsets"][1] - v["data_offsets"][0]
            new_header[k] = v

    cursor = 0
    for k, v in new_header.items():
        if k == "__metadata__":
            continue
        v["data_offsets"] = [cursor, cursor + sizes[k]]
        cursor += sizes[k]

    header_bytes = json.dumps(new_header, separators=(",", ":")).encode()
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)

    base = 8 + header_len

    with open(src, "rb") as s, open(dst, "wb") as d:
        d.write(struct.pack("<Q", len(header_bytes)))
        d.write(header_bytes)
        for k, v in header.items():
            if k == "__metadata__":
                continue
            start, end = v["data_offsets"]
            _stream_copy_range(s, d, base + start, end - start)
            if k == tensor:
                d.write(extra)


# =========================
# main
# =========================

def main():
    args = _parse_args()
    base = Path(args.model_path)
    out = Path(args.output_path)

    _prepare_output_dir(base, out, args.overwrite)

    tokenizer, token_id = _sync_added_tokens(out, args.special_tokens)

    config = _load_json(out / "config.json")
    old_vocab = config["vocab_size"]
    new_vocab = max(old_vocab, len(tokenizer))
    config["vocab_size"] = new_vocab
    _dump_json(out / "config.json", config)

    index = _load_json(base / "model.safetensors.index.json")
    shard = index["weight_map"]["model.embed_tokens.weight"]

    if new_vocab > old_vocab:
        _rewrite_safetensors_with_vocab_append(
            src=base / shard,
            dst=out / shard,
            tensor="model.embed_tokens.weight",
            old_vocab=old_vocab,
            new_vocab=new_vocab,
            initializer_range=config.get("initializer_range", 0.02),
            seed=args.seed,
            token_id_map=token_id,
        )


if __name__ == "__main__":
    main()
