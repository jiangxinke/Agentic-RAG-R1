import argparse
import json
import shutil
import struct
from collections import OrderedDict
from pathlib import Path


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tokens", nargs="*", default=[])
    parser.add_argument("--special-tokens", nargs="*", default=[])
    parser.add_argument("--overwrite", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text())


def _dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _prepare_output_dir(src_dir: Path, dst_dir: Path, overwrite: bool) -> None:
    dst_dir = dst_dir.resolve()
    src_dir = src_dir.resolve()
    if dst_dir == src_dir:
        raise ValueError("output-path must be different from model-path")
    if dst_dir.exists():
        if not overwrite and any(dst_dir.iterdir()):
            raise ValueError(f"output-path exists and not empty: {dst_dir}")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def _read_safetensors_header(path: Path) -> tuple[int, OrderedDict]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len), object_pairs_hook=OrderedDict)
    return header_len, header


def _dtype_nbytes(dtype: str) -> int:
    match dtype:
        case "BF16" | "F16":
            return 2
        case "F32" | "I32" | "U32":
            return 4
        case "F64" | "I64" | "U64":
            return 8
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def _stream_copy_range(src_f, dst_f, offset: int, length: int, chunk_size: int = 64 * 1024 * 1024) -> None:
    src_f.seek(offset)
    remaining = length
    while remaining > 0:
        n = chunk_size if remaining > chunk_size else remaining
        buf = src_f.read(n)
        if not buf:
            raise IOError("unexpected EOF")
        dst_f.write(buf)
        remaining -= n


def _bf16_from_f32_bytes(f32_bytes: bytes) -> bytes:
    import numpy as np

    arr = np.frombuffer(f32_bytes, dtype="<f4")
    u32 = arr.view("<u4")
    u16 = (u32 >> 16).astype("<u2")
    return u16.tobytes()


def _rand_append_bytes(dtype: str, shape: list[int], initializer_range: float, seed: int) -> bytes:
    import numpy as np

    if len(shape) != 2:
        raise ValueError(f"Expected 2D tensor, got {shape}")
    rows, cols = shape
    rng = np.random.default_rng(seed)
    f32 = rng.normal(loc=0.0, scale=initializer_range, size=(rows, cols)).astype("<f4")
    if dtype == "BF16":
        return _bf16_from_f32_bytes(f32.tobytes())
    if dtype == "F16":
        return f32.astype("<f2").tobytes()
    if dtype == "F32":
        return f32.tobytes()
    raise ValueError(f"Unsupported dtype for init: {dtype}")


def _rewrite_safetensors_with_vocab_append(
    src_path: Path,
    dst_path: Path,
    tensor_name: str,
    new_vocab_size: int,
    initializer_range: float,
    seed: int,
) -> dict[str, int]:
    # FIXME: gjr random embedding replace
    header_len, header = _read_safetensors_header(src_path)
    if tensor_name not in header:
        raise ValueError(f"Tensor not found in safetensors: {tensor_name}")

    tinfo = header[tensor_name]
    dtype = tinfo["dtype"]
    old_shape = tinfo["shape"]
    if len(old_shape) != 2:
        raise ValueError(f"Expected 2D embedding, got shape={old_shape}")

    old_vocab = int(old_shape[0])
    hidden = int(old_shape[1])
    if new_vocab_size < old_vocab:
        raise ValueError(f"new_vocab_size({new_vocab_size}) < old_vocab({old_vocab})")

    append_rows = new_vocab_size - old_vocab
    if append_rows == 0:
        shutil.copy2(src_path, dst_path)
        return {"old_file_size": src_path.stat().st_size, "new_file_size": dst_path.stat().st_size}

    extra_bytes = _rand_append_bytes(dtype, [append_rows, hidden], initializer_range, seed)

    new_header: OrderedDict = OrderedDict()
    sizes: dict[str, int] = {}
    for k, v in header.items():
        if k == "__metadata__":
            new_header[k] = v
            continue
        start, end = v["data_offsets"]
        size = int(end - start)
        if k == tensor_name:
            nbytes = (new_vocab_size * hidden) * _dtype_nbytes(dtype)
            sizes[k] = nbytes
            new_v = dict(v)
            new_v["shape"] = [new_vocab_size, hidden]
            new_header[k] = new_v
        else:
            sizes[k] = size
            new_header[k] = v

    cursor = 0
    for k, v in new_header.items():
        if k == "__metadata__":
            continue
        nbytes = sizes[k]
        v["data_offsets"] = [cursor, cursor + nbytes]
        cursor += nbytes

    header_bytes = json.dumps(new_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    if pad:
        header_bytes += b" " * pad

    data_base_in = 8 + header_len

    with open(src_path, "rb") as src_f, open(dst_path, "wb") as dst_f:
        dst_f.write(struct.pack("<Q", len(header_bytes)))
        dst_f.write(header_bytes)
        for k, v in header.items():
            if k == "__metadata__":
                continue
            start, end = v["data_offsets"]
            if k == tensor_name:
                _stream_copy_range(src_f, dst_f, data_base_in + start, end - start)
                dst_f.write(extra_bytes)
            else:
                _stream_copy_range(src_f, dst_f, data_base_in + start, end - start)

    return {"old_file_size": src_path.stat().st_size, "new_file_size": dst_path.stat().st_size}


def _update_tokenizer_files(
    base_model_dir: Path,
    output_dir: Path,
    tokens: list[str],
    special_tokens: list[str],
    vocab_size_before: int,
) -> dict[str, object]:
    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer_cfg_path = output_dir / "tokenizer_config.json"

    tok = _load_json(tokenizer_json_path)
    tok_cfg = _load_json(tokenizer_cfg_path)

    added_tokens: list[dict] = tok.get("added_tokens", [])
    added_by_content = {x["content"]: x for x in added_tokens}
    model_vocab = tok.get("model", {}).get("vocab", {})
    exists = set(model_vocab.keys()) | set(added_by_content.keys())

    tokens = [t for t in (tokens or []) if t]
    special_tokens = [t for t in (special_tokens or []) if t]
    special_tokens = _dedupe_preserve_order(special_tokens)

    new_ids_start = int(vocab_size_before)
    next_id = new_ids_start

    added_tokens_decoder = tok_cfg.get("added_tokens_decoder") or {}
    additional_special_tokens = tok_cfg.get("additional_special_tokens") or []

    newly_added = []
    for t in tokens:
        if t in exists:
            continue
        entry = {
            "id": next_id,
            "content": t,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": False,
        }
        added_tokens.append(entry)
        added_tokens_decoder[str(next_id)] = {k: entry[k] for k in entry if k != "id"}
        exists.add(t)
        newly_added.append(t)
        next_id += 1

    added_by_content = {x["content"]: x for x in added_tokens}

    newly_added_special = []
    for t in special_tokens:
        if t in exists:
            if t in added_by_content:
                added_by_content[t]["special"] = True
                tid = added_by_content[t]["id"]
                if str(tid) in added_tokens_decoder:
                    added_tokens_decoder[str(tid)]["special"] = True
                else:
                    added_tokens_decoder[str(tid)] = {
                        "content": t,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
            if t not in additional_special_tokens:
                additional_special_tokens.append(t)
            continue

        entry = {
            "id": next_id,
            "content": t,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        added_tokens.append(entry)
        added_tokens_decoder[str(next_id)] = {k: entry[k] for k in entry if k != "id"}
        additional_special_tokens.append(t)
        exists.add(t)
        newly_added_special.append(t)
        next_id += 1

    tok["added_tokens"] = added_tokens
    tok_cfg["added_tokens_decoder"] = added_tokens_decoder
    tok_cfg["additional_special_tokens"] = _dedupe_preserve_order(additional_special_tokens)

    _dump_json(tokenizer_json_path, tok)
    _dump_json(tokenizer_cfg_path, tok_cfg)

    return {
        "added_tokens": newly_added,
        "added_special_tokens": newly_added_special,
        "marked_as_special": [t for t in special_tokens if t in exists and t not in newly_added_special],
        "new_vocab_size": next_id,
    }


def main() -> None:
    args = _parse_args()

    base_dir = Path(args.model_path)
    output_dir = Path(args.output_path)

    config_path = base_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {base_dir}")
    base_config = _load_json(config_path)
    initializer_range = float(base_config.get("initializer_range", 0.02))

    index_path = base_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors.index.json under {base_dir}")
    index_obj = _load_json(index_path)
    weight_map = index_obj.get("weight_map") or {}
    if "model.embed_tokens.weight" not in weight_map:
        raise ValueError("weight_map missing model.embed_tokens.weight")
    embed_shard_name = weight_map["model.embed_tokens.weight"]

    base_embed_path = base_dir / embed_shard_name
    _, embed_header = _read_safetensors_header(base_embed_path)
    embed_info = embed_header["model.embed_tokens.weight"]
    old_vocab_size = int(embed_info["shape"][0])

    _prepare_output_dir(base_dir, output_dir, overwrite=args.overwrite)

    tok_update = _update_tokenizer_files(
        base_model_dir=base_dir,
        output_dir=output_dir,
        tokens=args.tokens,
        special_tokens=args.special_tokens,
        vocab_size_before=old_vocab_size,
    )

    new_vocab_size = int(tok_update["new_vocab_size"])

    out_config_path = output_dir / "config.json"
    out_config = _load_json(out_config_path)
    out_config["vocab_size"] = new_vocab_size
    _dump_json(out_config_path, out_config)

    out_embed_path = output_dir / embed_shard_name
    rewrite_stats = _rewrite_safetensors_with_vocab_append(
        src_path=base_embed_path,
        dst_path=out_embed_path,
        tensor_name="model.embed_tokens.weight",
        new_vocab_size=new_vocab_size,
        initializer_range=initializer_range,
        seed=args.seed,
    )

    out_index_path = output_dir / "model.safetensors.index.json"
    out_index = _load_json(out_index_path)
    meta = out_index.get("metadata") or {}
    if "total_size" in meta:
        delta = int(rewrite_stats["new_file_size"]) - int(rewrite_stats["old_file_size"])
        meta["total_size"] = int(meta["total_size"]) + delta
        out_index["metadata"] = meta
    _dump_json(out_index_path, out_index)

    _, out_embed_header = _read_safetensors_header(out_embed_path)
    out_shape = out_embed_header["model.embed_tokens.weight"]["shape"]
    print(
        {
            "model_path": str(base_dir),
            "output_path": str(output_dir),
            "old_vocab_size": old_vocab_size,
            "new_vocab_size": new_vocab_size,
            "tokenizer_update": tok_update,
            "embed_shape": out_shape,
        }
    )


if __name__ == "__main__":
    main()
