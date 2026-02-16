#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from infer2 import build_engine, infer_batch


def apply_template(messages: List[Dict[str, str]]) -> str:
    text = ""
    if not messages or (messages and messages[0].get("role") != "system"):
        text += "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


system_prompt = """
You are a helpful and harmless assistant.

# Tools

You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:

<tools>
{"type": "function", "function": {"name": "search", "description": "Searches the web for relevant information based on the given query.", "parameters": {"type": "object", "properties": {"query_list": {"type": "array", "description": "A list of fully-formed semantic queries. The tool will return search results for each query."}}, "required": ["query_list"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""


def prompt_to_messages(prompt_obj: Any) -> List[Dict[str, str]]:
    if isinstance(prompt_obj, np.ndarray):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, list):
        msgs: List[Dict[str, str]] = []
        for m in prompt_obj:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = system_prompt if role == "system" else m.get("content", "")
                msgs.append({"role": str(role), "content": str(content)})
        if msgs:
            return msgs
    if isinstance(prompt_obj, str):
        return [{"role": "user", "content": prompt_obj}]
    return [{"role": "user", "content": str(prompt_obj)}]


def apply_prompt(prompt_obj: Any) -> str:
    return apply_template(prompt_to_messages(prompt_obj))


def extract_user_from_prompt_str(prompt_str: str) -> str:
    pattern = r"<\|im_start\|>user\s*(.*?)<\|im_end\|>"
    matches = re.findall(pattern, prompt_str, flags=re.DOTALL)
    if not matches:
        return ""
    return matches[-1].strip()


def get_ground_truth(row: Dict[str, Any]) -> Optional[str]:
    rm = row.get("reward_model", None)
    if not isinstance(rm, dict):
        return None
    gt = rm.get("ground_truth", None)
    if not isinstance(gt, dict):
        return None
    target = gt.get("target", None)
    if target is None:
        return None
    try:
        t = target.tolist() if hasattr(target, "tolist") else target
    except Exception:
        t = target
    if isinstance(t, (list, tuple)):
        return str(t[0]) if len(t) > 0 else None
    return str(t)


def dump_json_atomic(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-iter", type=int, default=10)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--out-dir", type=str, default="./gen_res")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--compress-before-judge", action="store_true", default=True)
    parser.add_argument("--compress-max-tokens", type=int, default=64)
    parser.add_argument("--eval-llm-timeout", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"gen_{args.name}.json")

    print("[stage] READ_PARQUET_START", flush=True)
    df = pd.read_parquet(args.parquet)
    if args.limit != -1:
        df = df.iloc[: args.limit].copy()
    total = len(df)
    print(f"[worker] Load datasets: len = {total}", flush=True)
    print(f"[stage] READ_PARQUET_DONE total={total}", flush=True)

    results: List[Dict[str, Any]] = []
    total_batches = (total + args.batch_size - 1) // args.batch_size if total > 0 else 0

    lora = args.lora_path.strip() or None
    print("[stage] BUILD_ENGINE_START", flush=True)
    llm = build_engine(args.model_dir, lora, device=args.device)
    print("[stage] BUILD_ENGINE_DONE", flush=True)

    for batch_id in range(total_batches):
        start = batch_id * args.batch_size
        end = min(start + args.batch_size, total)
        batch_df = df.iloc[start:end]

        print(f"[stage] BATCH_START id={batch_id} rows={start}-{end} size={end-start}", flush=True)
        batch_row_idx: List[int] = []
        batch_questions: List[str] = []
        batch_prompts: List[str] = []
        batch_gts: List[str] = []
        batch_data_source: List[Any] = []
        batch_ability: List[Any] = []

        for idx, row in batch_df.iterrows():
            row_dict = row.to_dict()
            prompt_obj = row_dict["prompt"]
            prompt_str = apply_prompt(prompt_obj)
            question = extract_user_from_prompt_str(prompt_str)
            gt = get_ground_truth(row_dict) or ""
            batch_row_idx.append(int(idx))
            batch_questions.append(question)
            batch_prompts.append(prompt_str)
            batch_gts.append(gt)
            batch_data_source.append(row_dict.get("data_source", None))
            batch_ability.append(row_dict.get("ability", None))

        t0 = time.perf_counter()
        print(f"[stage] GENERATE_START batch_id={batch_id} num_iter={args.num_iter}", flush=True)
        outs = infer_batch(llm=llm, prompt=batch_prompts, num_iter=args.num_iter)
        t1 = time.perf_counter()
        batch_time_cost = float(t1 - t0)
        print(f"[stage] GENERATE_DONE batch_id={batch_id} time_sec={batch_time_cost:.2f}", flush=True)

        for j, out in enumerate(outs):
            proc_out = out.get("text", "") or ""
            raw_out = out.get("raw_text", "") or ""
            results.append({
                "question": batch_questions[j],
                "output": raw_out,
                "ground_truth": batch_gts[j],
                "data_source": batch_data_source[j],
                "ability": batch_ability[j],
                "row_index": int(batch_row_idx[j]),
            })

        payload = {
            "name": args.name,
            "batches_done": int(batch_id + 1),
            "total_batches": int(total_batches),
            "summary": {
                "total": int(total),
                "done": int(len(results)),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": results,
        }
        dump_json_atomic(out_path, payload)
        print(f"[stage] DUMP_DONE batch_id={batch_id} path={out_path}", flush=True)

    print("\n==== GEN SUMMARY ====")
    print(f"Total={payload['summary'].get('total', 0)}, Done={payload['summary'].get('done', 0)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
