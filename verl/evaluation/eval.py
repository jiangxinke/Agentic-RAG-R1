#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

# 注意：infer2 在子进程中 import（避免主进程加载 sglang/engine）
# from infer2 import build_engine, infer


# -----------------------------
# Prompt template utils
# -----------------------------

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


def prompt_to_messages(prompt_obj: Any) -> List[Dict[str, str]]:
    if isinstance(prompt_obj, np.ndarray):
        prompt_obj = prompt_obj.tolist()

    if isinstance(prompt_obj, list):
        msgs: List[Dict[str, str]] = []
        for m in prompt_obj:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
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


# -----------------------------
# Ground truth & eval utils
# -----------------------------

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


def norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gt: str) -> bool:
    return norm(pred) == norm(gt)


def dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Worker: run eval in subprocess
# -----------------------------

def eval_worker(args_dict: Dict[str, Any], out_queue: "mp.Queue") -> None:
    """
    Entire evaluation runs here (subprocess).
    Create sgl.Engine inside this process and destroy it when process exits.
    """
    # 子进程内 import，避免主进程加载 sglang/engine 相关资源
    from infer2 import build_engine, infer

    model_dir = args_dict["model_dir"]
    parquet = args_dict["parquet"]
    lora_path = args_dict["lora_path"]
    device = args_dict["device"]
    num_iter = int(args_dict["num_iter"])
    limit = int(args_dict["limit"])
    chunk_size = int(args_dict["chunk_size"])

    df = pd.read_parquet(parquet)
    if limit != -1:
        df = df.iloc[:limit].copy()

    total = len(df)
    print(f"[worker] Load datasets: len = {total}", flush=True)

    lora = lora_path.strip() or None
    llm = build_engine(model_dir, lora, device=device)

    results: List[Dict[str, Any]] = []
    correct = 0

    num_chunks = (total + chunk_size - 1) // chunk_size if total > 0 else 0

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min((chunk_id + 1) * chunk_size, total)
        chunk_df = df.iloc[start:end]

        pbar = tqdm(
            total=(end - start),
            desc=f"Chunk {chunk_id + 1}/{num_chunks} [{start}:{end}]",
            ncols=110,
        )

        for idx, row in chunk_df.iterrows():
            row_dict = row.to_dict()

            prompt_obj = row_dict["prompt"]
            prompt_str = apply_prompt(prompt_obj)
            question = extract_user_from_prompt_str(prompt_str)

            gt = get_ground_truth(row_dict) or ""

            out = infer(llm=llm, prompt=prompt_str, num_iter=num_iter)

            proc_out = out.get("text", "")
            raw_out = out.get("raw_text", "")
            pred_ans = out.get("answer", "")
            iters_used = out.get("iters", 0)

            ok = exact_match(pred_ans, gt)
            correct += int(ok)

            done_global = len(results) + 1

            results.append({
                "index": int(len(results)),
                "row_index": int(idx),
                "question": question,
                "ground_truth": gt,
                "raw_output": raw_out,
                "proc_output": proc_out,
                "pred_answer": pred_ans,
                "correct": bool(ok),
                "data_source": row_dict.get("data_source", None),
                "ability": row_dict.get("ability", None),
                "iters": int(iters_used),
                "if_tool_call" : out.get("if_tool_call", False)
            })

            pbar.update(1)
            pbar.set_postfix(
                acc=f"{correct / done_global:.4f}",
                done=f"{done_global}/{total}"
            )

        pbar.close()

    overall_acc = (correct / total) if total > 0 else 0.0

    def group_acc(key: str) -> Dict[str, Dict[str, Any]]:
        stat: Dict[str, Dict[str, Any]] = {}
        for r in results:
            kk = str(r.get(key, ""))
            stat.setdefault(kk, {"count": 0, "correct": 0, "acc": 0.0})
            stat[kk]["count"] += 1
            stat[kk]["correct"] += int(r["correct"])
        for kk, v in stat.items():
            v["acc"] = v["correct"] / v["count"] if v["count"] else 0.0
        return stat

    summary = {
        "model_dir": model_dir,
        "lora_path": lora_path,
        "parquet": parquet,
        "num_iter": num_iter,
        "limit": limit,
        "chunk_size": chunk_size,
        "total": total,
        "correct": correct,
        "overall_acc": overall_acc,
        "acc_by_data_source": group_acc("data_source"),
        "acc_by_ability": group_acc("ability"),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # best-effort engine shutdown (optional; process exit is the real cleanup)
    for name in ("shutdown", "close", "terminate", "stop"):
        fn = getattr(llm, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    payload = {"summary": summary, "results": results}
    out_queue.put(payload)


# -----------------------------
# Main (parent process)
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-iter", type=int, default=10)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--out-dir", type=str, default="./res")
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Spawn worker
    out_queue: mp.Queue = mp.Queue()
    args_dict = {
        "model_dir": args.model_dir,
        "parquet": args.parquet,
        "lora_path": args.lora_path,
        "device": args.device,
        "num_iter": args.num_iter,
        "limit": args.limit,
        "chunk_size": args.chunk_size,
    }

    p = mp.Process(target=eval_worker, args=(args_dict, out_queue), daemon=False)
    p.start()

    payload = None
    try:
        payload = out_queue.get()  # block until worker finishes and puts result
    finally:
        p.join()

    if p.exitcode != 0:
        raise SystemExit(f"[parent] Worker exited with code {p.exitcode}")

    # Save results in parent (avoid worker writing + potential partial files)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"eval_{ts}.json")
    dump_json(out_path, payload)

    summary = payload.get("summary", {})
    print("\n==== SUMMARY ====")
    print(
        f"Total={summary.get('total', 0)}, "
        f"Correct={summary.get('correct', 0)}, "
        f"Overall Acc={summary.get('overall_acc', 0.0):.6f}"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    main()


# system\nYou are a helpful and harmless assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"search\", \"description\": \"Searches the web for relevant information based on the given query.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query_list\": {\"type\": \"array\", \"description\": \"A list of fully-formed semantic queries. The tool will return search results for each query.\"}}, \"required\": [\"query_list\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\nuser\n\n