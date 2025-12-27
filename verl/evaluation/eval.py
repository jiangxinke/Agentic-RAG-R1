import argparse
import re
from collections import defaultdict
from typing import Any, Dict, Optional

import pandas as pd
from infer import ProtocolRunner, GenConfig, SearchConfig

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>\s*$", re.DOTALL)


def extract_answer(text: str) -> Optional[str]:
    m = ANSWER_RE.search(text.strip())
    if not m:
        return None
    return m.group(1).strip()


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


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


def build_prompt(prompt_value: Any) -> Optional[str]:
    if prompt_value is None:
        return None
    if isinstance(prompt_value, str):
        return prompt_value
    try:
        messages = prompt_value.tolist()
    except Exception:
        messages = prompt_value
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content", "")
        if content:
            parts.append(content.strip())

    return "\n".join(parts).strip()



def update_stats(stats: Dict[str, Dict[str, int]], key: str, correct: bool) -> None:
    stats[key]["total"] += 1
    if correct:
        stats[key]["correct"] += 1


def accuracy_from_stats(stats: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    out = {}
    for k, v in stats.items():
        total = v["total"]
        corr = v["correct"]
        out[k] = (corr / total) if total > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="/data2/gjr/models/Qwen2.5-3B-Instruct")
    ap.add_argument("--N", type=int, default=8, help="最大中断次数")

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--no_sample", action="store_true")
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["sdpa", "flash_attention_2", "eager"])

    ap.add_argument("--search_paths", type=int, default=3)
    ap.add_argument("--search_max_new_tokens", type=int, default=96)
    ap.add_argument("--search_temperature", type=float, default=1.0)
    ap.add_argument("--search_top_p", type=float, default=0.5)
    ap.add_argument("--search_repetition_penalty", type=float, default=1.1)
    ap.add_argument("--search_no_repeat_ngram", type=int, default=3)

    ap.add_argument("--parquet_path", type=str, required=True, help="/data2/gjr/workshop/r1/data/searchR1_processed_direct/test.parquet")
    ap.add_argument("--limit", type=int, default=10, help="Only evaluate first N samples; -1 means all")
    ap.add_argument("--print_every", type=int, default=50, help="Progress logging interval")
    ap.add_argument("--dump_errors", type=int, default=0, help="Dump first K mismatches (0 disables)")

    args = ap.parse_args()

    search_cfg = SearchConfig(
        num_paths=args.search_paths,
        max_new_tokens=args.search_max_new_tokens,
        temperature=args.search_temperature,
        top_p=args.search_top_p,
        repetition_penalty=args.search_repetition_penalty,
        no_repeat_ngram_size=args.search_no_repeat_ngram,
    )

    runner = ProtocolRunner(
        model_dir=args.model_dir,
        attn_impl=args.attn_impl,
        search_cfg=search_cfg,
    )

    gen_cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(not args.no_sample),
    )

    df = pd.read_parquet(args.parquet_path)
    if args.limit is not None and args.limit > 0:
        df = df.iloc[: args.limit].copy()

    total = 0
    correct = 0

    by_data_source = defaultdict(lambda: {"total": 0, "correct": 0})
    by_ability = defaultdict(lambda: {"total": 0, "correct": 0})

    dumped = 0

    for i in range(len(df)):
        row = df.iloc[i].to_dict()

        prompt_value = row.get("prompt", None)
        prompt_text = build_prompt(prompt_value)
        gt = get_ground_truth(row)

        data_source = str(row.get("data_source", "UNKNOWN"))
        ability = str(row.get("ability", "UNKNOWN"))

        if prompt_text is None or gt is None:
            continue

        pred_full = runner.Generate(prompt_text, N=args.N, gen_cfg=gen_cfg)
        pred_ans = extract_answer(pred_full)

        if pred_ans is None:
            is_correct = False
        else:
            is_correct = (normalize(pred_ans) == normalize(gt))

        total += 1
        if is_correct:
            correct += 1

        update_stats(by_data_source, data_source, is_correct)
        update_stats(by_ability, ability, is_correct)

        if args.print_every > 0 and total % args.print_every == 0:
            print(f"[{total}/{len(df)}] running_acc={correct/total:.4f}")

        if args.dump_errors > 0 and (not is_correct) and dumped < args.dump_errors:
            dumped += 1
            print("----- MISMATCH -----")
            print(f"index: {row.get('extra_info', {}).get('index', i)}")
            print(f"data_source: {data_source} | ability: {ability}")
            print(f"Q(extra_info.question): {row.get('extra_info', {}).get('question', None)}")
            print(f"GT: {gt}")
            print(f"PRED(<answer>): {pred_ans}")
            print("--------------------")

    overall_acc = (correct / total) if total > 0 else 0.0
    print("\n===== RESULTS =====")
    print(f"Total evaluated: {total}")
    print(f"Overall accuracy: {overall_acc:.6f}")

    print("\n--- Accuracy by data_source ---")
    acc_ds = accuracy_from_stats(by_data_source)
    for k in sorted(acc_ds.keys()):
        v = acc_ds[k]
        t = by_data_source[k]["total"]
        c = by_data_source[k]["correct"]
        print(f"{k:30s}  acc={v:.6f}  ({c}/{t})")

    print("\n--- Accuracy by ability ---")
    acc_ab = accuracy_from_stats(by_ability)
    for k in sorted(acc_ab.keys()):
        v = acc_ab[k]
        t = by_ability[k]["total"]
        c = by_ability[k]["correct"]
        print(f"{k:30s}  acc={v:.6f}  ({c}/{t})")


if __name__ == "__main__":
    main()
