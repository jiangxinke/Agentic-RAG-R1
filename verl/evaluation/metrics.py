#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import csv
from typing import Any, Dict, List


def dump_json_atomic(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def normalize_answer(s: Any) -> str:
    import string
    if s is None:
        return ""
    s = str(s)
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text: str) -> str:
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_prf1(pred: Any, gt: Any) -> Dict[str, float]:
    pred_tokens = normalize_answer(str(pred)).split()
    gt_tokens = normalize_answer(str(gt)).split()
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    import collections
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    if precision + recall == 0:
        return {"precision": precision, "recall": recall, "f1": 0.0}
    f1 = (2 * precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text or "", flags=re.DOTALL)
    return m.group(1) if m else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="evaluation/metrics_res")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--answer-only", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])

    name = args.name or os.path.splitext(os.path.basename(args.input))[0].replace("gen_", "")
    out_path = os.path.join(args.out_dir, f"metrics_{name}.json")

    total = len(results)
    em_sum = 0.0
    f1_sum = 0.0
    em_sum_f = 0.0
    f1_sum_f = 0.0
    filtered = 0

    out_items: List[Dict[str, Any]] = []
    for i, r in enumerate(results):
        q = r.get("question", "")
        raw = r.get("output", "") or ""
        gt = r.get("ground_truth", "") or ""
        ds = r.get("data_source", "")
        ab = r.get("ability", "")
        ridx = r.get("row_index", None)

        cand = extract_answer(raw) if args.answer_only else raw
        em = 1.0 if normalize_answer(cand) == normalize_answer(gt) else 0.0
        prf = compute_prf1(cand, gt)

        em_sum += em
        f1_sum += prf["f1"]

        has_ans = bool(cand.strip())
        if has_ans:
            filtered += 1
            em_sum_f += em
            f1_sum_f += prf["f1"]

        out_items.append({
            "index": int(i),
            "row_index": ridx,
            "question": q,
            "answer": cand,
            "ground_truth": gt,
            "em": float(em),
            "precision": float(prf["precision"]),
            "recall": float(prf["recall"]),
            "f1": float(prf["f1"]),
            "data_source": ds,
            "ability": ab,
        })

    summary = {
        "total": int(total),
        "done": int(total),
        "overall_em": (em_sum / total) if total > 0 else 0.0,
        "avg_f1": (f1_sum / total) if total > 0 else 0.0,
        "filtered_count": int(filtered),
        "avg_em_filter": (em_sum_f / filtered) if filtered > 0 else 0.0,
        "avg_f1_filter": (f1_sum_f / filtered) if filtered > 0 else 0.0,
        "answer_only": bool(args.answer_only),
        "source_input": str(args.input),
    }

    payload = {
        "name": name,
        "summary": summary,
        "items": out_items,
    }
    dump_json_atomic(out_path, payload)

    print("[metrics] DONE")
    print(f"Total={summary['total']} Done={summary['done']} Filtered={summary['filtered_count']} OverallEM={summary['overall_em']:.6f} AvgF1={summary['avg_f1']:.6f} EM_filter={summary['avg_em_filter']:.6f} F1_filter={summary['avg_f1_filter']:.6f}")
    print(f"Saved: {out_path}")

    groups: Dict[str, Dict[str, float]] = {}
    for it in out_items:
        ds = str(it.get("data_source", ""))
        g = groups.setdefault(ds, {"count": 0, "filtered": 0, "em_sum": 0.0, "f1_sum": 0.0, "emf_sum": 0.0, "f1f_sum": 0.0})
        g["count"] += 1
        g["em_sum"] += float(it.get("em", 0.0))
        g["f1_sum"] += float(it.get("f1", 0.0))
        has_ans = bool(str(it.get("answer", "")).strip())
        if has_ans:
            g["filtered"] += 1
            g["emf_sum"] += float(it.get("em", 0.0))
            g["f1f_sum"] += float(it.get("f1", 0.0))

    groups_out: Dict[str, Dict[str, float]] = {}
    for ds, g in groups.items():
        cnt = int(g["count"])
        filt = int(g["filtered"])
        groups_out[ds] = {
            "count": cnt,
            "filtered_count": filt,
            "overall_em": (g["em_sum"] / cnt) if cnt > 0 else 0.0,
            "avg_f1": (g["f1_sum"] / cnt) if cnt > 0 else 0.0,
            "em_filter": (g["emf_sum"] / filt) if filt > 0 else 0.0,
            "f1_filter": (g["f1f_sum"] / filt) if filt > 0 else 0.0,
        }

    payload2 = {"name": name, "by_data_source": groups_out}
    out_path_groups = os.path.join(args.out_dir, f"metrics_{name}_by_data_source.json")
    dump_json_atomic(out_path_groups, payload2)
    print(f"Saved: {out_path_groups}")

    csv_path = os.path.join(args.out_dir, f"metrics_{name}_by_data_source.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["data_source", "count", "filtered_count", "overall_em", "avg_f1", "em_filter", "f1_filter"])
        for ds, v in groups_out.items():
            w.writerow([ds, int(v["count"]), int(v["filtered_count"]), float(v["overall_em"]), float(v["avg_f1"]), float(v["em_filter"]), float(v["f1_filter"])])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
