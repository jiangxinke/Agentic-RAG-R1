import json
import os
from typing import Optional

import pandas as pd


def export_metrics(metrics_path: str, out_dir: Optional[str] = None):
    if not os.path.exists(metrics_path):
        return
    out_dir = out_dir or os.path.dirname(metrics_path)
    rows = []
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    try:
        df.to_parquet(os.path.join(out_dir, "metrics.parquet"), index=False)
    except Exception:
        pass


def safe_export(metrics_path: str):
    try:
        export_metrics(metrics_path)
    except Exception:
        pass
