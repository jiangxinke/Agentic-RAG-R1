#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1


NAME=gen_3B_wo-ole_wo-decoupling_tool-agent_2D-mask_w-naive-process_bs-256_n-16_step138_it20
gen_res="evaluation/gen_res/${NAME}.json"

METRICS_PY="evaluation/metrics.py"


python "${METRICS_PY}" --input "${gen_res}"

