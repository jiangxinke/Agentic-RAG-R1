#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=6,7
export PYTHONWARNINGS="ignore::UserWarning:multiprocessing.resource_tracker"

NAME="sp_r1_like_async_rl_after_2026_01_03/3B_w-ole-0.6_wo-decoupling_tool-agent_2D-mask_w-naive-process_bs16"
STEP="300"
SRC_DIR="checkpoints/${NAME}/global_step_${STEP}/actor"
MODEL_DIR=${SRC_DIR}"/output"

PARQUET_PATH="data/searchR1_processed_direct/test.parquet"
LORA_PATH=""

if [ ! -d "${MODEL_DIR}" ]; then
  echo "[INFO] MODEL_DIR not found, merging model..."
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${SRC_DIR}" \
    --target_dir "${MODEL_DIR}"
else
  echo "[INFO] MODEL_DIR exists, skip merging."
fi

wait

EVAL_PY="evaluation/eval.py"

N=10 # number of iterations
SEARCH_PATHS=1
LIMIT=16 # -1 for all

python "${EVAL_PY}" \
  --model-dir "${MODEL_DIR}" \
  --parquet "${PARQUET_PATH}" \
  --lora-path "${LORA_PATH}" \
  --device "cuda" \
  --num-iter "${N}" \
  --limit "${LIMIT}" \
  --out-dir "evaluation/res"
