#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=3
export PYTHONWARNINGS="ignore::UserWarning:multiprocessing.resource_tracker"

PROJECT="sp_r1_like_async_rl_after_2026_01_17"
EXP="3B_wo-ole_wo-decoupling_tool-agent_2D-mask_w-naive-process_bs-256_n-16"
STEP="138"

SRC_DIR="/data0/sp/gjr/workshop/sp-r1-v2/verl/checkpoints/"${PROJECT}"/"${EXP}"/global_step_"${STEP}"/actor"
MODEL_DIR=${SRC_DIR}"/output"

PARQUET_PATH="/data0/sp/gjr/workshop/sp-r1-v2/data/searchR1_processed_direct/test.parquet"
LORA_PATH=""
GEN_PY="evaluation/gen.py"

N=20
LIMIT=-1
BATCH_SIZE=64
NAME="${EXP}_step${STEP}_it${N}"

echo "[INFO] Gen started at $(date)"
echo "[INFO] Config: MODEL_DIR=${MODEL_DIR}"
echo "[INFO] Config: PARQUET_PATH=${PARQUET_PATH}"
echo "[INFO] Config: NUM_ITER=${N} BATCH_SIZE=${BATCH_SIZE} LIMIT=${LIMIT} NAME=${NAME}"
echo "[INFO] Env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "[INFO] MODEL_DIR is missing HuggingFace files, merging shards -> ${MODEL_DIR} ..."
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${SRC_DIR}" \
    --target_dir "${MODEL_DIR}"
else
  echo "[INFO] MODEL_DIR has HuggingFace model, skip merging."
fi

wait

echo "[INFO] Launching generation python at $(date)"
python "${GEN_PY}" \
  --model-dir "${MODEL_DIR}" \
  --parquet "${PARQUET_PATH}" \
  --lora-path "${LORA_PATH}" \
  --device "cuda" \
  --num-iter "${N}" \
  --limit "${LIMIT}" \
  --name "${NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --out-dir "evaluation/gen_res" \
  --compress-before-judge \
  --compress-max-tokens "64" \
  --eval-llm-timeout "15"

echo "[INFO] Gen finished at $(date)"

