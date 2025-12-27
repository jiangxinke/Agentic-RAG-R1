#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=4

MODEL_DIR="/data2/gjr/models/Qwen2.5-3B-Instruct"
PARQUET_PATH="/data2/gjr/workshop/r1/data/searchR1_processed_direct/test.parquet"

EVAL_PY="eval.py"

# 推理相关配置
N=8 # 中断次数
MAX_NEW_TOKENS=256
TEMPERATURE=0.7
TOP_P=0.9
ATTN_IMPL="flash_attention_2"

# 工具调用相关配置
SEARCH_PATHS=3
SEARCH_MAX_NEW_TOKENS=96
SEARCH_TEMPERATURE=1.5
SEARCH_TOP_P=0.5
SEARCH_REPETITION_PENALTY=1.1
SEARCH_NO_REPEAT_NGRAM=3

# 测试样本
LIMIT=10          # -1 表示全量
PRINT_EVERY=50
DUMP_ERRORS=0     # >0 会打印前 K 个错误样本

NO_SAMPLE="--no_sample"

python "${EVAL_PY}" \
  --model_dir "${MODEL_DIR}" \
  --parquet_path "${PARQUET_PATH}" \
  --N "${N}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  ${NO_SAMPLE} \
  --attn_impl "${ATTN_IMPL}" \
  --search_paths "${SEARCH_PATHS}" \
  --search_max_new_tokens "${SEARCH_MAX_NEW_TOKENS}" \
  --search_temperature "${SEARCH_TEMPERATURE}" \
  --search_top_p "${SEARCH_TOP_P}" \
  --search_repetition_penalty "${SEARCH_REPETITION_PENALTY}" \
  --search_no_repeat_ngram "${SEARCH_NO_REPEAT_NGRAM}" \
  --limit "${LIMIT}" \
  --print_every "${PRINT_EVERY}" \
  --dump_errors "${DUMP_ERRORS}"
