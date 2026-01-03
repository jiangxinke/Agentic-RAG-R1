#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CFG="${SCRIPT_DIR}/local_config.sh"

if [[ -f "$LOCAL_CFG" ]]; then
  source "$LOCAL_CFG"
fi

BASE_MODEL_PATH="${1:-${MODEL_PATH:-/data2/gjr/models/Qwen2.5-1.5B-Instruct}}"
OUTPUT_MODEL_PATH="${2:-/data2/gjr/models/Qwen2.5-1.5B-Instruct-spr1-mean-v2}"

PY_SCRIPT="${SCRIPT_DIR}/../verl/spr1_special_tokens/sync_tokenizer_and_embeddings.py"

python3 "$PY_SCRIPT" \
  --model-path "$BASE_MODEL_PATH" \
  --output-path "$OUTPUT_MODEL_PATH" \
  --special-tokens "<think>" "</think>" "<tool_call>" "</tool_call>" "<reflect>" "</reflect>" "<answer>" "</answer>" \
  --overwrite \
  "${@:3}"
