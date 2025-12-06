#!/bin/bash

# export HF_ENDPOINT=https://hf-mirror.com
export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=1,7 accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero2.yaml \
    --main_process_port 12347 \
    --num_processes 2 \
    --mixed_precision "bf16" \
    ./src/train.py