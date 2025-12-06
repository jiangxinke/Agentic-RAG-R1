#!/bin/bash

set -ex

GPUs="1,2"

# export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=$GPUs
export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1

number_of_gpus=$(echo $GPUs | tr ',' '\n' | wc -l)

accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero2.yaml \
    --main_process_port 12347 \
    --num_processes $number_of_gpus \
    --mixed_precision "bf16" \
    ./src/train.py