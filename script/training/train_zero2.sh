#!/bin/bash
source /data/xiaobei/anaconda3/etc/profile.d/conda.sh && conda activate r1

export HF_ENDPOINT=https://hf-mirror.com
export NCCL_P2P_LEVEL=NVL

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero2.yaml \
    --main_process_port 12347 \
    --num_processes 2 \
    --mixed_precision "fp16" \
    ./src/train.py