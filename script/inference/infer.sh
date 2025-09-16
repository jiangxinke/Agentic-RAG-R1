#!/bin/bash
# source /data/xiaobei/anaconda3/etc/profile.d/conda.sh && conda activate r1

export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=5,4,3,2 accelerate launch \
    --config_file ./src/config/accelerate_config/eval_multigpu.yaml \
    --main_process_port 12342 \
    --num_processes 4 \
    ./src/inference.py
