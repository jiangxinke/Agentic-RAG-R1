#!/bin/bash
# source /data/xiaobei/anaconda3/etc/profile.d/conda.sh && conda activate r1

export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/neuron/main.py
