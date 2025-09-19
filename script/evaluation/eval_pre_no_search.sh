#!/bin/bash

NUM_EVAL=200
export PYTHONPATH=$PYTHONPATH:/home/xiaobei/jxk/agentic-rag-r1/Agentic-RAG-R1

echo "Starting pre-training evaluation (no search)..."
echo "Number of evaluations: $NUM_EVAL"

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --config_file ./src/config/accelerate_config/eval_multigpu.yaml \
    --main_process_port 12342 \
    --num_processes 6 \
    ./src/evaluation/unified_eval.py \
    --mode pre_no_search \
    --num_eval $NUM_EVAL

echo "Pre-training evaluation (no search) completed!"

