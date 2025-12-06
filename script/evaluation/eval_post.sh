#!/bin/bash

NUM_EVAL=100
DATE="2025-09-19"
STEPS="100"

export PYTHONPATH=$PYTHONPATH:/home/xiaobei/jxk/agentic-rag-r1/Agentic-RAG-R1

echo "Starting post-training evaluation..."
echo "Date: $DATE"
echo "Steps: $STEPS"
echo "Number of evaluations: $NUM_EVAL"

for STEP in $STEPS; do  
    echo "Evaluating checkpoint step: $STEP"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
        --config_file ./src/config/accelerate_config/eval_multigpu.yaml \
        --main_process_port 12342 \
        --num_processes 8 \
        ./src/evaluation/unified_eval.py \
        --mode post \
        --date "$DATE" \
        --checkpoint_step $STEP \
        --num_eval $NUM_EVAL
    
    echo "Completed evaluation for step $STEP"
done

echo "All post-training evaluations completed!"
