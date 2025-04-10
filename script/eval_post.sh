source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

NUM_EVAL=200
DATE="2025-04-10"
STEPS="10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100"

for STEP in $STEPS; do
    CUDA_VISIBLE_DEVICES=1,7 accelerate launch \
        --config_file ./config/accelerate_config/eval_multigpu.yaml \
        --main_process_port 12349 \
        --num_processes 2 \
        --mixed_precision "bf16" \
        eval_post.py \
        --date "$DATE" \
        --checkpoint_step $STEP \
        --num_eval $NUM_EVAL
done
