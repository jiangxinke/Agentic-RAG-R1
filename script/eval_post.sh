source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

NUM_EVAL=100
DATE="2025-04-12"
STEPS="150"

for STEP in $STEPS; do
    CUDA_VISIBLE_DEVICES=0,1,3,6 accelerate launch \
        --config_file ./config/accelerate_config/eval_multigpu.yaml \
        --main_process_port 12349 \
        --num_processes 4 \
        --mixed_precision "bf16" \
        eval_post.py \
        --date "$DATE" \
        --checkpoint_step $STEP \
        --num_eval $NUM_EVAL
done
