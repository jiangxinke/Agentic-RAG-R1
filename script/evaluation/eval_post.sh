source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

NUM_EVAL=200
DATE="2025-04-20"
# STEPS="100 95 90 85 80"
STEPS="100"

for STEP in $STEPS; do
    CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
        --config_file ./src/config/accelerate_config/eval_multigpu.yaml \
        --main_process_port 12341 \
        --num_processes 2 \
        ./post.py \
        --date "$DATE" \
        --checkpoint_step $STEP \
        --num_eval $NUM_EVAL
done
