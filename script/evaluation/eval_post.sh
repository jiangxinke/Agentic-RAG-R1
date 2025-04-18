source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

NUM_EVAL=100
DATE="2025-04-12"
STEPS="150"

for STEP in $STEPS; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --config_file ./src/config/accelerate_config/eval_multigpu.yaml \
        --main_process_port 12349 \
        --num_processes 2 \
        ./src/evaluation/eval_post.py \
        --date "$DATE" \
        --checkpoint_step $STEP \
        --num_eval $NUM_EVAL
done
