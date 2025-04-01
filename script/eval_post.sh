source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

export NCCL_P2P_LEVEL=NVL

CUDA_VISIBLE_DEVICES=3 accelerate launch \
    --config_file ./config/eval_multigpu_accelerate_config.yaml \
    --main_process_port 12348 \
    --num_processes 1 \
    --mixed_precision "bf16" \
    eval_post.py