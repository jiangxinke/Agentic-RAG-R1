source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

export NCCL_P2P_LEVEL=NVL

CUDA_VISIBLE_DEVICES=0,2,3,6 accelerate launch \
    --config_file ./config/accelerate_eval_multigpu_config.yaml \
    --main_process_port 12342 \
    --num_processes 4 \
    --mixed_precision "bf16" \
    eval_post.py
