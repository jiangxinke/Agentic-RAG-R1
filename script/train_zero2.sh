source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

export NCCL_P2P_LEVEL=NVL

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
    --config_file ./config/accelerate_config/train_zero2.yaml \
    --main_process_port 12347 \
    --num_processes 7 \
    --mixed_precision "fp16" \
    train.py