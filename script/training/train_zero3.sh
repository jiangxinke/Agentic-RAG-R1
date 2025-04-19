source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL

CUDA_VISIBLE_DEVICES=1,2 accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero3.yaml \
    --main_process_port 12347 \
    --num_processes 2 \
    --mixed_precision "fp16" \
    ./src/train.py