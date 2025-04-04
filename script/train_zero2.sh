source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

export NCCL_P2P_LEVEL=NVL
export ELASTIC_SEARCH_URL="https://123.57.228.132:8285"
export ELASTIC_SEARCH_PASSWORD="Qvw_toYzUTZkwmYaT5bC"

CUDA_VISIBLE_DEVICES=2 accelerate launch \
    --config_file ./config/accelerate_train_zero2_config.yaml \
    --main_process_port 12347 \
    --num_processes 1 \
    --mixed_precision "bf16" \
    train.py