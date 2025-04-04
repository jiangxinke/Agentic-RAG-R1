source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --config_file ./config/accelerate_train_zero2_config.yaml \
    --main_process_port 12347 \
    --num_processes 2 \
    --mixed_precision "bf16" \
    train.py