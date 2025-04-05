source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

CUDA_VISIBLE_DEVICES=1,2,3,6 accelerate launch \
    --config_file ./config/accelerate_train_zero2_config.yaml \
    --main_process_port 12347 \
    --num_processes 4 \
    --mixed_precision "bf16" \
    train.py