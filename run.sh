source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1


CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --config_file ./config/accelerate_config/train_zero2.yaml \
    --main_process_port 12348 \
    --num_processes 2 \
    --mixed_precision "bf16" \
    test.py