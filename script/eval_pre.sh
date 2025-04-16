source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

CUDA_VISIBLE_DEVICES=0,1,3,6 accelerate launch \
    --config_file ./config/accelerate_config/eval_multigpu.yaml \
    --main_process_port 12342 \
    --num_processes 4 \
    --mixed_precision "bf16" \
    eval_pre_search.py

