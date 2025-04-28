source ~/miniconda3/etc/profile.d/conda.sh && conda activate r1

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
    --config_file ./src/config/accelerate_config/eval_multigpu.yaml \
    --main_process_port 12342 \
    --num_processes 4 \
    ./pre_search.py

