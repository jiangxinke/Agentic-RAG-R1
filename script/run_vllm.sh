CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ./models/finetuned_Qwen2.5-7B-Instruct \
    --host 127.0.0.1 \
    --port 12333