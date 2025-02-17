mkdir ./result/model_lora -p

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter super \
    --dataset boolq \
    --batch_size 1 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --sparse_rate 0.011363636363636364 \
    --lora_weights './trained_models/llama-sift' | tee -a './result/model_lora/boolq.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter super \
    --dataset piqa \
    --batch_size 1 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --sparse_rate 0.011363636363636364 \
    --lora_weights './trained_models/llama-sift' | tee -a './result/model_lora/piqa.txt'
