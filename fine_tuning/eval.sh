export HF_HOME=/mnt/LLM
export OMP_NUM_THREADS=8

mkdir ./result/test_8b_model_lora -p

CUDA_VISIBLE_DEVICES=2 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset boolq \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --lora_weights './trained_models/buffer' | tee -a './result/test_8b_model_lora/boolq.txt'

CUDA_VISIBLE_DEVICES=2 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset piqa \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --lora_weights './trained_models/buffer' | tee -a './result/test_8b_model_lora/piqa.txt'

CUDA_VISIBLE_DEVICES=2 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset social_i_qa \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --lora_weights './trained_models/buffer' | tee -a './result/test_8b_model_lora/social_i_qa.txt'

CUDA_VISIBLE_DEVICES=2 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset hellaswag \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --lora_weights './trained_models/buffer' | tee -a './result/test_8b_model_lora/hellaswag.txt'

CUDA_VISIBLE_DEVICES=2 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset winogrande \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --lora_weights './trained_models/buffer' | tee -a './result/test_8b_model_lora/winogrande.txt'

CUDA_VISIBLE_DEVICES=2 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset ARC-Easy \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --lora_weights './trained_models/buffer' | tee -a './result/test_8b_model_lora/ARC-Easy.txt'

CUDA_VISIBLE_DEVICES=2 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset ARC-Challenge \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --lora_weights './trained_models/buffer' | tee -a './result/test_8b_model_lora/ARC-Challenge.txt'

CUDA_VISIBLE_DEVICES=2 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset openbookqa \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.1-8B' \
    --lora_weights './trained_models/buffer' | tee -a './result/test_8b_model_lora/openbookqa.txt'