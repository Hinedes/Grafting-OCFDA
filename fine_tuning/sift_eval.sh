export HF_HOME=/mnt/LLM
export OMP_NUM_THREADS=8

mkdir ./result/test_1b_model_sift -p

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset boolq \
    --batch_size 1 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --debug \
    --lora_weights './trained_models/llama-sift' | tee -a './result/test_1b_model_sift/boolq.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset piqa \
    --batch_size 1 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --debug \
    --lora_weights './trained_models/llama-sift' | tee -a './result/test_1b_model_sift/piqa.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset social_i_qa \
    --batch_size 1 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --debug \
    --lora_weights './trained_models/llama-sift' | tee -a './result/test_1b_model_sift/social_i_qa.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset hellaswag \
    --batch_size 1 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --debug \
    --lora_weights './trained_models/llama-sift' | tee -a './result/test_1b_model_sift/hellaswag.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset winogrande \
    --batch_size 1 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --debug \
    --lora_weights './trained_models/llama-sift' | tee -a './result/test_1b_model_sift/winogrande.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset ARC-Easy \
    --batch_size 1 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --debug \
    --lora_weights './trained_models/llama-sift' | tee -a './result/test_1b_model_sift/ARC-Easy.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset ARC-Challenge \
    --batch_size 1 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --debug \
    --lora_weights './trained_models/llama-sift' | tee -a './result/test_1b_model_sift/ARC-Challenge.txt'

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset openbookqa \
    --batch_size 1 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --debug \
    --lora_weights './trained_models/llama-sift' | tee -a './result/test_1b_model_sift/openbookqa.txt'