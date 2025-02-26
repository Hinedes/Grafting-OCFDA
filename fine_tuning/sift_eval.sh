mkdir ./result/model_super -p

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter super \
    --dataset boolq \
    --batch_size 1 \
    --base_model 'unsloth/Llama-3.2-1B' \
    --sparse_rate 0.01171875 \
    --debug \
    --lora_weights './trained_models/llama-super' | tee -a './result/model_super/boolq.txt'

python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter super \
    --dataset piqa \
    --batch_size 1 \
    --base_model 'unsloth/Llama-3.2-1B' \
    --sparse_rate 0.01171875 \
    --debug \
    --lora_weights './trained_models/llama-super' | tee -a './result/model_super/piqa.txt'
