# export WANDB_API_KEY=$(cat /slot/sandbox/d/secret/*)
export HF_HOME=/mnt/LLM
export OMP_NUM_THREADS=8

CUDA_VISIBLE_DEVICES=4 python finetune.py \
  --base_model meta-llama/Llama-3.2-1B \
  --data_path 'commonsense_15k.json' \
  --output_dir './trained_models/llama-sift' \
  --save_step 10 \
  --eval_step 10 \
  --batch_size 16 \
  --micro_batch_size 16 \
  --num_epochs 3 \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --compile 0 \
  --seed 2 \
  --sparse_rate 0.013392857142857142 \
  --adapter_name super \
  --max_steps 22 \