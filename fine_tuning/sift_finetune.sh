export HF_HOME=/mnt/LLM
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

python finetune.py \
  --base_model 'meta-llama/Llama-3.2-1B' \
  --data_path 'commonsense_15k.json' \
  --output_dir './trained_models/llama-sift' \
  --save_step 10 \
  --eval_step 10 \
  --batch_size 16 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --adapter_name sift \
  --compile 0 \
  --max_steps 12 \
  # --attn_implementation flash_attention_2 \