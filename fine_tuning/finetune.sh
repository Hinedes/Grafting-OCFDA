export HF_HOME=/mnt/LLM
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=8
export WANDB_API_KEY=$(cat /slot/sandbox/d/secret/*)

export CUDA_VISIBLE_DEVICES=4,5
WORLD_SIZE=2 torchrun --nproc_per_node=2 --master_port=3192 finetune.py \
  --base_model 'meta-llama/Llama-3.2-1B' \
  --data_path 'commonsense_15k.json' \
  --output_dir './trained_models/llama-lora' \
  --save_step 1000 \
  --eval_step 1000 \
  --batch_size 16 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 1e-5 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --optimizer_name galore_adamw \
  --proj_type power_iteration \
  --compile 0 \
  --use_gradient_checkpointing \
  --weight_decay 0.1 \