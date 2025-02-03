# export HF_HOME=/mnt/LLM

export OMP_NUM_THREADS=8

# export WANDB_API_KEY=$(cat /slot/sandbox/d/secret/*)

WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=3192 finetune.py \
  --base_model 'meta-llama/Llama-3.1-8B' \
  --data_path 'commonsense_15k.json' \
  --output_dir './trained_models/llama-sift' \
  --save_step 1000 \
  --eval_step 1000 \
  --batch_size 16 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 1e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --compile 0 \
  --wandb_project galore_commonsense_8b \
  --wandb_run_name sift-lr_1e-4 \
  --sparse_rate 0.013392857142857142 \
  # --use_gradient_checkpointing \