export OMP_NUM_THREADS=8

export WANDB_API_KEY=$(cat /slot/sandbox/d/secret/*)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
# WORLD_SIZE=2 torchrun --nproc_per_node=2 --master_port=3192 finetune.py \
python finetune.py \
  --base_model 'unsloth/Llama-3.2-1B' \
  --data_path 'commonsense_15k.json' \
  --output_dir './trained_models/llama-super' \
  --save_step 10 \
  --eval_step 10 \
  --batch_size 16 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 5e-5 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
  --compile 0 \
  --wandb_project galore_commonsense_1b_TEST \
  --wandb_run_name TEST_EVAL_TO_WANDB \
  --sparse_rate 0.01171875 \
  --adapter_name super \
  --seed 0 \
  --max_steps 12 \