#!/bin/bash
#SBATCH --job-name=llama1b_math_r8_main
#SBATCH --partition=pi-richtap
#SBATCH --qos=pi-richtap
#SBATCH --account=pi-richtap
#SBATCH --gpus=nvidia_h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=375G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/data/u/%u/runs/llama1b_math_r8_main-%j.out
#SBATCH --error=/mnt/data/u/%u/runs/llama1b_math_r8_main-%j.err
#SBATCH --signal=B:TERM@60
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-$HOME/dev/super_supra_NeurIPS2025}
VENV=${VENV:-$HOME/.venvs/supertuning}
RUN_ROOT=${RUN_ROOT:-/mnt/data/u/$USER/runs/llama1b_math_r8_main}
CALIBRATION_DATA=${CALIBRATION_DATA:-same_as_train}
MODELS=${MODELS:-meta-llama/Llama-3.2-1B}
LORA_RS=${LORA_RS:-8}
SEEDS=${SEEDS:-0}
DATASETS=${DATASETS:-AddSub,MultiArith,SingleEq,gsm8k,AQuA,SVAMP}
WORKER_GPUS=${WORKER_GPUS:-0,1,2,3}
WANDB_PROJECT=${WANDB_PROJECT:-super-supra-math}
WANDB_GROUP=${WANDB_GROUP:-math-r8-llama3p2-1b-main}
WANDB_TAGS=${WANDB_TAGS:-math10k,llama3.2-1b,r8,orix,h200}

source "$VENV/bin/activate"
cd "$REPO/SuperTuning/fine_tuning"

export PYTHONPATH="$PWD/peft/src:$PWD:$REPO/SuperTuning:${PYTHONPATH:-}"

export HF_HOME=/mnt/data/u/$USER/hf
export TRANSFORMERS_CACHE=/mnt/data/u/$USER/hf
export HF_DATASETS_CACHE=/mnt/data/u/$USER/hf/datasets
export WANDB_DIR=/mnt/data/u/$USER/wandb
export TOKENIZERS_PARALLELISM=false

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

OUT="$RUN_ROOT/out_math_experiments_parallel"
CKPT="$RUN_ROOT/checkpoints_math_parallel"
mkdir -p "$RUN_ROOT" "$OUT" "$CKPT" "$WANDB_DIR" "$HF_HOME"

test -f ft-training_set/math_10k.json
test -f dataset/AddSub/test.json
test -f dataset/MultiArith/test.json
test -f dataset/SingleEq/test.json
test -f dataset/gsm8k/test.json
test -f dataset/AQuA/test.json
test -f dataset/SVAMP/test.json

EXTRA_ARGS=()
if [[ -n "${METHODS:-}" ]]; then
  EXTRA_ARGS+=(--methods "$METHODS")
fi
if [[ -n "${LRS:-}" ]]; then
  EXTRA_ARGS+=(--lrs "$LRS")
fi
if [[ -n "${TRAIN_DATA:-}" ]]; then
  EXTRA_ARGS+=(--train_data "$TRAIN_DATA")
fi
if [[ -n "${TARGET_MODULES:-}" ]]; then
  EXTRA_ARGS+=(--target_modules "$TARGET_MODULES")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--batch_size "$BATCH_SIZE")
fi
if [[ -n "${MICRO_BATCH_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--micro_batch_size "$MICRO_BATCH_SIZE")
fi
if [[ -n "${VAL_SET_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--val_set_size "$VAL_SET_SIZE")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "${WARMUP_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--warmup_steps "$WARMUP_STEPS")
fi
if [[ -n "${NUM_EPOCHS:-}" ]]; then
  EXTRA_ARGS+=(--num_epochs "$NUM_EPOCHS")
fi
if [[ -n "${LR_TUNING_MAX_EXAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--lr_tuning_max_examples "$LR_TUNING_MAX_EXAMPLES")
fi
if [[ -n "${PPL_MAX_EXAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--ppl_max_examples "$PPL_MAX_EXAMPLES")
fi
if [[ -n "${CALIBRATION_NSAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--calibration_nsamples "$CALIBRATION_NSAMPLES")
fi
if [[ -n "${WANDB_MODE:-}" ]]; then
  EXTRA_ARGS+=(--wandb_mode "$WANDB_MODE")
fi

python launch_math_methods.py \
  --gpus "$WORKER_GPUS" \
  --datasets "$DATASETS" \
  --base_out_dir "$OUT" \
  --base_checkpoint_dir "$CKPT" \
  --models "$MODELS" \
  --lora_rs "$LORA_RS" \
  --seeds "$SEEDS" \
  --calibration_data "$CALIBRATION_DATA" \
  --eval_selected_only \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_group "$WANDB_GROUP" \
  --wandb_tags "$WANDB_TAGS" \
  "${EXTRA_ARGS[@]}"
