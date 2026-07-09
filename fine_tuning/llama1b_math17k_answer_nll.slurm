#!/bin/bash
#SBATCH --job-name=llama1b_answer_nll
#SBATCH --partition=freecycle
#SBATCH --qos=freecycle
#SBATCH --gpus=nvidia_h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/data/u/%u/runs/llama1b_answer_nll-%j.out
#SBATCH --error=/mnt/data/u/%u/runs/llama1b_answer_nll-%j.err
#SBATCH --signal=B:TERM@60
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-$HOME/dev/super_supra_NeurIPS2025}
VENV=${VENV:-$HOME/.venvs/supertuning}
RUN_ROOT=${RUN_ROOT:-/mnt/data/u/$USER/runs/llama1b_math17k_answer_only_nll}
MODELS=${MODELS:-meta-llama/Llama-3.2-1B}
LORA_RS=${LORA_RS:-8}
SEEDS=${SEEDS:-0}
DATASETS=${DATASETS:-Math17K}
WORKER_GPUS=${WORKER_GPUS:-0,1,2,3}
TRAIN_DATA=${TRAIN_DATA:-ft-training_set/math_17k.json}
PPL_EVAL_DATA=${PPL_EVAL_DATA:-$TRAIN_DATA}
PPL_EVAL_NAME=${PPL_EVAL_NAME:-Math17K}
CALIBRATION_DATA=${CALIBRATION_DATA:-c4}
NUM_EPOCHS=${NUM_EPOCHS:-1}
WARMUP_STEPS=${WARMUP_STEPS:-100}
METHOD_LRS=${METHOD_LRS:-lora:1e-3,rosa:5e-4,sift-topk:1e-4,magnitude-bottomk:1e-3,super-wanda-bottom:1e-3}
WANDB_PROJECT=${WANDB_PROJECT:-super-supra-math}
WANDB_GROUP=${WANDB_GROUP:-math17k-answer-nll-llama1b-1epoch}
WANDB_TAGS=${WANDB_TAGS:-math17k,llama3.2-1b,r8,answer-nll,1epoch,orix,h200}

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

OUT="$RUN_ROOT/out_answer_only_nll"
CKPT="$RUN_ROOT/checkpoints_answer_only_nll"
mkdir -p "$RUN_ROOT" "$OUT" "$CKPT" "$WANDB_DIR" "$HF_HOME"

test -f "$TRAIN_DATA"
test -f "$PPL_EVAL_DATA"

EXTRA_ARGS=()
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

python run_answer_only_nll_methods.py \
  --gpus "$WORKER_GPUS" \
  --method_lrs "$METHOD_LRS" \
  --datasets "$DATASETS" \
  --base_out_dir "$OUT" \
  --base_checkpoint_dir "$CKPT" \
  --models "$MODELS" \
  --lora_rs "$LORA_RS" \
  --seeds "$SEEDS" \
  --train_data "$TRAIN_DATA" \
  --ppl_eval_data "$PPL_EVAL_DATA" \
  --ppl_eval_name "$PPL_EVAL_NAME" \
  --calibration_data "$CALIBRATION_DATA" \
  --num_epochs "$NUM_EPOCHS" \
  --warmup_steps "$WARMUP_STEPS" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_group "$WANDB_GROUP" \
  --wandb_tags "$WANDB_TAGS" \
  "${EXTRA_ARGS[@]}"
