#!/bin/bash
#SBATCH --job-name=profile_efficiency
#SBATCH --partition=freecycle
#SBATCH --qos=freecycle
#SBATCH --gpus=nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/data/u/%u/runs/profile_efficiency-%j.out
#SBATCH --error=/mnt/data/u/%u/runs/profile_efficiency-%j.err

set -euo pipefail

REPO=${REPO:-$HOME/dev/SuperTuning}
VENV=${VENV:-$HOME/.venvs/supertuning}
RUN_ROOT=${RUN_ROOT:-/mnt/data/u/$USER/runs/efficiency_profile_math17k_r8_h200}
MODELS=${MODELS:-meta-llama/Llama-3.2-1B,meta-llama/Meta-Llama-3-8B}
METHODS=${METHODS:-full,rosa,sift-topk,lora,super-wanda-bottom,magnitude-bottomk,supra-0.8-bottom,supra-magnitude-0.3}
TRAIN_DATA=${TRAIN_DATA:-ft-training_set/math_17k.json}
CALIBRATION_DATA=${CALIBRATION_DATA:-c4}
CALIBRATION_NSAMPLES=${CALIBRATION_NSAMPLES:-128}
PROFILE_STEPS=${PROFILE_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
WARMUP_STEPS=${WARMUP_STEPS:-100}

source "$VENV/bin/activate"

if [[ -d "$REPO/SuperTuning/fine_tuning" ]]; then
  REPO_DIR="$REPO/SuperTuning"
else
  REPO_DIR="$REPO"
fi

cd "$REPO_DIR/fine_tuning"

export PYTHONPATH="$PWD:$REPO_DIR:${PYTHONPATH:-}"
export HF_HOME=/mnt/data/u/$USER/hf
export TRANSFORMERS_CACHE=/mnt/data/u/$USER/hf
export HF_DATASETS_CACHE=/mnt/data/u/$USER/hf/datasets
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

OUT="$RUN_ROOT/out_efficiency_profile"
mkdir -p "$RUN_ROOT" "$OUT" "$HF_HOME" "$HF_DATASETS_CACHE"

test -f "$TRAIN_DATA"

python profile_efficiency.py \
  --models "$MODELS" \
  --methods "$METHODS" \
  --output_dir "$OUT" \
  --train_data "$TRAIN_DATA" \
  --calibration_data "$CALIBRATION_DATA" \
  --calibration_nsamples "$CALIBRATION_NSAMPLES" \
  --profile_steps "$PROFILE_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --micro_batch_size "$MICRO_BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --warmup_steps "$WARMUP_STEPS"
