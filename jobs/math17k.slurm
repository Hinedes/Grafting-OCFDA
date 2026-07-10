#!/bin/bash
#SBATCH --job-name=supertuning-math17k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/math17k-%j.out
#SBATCH --error=logs/math17k-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-$PWD}
CONFIG=${CONFIG:-configs/math17k/llama-1b-1epoch.json}
RUN_ROOT=${RUN_ROOT:-$REPO_DIR/runs/math17k}
GPUS=${GPUS:-0}
PYTHON=${PYTHON:-python}

cd "$REPO_DIR"
mkdir -p logs "$RUN_ROOT/results" "$RUN_ROOT/checkpoints"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXTRA_ARGS=()
if [[ -n "${METHODS:-}" ]]; then
  EXTRA_ARGS+=(--methods "$METHODS")
fi
if [[ -n "${LRS:-}" ]]; then
  EXTRA_ARGS+=(--lrs "$LRS")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  EXTRA_ARGS+=(--wandb_project "$WANDB_PROJECT")
fi

"$PYTHON" -m fine_tuning.launch_math_methods \
  --gpus "$GPUS" \
  --config "$CONFIG" \
  --base_out_dir "$RUN_ROOT/results" \
  --base_checkpoint_dir "$RUN_ROOT/checkpoints" \
  "${EXTRA_ARGS[@]}"
