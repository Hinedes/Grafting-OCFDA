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

source "$VENV/bin/activate"
cd "$REPO/SuperTuning/fine_tuning"

export HF_HOME=/mnt/data/u/$USER/hf
export TRANSFORMERS_CACHE=/mnt/data/u/$USER/hf
export HF_DATASETS_CACHE=/mnt/data/u/$USER/hf/datasets
export WANDB_DIR=/mnt/data/u/$USER/wandb
export TOKENIZERS_PARALLELISM=false

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
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

python launch_math_methods.py \
  --gpus 0,1,2,3 \
  --base_out_dir "$OUT" \
  --base_checkpoint_dir "$CKPT" \
  --models meta-llama/Llama-3.2-1B \
  --lora_rs 8 \
  --seeds 0 \
  --wandb_project super-supra-math \
  --wandb_group math-r8-llama3p2-1b-main \
  --wandb_tags math10k,llama3.2-1b,r8,orix,h200
