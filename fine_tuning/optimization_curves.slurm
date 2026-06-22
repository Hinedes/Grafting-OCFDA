#!/bin/bash
#SBATCH --job-name=optimization_curves
#SBATCH --partition=pi-richtap
#SBATCH --qos=pi-richtap
#SBATCH --account=pi-richtap
#SBATCH --gpus=nvidia_h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=375G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/data/u/%u/runs/optimization_curves-%j.out
#SBATCH --error=/mnt/data/u/%u/runs/optimization_curves-%j.err
#SBATCH --signal=B:TERM@60
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-$HOME/dev/super_supra_NeurIPS2025}
VENV=${VENV:-$HOME/.venvs/supertuning}
RUN_ROOT=${RUN_ROOT:-/mnt/data/u/$USER/runs/optimization_curves}

MODELS=${MODELS:-meta-llama/Llama-3.2-1B}
METHODS=${METHODS:-lora,rosa,sift-topk,sift-rand,super-rand,super-wanda-bottom,supra-0.5-bottom}
PRESET=${PRESET:-full_epoch}
LORA_RS=${LORA_RS:-8}
SEEDS=${SEEDS:-0}
TRAIN_DATA=${TRAIN_DATA:-ft-training_set/math_10k.json}
CALIBRATION_DATA=${CALIBRATION_DATA:-c4}
NUM_EPOCHS=${NUM_EPOCHS:-3}
WARMUP_STEPS=${WARMUP_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-10}
CURVE_GPUS=${CURVE_GPUS:-0,1,2,3}

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

OUT="$RUN_ROOT/out_optimization_curves"
CKPT="$RUN_ROOT/checkpoints_optimization_curves"
mkdir -p "$RUN_ROOT" "$OUT" "$CKPT" "$WANDB_DIR" "$HF_HOME"

test -f "$TRAIN_DATA"

EXTRA_ARGS=()
if [[ -n "${METHOD_LRS:-}" ]]; then
  EXTRA_ARGS+=(--method_lrs "$METHOD_LRS")
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
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "${CALIBRATION_NSAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--calibration_nsamples "$CALIBRATION_NSAMPLES")
fi
if [[ "${BF16:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--bf16)
  export ACCELERATE_MIXED_PRECISION=bf16
fi
if [[ "${SAVE_ADAPTERS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_adapters)
fi
if [[ "${RERUN_EXISTING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--rerun_existing)
fi

IFS=',' read -r -a GPU_LIST <<< "$CURVE_GPUS"
NUM_SHARDS=${#GPU_LIST[@]}
echo "Optimization-curve shards: $NUM_SHARDS"
echo "Models: $MODELS"
echo "Methods: $METHODS"
echo "Preset: $PRESET"
echo "Output: $OUT"

pids=()
for shard_id in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$shard_id]}"
  log="$OUT/shard_${shard_id}_gpu_${gpu}.log"
  echo "Launching shard $shard_id/$NUM_SHARDS on GPU $gpu; log $log"
  CUDA_VISIBLE_DEVICES="$gpu" python run_optimization_curves.py \
    --models "$MODELS" \
    --methods "$METHODS" \
    --preset "$PRESET" \
    --lora_rs "$LORA_RS" \
    --seeds "$SEEDS" \
    --train_data "$TRAIN_DATA" \
    --calibration_data "$CALIBRATION_DATA" \
    --out_dir "$OUT" \
    --checkpoint_dir "$CKPT" \
    --num_epochs "$NUM_EPOCHS" \
    --warmup_steps "$WARMUP_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --num_shards "$NUM_SHARDS" \
    --shard_id "$shard_id" \
    "${EXTRA_ARGS[@]}" > "$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
