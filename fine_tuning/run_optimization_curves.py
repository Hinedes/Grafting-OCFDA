import argparse
import gc
import json
import math
import os
import traceback
from dataclasses import asdict
from typing import Dict, Iterable, List

import torch

try:
    from .math_experiment_tables import (
        FULL_LLAMA_TARGET_MODULES,
        RunSpec,
        build_budget_plan,
        collect_trainable_param_report,
        parse_csv_list,
        print_environment,
        print_spec_header,
        set_seed,
        train_one_run,
    )
    from .training_curve_utils import append_jsonl
except ImportError:
    from math_experiment_tables import (
        FULL_LLAMA_TARGET_MODULES,
        RunSpec,
        build_budget_plan,
        collect_trainable_param_report,
        parse_csv_list,
        print_environment,
        print_spec_header,
        set_seed,
        train_one_run,
    )
    from training_curve_utils import append_jsonl


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_METHODS = "lora,rosa,sift-topk,sift-rand,super-rand,super-wanda-bottom,supra-0.5-bottom"

PRESET_LRS: Dict[str, Dict[str, Dict[str, float]]] = {
    "paper_200": {
        "meta-llama/Llama-3.2-1B": {
            "lora": 1e-3,
            "rosa": 5e-4,
            "sift-topk": 1e-4,
            "sift-rand": 1e-3,
            "super-rand": 1e-3,
            "super-wanda-bottom": 5e-3,
            "supra-0.5-bottom": 1e-3,
        },
        "meta-llama/Meta-Llama-3-8B": {
            "lora": 5e-4,
            "rosa": 5e-4,
            "sift-topk": 1e-4,
            "sift-rand": 1e-3,
            "super-rand": 1e-3,
            "super-wanda-bottom": 1e-3,
            "supra-0.5-bottom": 5e-4,
        },
    },
    "full_epoch": {
        "meta-llama/Llama-3.2-1B": {
            "lora": 5e-4,
            "rosa": 5e-4,
            "sift-topk": 1e-4,
            "sift-rand": 5e-4,
            "super-rand": 5e-4,
            "super-wanda-bottom": 5e-4,
            "supra-0.3-bottom": 5e-4,
            "supra-0.5-bottom": 5e-4,
        },
        "meta-llama/Meta-Llama-3-8B": {
            "lora": 5e-4,
            "rosa": 5e-5,
            "sift-topk": 5e-5,
            "sift-rand": 1e-4,
            "super-rand": 1e-3,
            "super-wanda-bottom": 1e-4,
            "supra-0.5-bottom": 1e-4,
            "supra-0.8-bottom": 1e-4,
        },
    },
}


def parse_method_lr_overrides(value: str) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for item in parse_csv_list(value, str):
        if "=" not in item:
            raise ValueError("--method_lrs entries must look like method=lr")
        method, lr_text = item.split("=", 1)
        overrides[method.strip()] = float(lr_text)
    return overrides


def method_lrs_for_model(args, model: str, methods: List[str]) -> Dict[str, float]:
    preset = PRESET_LRS.get(args.preset)
    if preset is None:
        raise ValueError(f"Unknown preset {args.preset!r}. Available presets: {sorted(PRESET_LRS)}")
    lrs = dict(preset.get(model, {}))
    lrs.update(parse_method_lr_overrides(args.method_lrs))
    missing = [method for method in methods if method not in lrs]
    if missing:
        raise ValueError(
            "Missing LR for methods "
            + ", ".join(missing)
            + ". Add them with --method_lrs method=lr,... or choose a preset that includes them."
        )
    return {method: lrs[method] for method in methods}


def curve_complete(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            for line in f:
                if line.strip() and json.loads(line).get("event") == "train_end":
                    return True
    except Exception:  # noqa: BLE001
        return False
    return False


def archive_incomplete_curve(path: str) -> None:
    if not os.path.exists(path) or curve_complete(path):
        return
    idx = 1
    while True:
        archived = f"{path}.partial{idx}"
        if not os.path.exists(archived):
            os.replace(path, archived)
            print("Archived incomplete curve:", archived)
            return
        idx += 1


def finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    return value if math.isfinite(value) else None


def summarize_curve(path: str) -> dict:
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    loss_rows = [row for row in rows if row.get("event") == "log" and row.get("loss") is not None]
    if not loss_rows:
        return {"logged_points": 0, "last_step": 0, "last_loss": None, "min_loss": None}
    losses = [float(row["loss"]) for row in loss_rows]
    return {
        "logged_points": len(loss_rows),
        "last_step": int(loss_rows[-1].get("step", 0)),
        "last_loss": finite_or_none(losses[-1]),
        "min_loss": finite_or_none(min(losses)),
    }


def iter_specs(args) -> Iterable[RunSpec]:
    methods = parse_csv_list(args.methods, str)
    for seed in parse_csv_list(args.seeds, int):
        for model in parse_csv_list(args.models, str):
            method_lrs = method_lrs_for_model(args, model, methods)
            for lora_r in parse_csv_list(args.lora_rs, int):
                for method in methods:
                    yield RunSpec(seed=seed, model=model, lora_r=lora_r, lr=method_lrs[method], method=method)


def run(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.training_curve_dir = os.path.join(args.out_dir, "curves")
    os.makedirs(args.training_curve_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "curve_runs.jsonl")
    failures_path = os.path.join(args.out_dir, "failed_curve_runs.jsonl")

    print_environment()
    target_modules = parse_csv_list(args.target_modules, str)
    specs = list(iter_specs(args))
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    specs = [spec for idx, spec in enumerate(specs) if idx % args.num_shards == args.shard_id]

    print("Optimization-curve run")
    print("Training data:", args.train_data)
    print("Calibration data:", args.calibration_data)
    print("Target modules:", target_modules)
    print("Output directory:", args.out_dir)
    print("Checkpoint directory:", args.checkpoint_dir)
    print("Specs in this shard:", len(specs))
    for spec in specs:
        print("  ", asdict(spec))

    action_count = 0
    for spec in specs:
        if args.max_runs is not None and action_count >= args.max_runs:
            break
        curve_path = os.path.join(args.training_curve_dir, f"{spec.run_id}.jsonl")
        if args.only_missing and curve_complete(curve_path):
            print("Curve already complete:", curve_path)
            continue
        archive_incomplete_curve(curve_path)
        set_seed(spec.seed)
        model = tokenizer = None
        budget_plan = {}
        try:
            budget_plan = build_budget_plan(args, spec, target_modules)
            print_spec_header(spec, budget_plan, stage="optimization_curve")
            if args.dry_run:
                action_count += 1
                continue
            model, tokenizer, checkpoint_dir = train_one_run(args, spec, budget_plan, target_modules)
            trainable_report = collect_trainable_param_report(model, budget_plan)
            summary = {
                "run_id": spec.run_id,
                **asdict(spec),
                **budget_plan,
                **trainable_report,
                "curve_path": curve_path,
                "checkpoint_dir": checkpoint_dir,
                "num_epochs": args.num_epochs,
                "max_steps": args.max_steps,
                "logging_steps": args.logging_steps,
                **summarize_curve(curve_path),
            }
            append_jsonl(summary_path, summary)
            print("Saved curve:", curve_path)
            print("Summary:", json.dumps(summary, sort_keys=True))
            action_count += 1
        except Exception as exc:  # noqa: BLE001
            failure = {
                "run_id": spec.run_id,
                **asdict(spec),
                **budget_plan,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(failures_path, failure)
            print(failure["traceback"])
            if not args.continue_on_error:
                raise
        finally:
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train selected PEFT methods and save optimization curves only.")
    parser.add_argument("--models", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--preset", choices=sorted(PRESET_LRS), default="full_epoch")
    parser.add_argument("--method_lrs", default="", help="Comma-separated overrides, e.g. lora=5e-4,rosa=5e-5")
    parser.add_argument("--lora_rs", default="8")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--target_modules", default=",".join(FULL_LLAMA_TARGET_MODULES))
    parser.add_argument(
        "--train_data",
        default=os.path.join(SCRIPT_DIR, "ft-training_set", "math_17k.json"),
    )
    parser.add_argument("--calibration_data", default="c4")
    parser.add_argument("--calibration_nsamples", type=int, default=128)
    parser.add_argument("--calibration_seed", type=int, default=228)
    parser.add_argument("--full_ft_checkpoint", default="")
    parser.add_argument("--out_dir", default="out_optimization_curves")
    parser.add_argument("--checkpoint_dir", default="checkpoints_optimization_curves")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument("--val_set_size", type=int, default=0)
    parser.add_argument("--val_split_seed", type=int, default=42)
    parser.add_argument("--eval_step", type=int, default=1000000)
    parser.add_argument("--save_step", type=int, default=1000000)
    parser.add_argument("--compile", type=int, default=0)
    parser.add_argument("--optimizer_name", default="adam")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--sparse_rate_override", type=float, default=None)
    parser.add_argument("--rosa_lora_budget_ratio", type=float, default=0.5)
    parser.add_argument("--rosa_schedule", default="wl64")
    parser.add_argument("--rosa_spa_num_grads", type=int, default=1)
    parser.add_argument("--rosa_dtype", default="bf16")
    parser.add_argument("--budget_tolerance_pct", type=float, default=3.0)
    parser.add_argument("--save_adapters", action="store_true")
    parser.add_argument("--only_missing", action="store_true", default=True)
    parser.add_argument("--rerun_existing", dest="only_missing", action="store_false")
    parser.add_argument("--continue_on_error", action="store_true", default=True)
    parser.add_argument("--stop_on_error", dest="continue_on_error", action="store_false")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    os.chdir(SCRIPT_DIR)
    run(parse_args())
