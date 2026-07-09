import argparse
import os
import sys
from typing import Dict, List, Tuple

from launch_math_methods import (
    launch_job,
    merge_results,
    parse_csv_list,
    safe_name,
    stop_running_jobs,
    wait_for_slot,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_METHOD_LRS = (
    "lora:1e-3,"
    "rosa:5e-4,"
    "sift-topk:1e-4,"
    "magnitude-bottomk:1e-3,"
    "super-wanda-bottom:1e-3"
)


def parse_method_lrs(value: str) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = []
    for item in parse_csv_list(value, str):
        if ":" not in item:
            raise ValueError(f"Method/LR entries must look like method:lr, got {item!r}")
        method, lr = item.split(":", 1)
        method = method.strip()
        lr = lr.strip()
        if not method or not lr:
            raise ValueError(f"Invalid method/LR entry: {item!r}")
        specs.append((method, lr))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-LR Math17K training and answer-only NLL evaluation for selected methods.",
        allow_abbrev=False,
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--method_lrs", default=DEFAULT_METHOD_LRS)
    parser.add_argument("--datasets", default="Math17K")
    parser.add_argument("--base_out_dir", default="out_answer_only_nll")
    parser.add_argument("--base_checkpoint_dir", default="checkpoints_answer_only_nll")
    parser.add_argument("--script", default=os.path.join(SCRIPT_DIR, "math_experiment_tables.py"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--merge_only", action="store_true")
    args, passthrough = parser.parse_known_args()

    reserved_child_args = {
        "--methods",
        "--lrs",
        "--datasets",
        "--out_dir",
        "--checkpoint_dir",
        "--ppl_target",
        "--skip_accuracy_eval",
        "--eval_all_lrs",
        "--eval_selected_only",
    }
    conflicts = [item for item in passthrough if item.split("=", 1)[0] in reserved_child_args]
    if conflicts:
        parser.error(
            "These child arguments are controlled by run_answer_only_nll_methods.py: "
            + ", ".join(conflicts)
        )
    args.passthrough = passthrough
    return args


def method_out_dir(base_out_dir: str, method: str, lr: str) -> str:
    return os.path.join(base_out_dir, f"{safe_name(method)}_lr{safe_name(lr)}")


def main() -> None:
    args = parse_args()
    method_lrs = parse_method_lrs(args.method_lrs)
    gpus = parse_csv_list(args.gpus, str)
    datasets = parse_csv_list(args.datasets, str)
    if not gpus and not args.merge_only:
        raise ValueError("At least one GPU id is required.")

    method_out_dirs = [method_out_dir(args.base_out_dir, method, lr) for method, lr in method_lrs]

    if not args.merge_only:
        os.makedirs(args.base_out_dir, exist_ok=True)
        os.makedirs(args.base_checkpoint_dir, exist_ok=True)
        running: Dict[str, dict] = {}
        available_gpus = list(gpus)

        try:
            for method, lr in method_lrs:
                if not available_gpus:
                    available_gpus.append(wait_for_slot(running, args.poll_seconds))
                gpu = available_gpus.pop(0)
                out_dir = method_out_dir(args.base_out_dir, method, lr)
                checkpoint_dir = os.path.join(args.base_checkpoint_dir, f"{safe_name(method)}_lr{safe_name(lr)}")
                log_path = os.path.join(out_dir, "launch.log")
                command = [
                    args.python,
                    args.script,
                    "--methods",
                    method,
                    "--lrs",
                    lr,
                    "--datasets",
                    args.datasets,
                    "--out_dir",
                    out_dir,
                    "--checkpoint_dir",
                    checkpoint_dir,
                    "--ppl_target",
                    "answer",
                    "--skip_accuracy_eval",
                    "--eval_all_lrs",
                    *args.passthrough,
                ]
                process, log_file = launch_job(command, gpu=gpu, env=os.environ, log_path=log_path)
                running[gpu] = {
                    "process": process,
                    "method": method,
                    "log_path": log_path,
                    "log_file": log_file,
                }

            while running:
                wait_for_slot(running, args.poll_seconds)
        except Exception:
            stop_running_jobs(running)
            raise

    merge_results(args.base_out_dir, method_out_dirs, datasets)


if __name__ == "__main__":
    main()
