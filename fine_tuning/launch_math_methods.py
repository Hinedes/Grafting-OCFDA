import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

from math_experiment_tables import DEFAULT_METHODS, MATH_BENCHMARKS, parse_csv_list, save_tables


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def merge_results(out_dir: str, method_out_dirs: Iterable[str], datasets: List[str]) -> None:
    by_run_id: Dict[str, dict] = {}
    tuning_by_run_id: Dict[str, dict] = {}
    for method_out_dir in method_out_dirs:
        for row in read_jsonl(os.path.join(method_out_dir, "run_results.jsonl")):
            by_run_id[row["run_id"]] = row
            if row.get("lr_tuning", {}).get("nll") is not None:
                tuning_by_run_id.setdefault(row["run_id"], row)
        for row in read_jsonl(os.path.join(method_out_dir, "tuning_results.jsonl")):
            tuning_by_run_id[row["run_id"]] = row

    merged_rows = list(by_run_id.values())
    merged_tuning_rows = list(tuning_by_run_id.values())
    write_jsonl(os.path.join(out_dir, "run_results.jsonl"), merged_rows)
    if merged_tuning_rows:
        write_jsonl(os.path.join(out_dir, "tuning_results.jsonl"), merged_tuning_rows)
    save_tables(out_dir, merged_rows, datasets, tuning_results=merged_tuning_rows or None)
    print(f"Merged {len(merged_rows)} completed runs into {out_dir}")
    if merged_tuning_rows:
        print(f"Merged {len(merged_tuning_rows)} LR tuning runs into {out_dir}")


def launch_job(command: List[str], gpu: str, env: dict, log_path: str) -> Tuple[subprocess.Popen, object]:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    job_env = dict(env)
    job_env["CUDA_VISIBLE_DEVICES"] = gpu
    print("Launching on GPU", gpu, ":", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=SCRIPT_DIR,
        env=job_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return process, log_file


def wait_for_slot(running: Dict[str, dict], poll_seconds: int) -> Optional[str]:
    while True:
        for gpu, job in list(running.items()):
            return_code = job["process"].poll()
            if return_code is None:
                continue
            job["log_file"].close()
            if return_code != 0:
                raise RuntimeError(
                    f"Method job failed on GPU {gpu} with exit code {return_code}. "
                    f"See {job['log_path']}"
                )
            print(f"Finished {job['method']} on GPU {gpu}")
            del running[gpu]
            return gpu
        time.sleep(poll_seconds)


def stop_running_jobs(running: Dict[str, dict]) -> None:
    for job in running.values():
        process = job["process"]
        if process.poll() is None:
            process.terminate()
    for job in running.values():
        process = job["process"]
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        job["log_file"].close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one math experiment method per GPU and merge the resulting tables.",
        allow_abbrev=False,
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--methods",
        default=DEFAULT_METHODS,
    )
    parser.add_argument("--datasets", default=",".join(MATH_BENCHMARKS))
    parser.add_argument("--base_out_dir", default="out_math_experiments_parallel")
    parser.add_argument("--base_checkpoint_dir", default="checkpoints_math_parallel")
    parser.add_argument("--script", default=os.path.join(SCRIPT_DIR, "math_experiment_tables.py"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--merge_only", action="store_true")
    args, passthrough = parser.parse_known_args()
    reserved_child_args = {"--methods", "--datasets", "--out_dir", "--checkpoint_dir"}
    conflicts = [item for item in passthrough if item.split("=", 1)[0] in reserved_child_args]
    if conflicts:
        parser.error(
            "Use launcher-level arguments for "
            + ", ".join(conflicts)
            + "; pass only shared experiment options through to math_experiment_tables.py."
        )
    args.passthrough = passthrough
    return args


def main() -> None:
    args = parse_args()
    methods = parse_csv_list(args.methods, str)
    gpus = parse_csv_list(args.gpus, str)
    datasets = parse_csv_list(args.datasets, str)
    if not gpus and not args.merge_only:
        raise ValueError("At least one GPU id is required.")

    method_out_dirs = [
        os.path.join(args.base_out_dir, safe_name(method))
        for method in methods
    ]

    if not args.merge_only:
        os.makedirs(args.base_out_dir, exist_ok=True)
        os.makedirs(args.base_checkpoint_dir, exist_ok=True)
        running: Dict[str, dict] = {}
        available_gpus = list(gpus)

        try:
            for method, method_out_dir in zip(methods, method_out_dirs):
                if not available_gpus:
                    available_gpus.append(wait_for_slot(running, args.poll_seconds))
                gpu = available_gpus.pop(0)
                method_checkpoint_dir = os.path.join(args.base_checkpoint_dir, safe_name(method))
                log_path = os.path.join(method_out_dir, "launch.log")
                command = [
                    args.python,
                    args.script,
                    "--methods",
                    method,
                    "--datasets",
                    args.datasets,
                    "--out_dir",
                    method_out_dir,
                    "--checkpoint_dir",
                    method_checkpoint_dir,
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
