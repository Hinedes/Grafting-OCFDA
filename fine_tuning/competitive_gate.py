"""Competitive-gate harness: aligned OCFDA (read-only) vs LoRA vs Super-Tuning.

Builds the gun, does not pull the trigger: `plan` dry-runs the full matrix
without training; `run` executes one spec per subprocess with resume, atomic
records, and per-run DONE markers; `status` prints completion; `summarize`
writes the final comparison (means AND individuals, exploratory excluded);
`export` archives the gate directory.

Frozen B1 settings (model, data, generation, budget) are reused from
fine_tuning.run_b1 — never redeclared here — so the evaluator invocation
matches the validated B1 evaluator by construction.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

try:
    from .artifacts import sha256_file
    from .b1_statistics import B1_BENCHMARKS, load_eval_progress, select_lr_per_method
    from .math_experiment_tables import MATH_BENCHMARKS, RunSpec, parse_method
    from .run_b1 import (
        EXPECTED_HELDOUT,
        MODEL_ID,
        MODEL_REVISION,
        OCFDA_K,
        TARGET_MODULES,
        _append_common_command,
        require_rocm,
        validate_eval_progress,
    )
except ImportError:
    from artifacts import sha256_file
    from b1_statistics import B1_BENCHMARKS, load_eval_progress, select_lr_per_method
    from math_experiment_tables import MATH_BENCHMARKS, RunSpec, parse_method
    from run_b1 import (
        EXPECTED_HELDOUT,
        MODEL_ID,
        MODEL_REVISION,
        OCFDA_K,
        TARGET_MODULES,
        _append_common_command,
        require_rocm,
        validate_eval_progress,
    )

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_PROTOCOL = "competitive-gate-v1"
OCFDA_TRAINABLE = 5_603_328
BENCHMARKS = list(B1_BENCHMARKS)
assert BENCHMARKS == list(MATH_BENCHMARKS), "gate benchmark set drifted from B1"

# Flags that must equal the validated B1 invocation for the comparison to be fair.
# Optimizer recipe is deliberately NOT in this set: baselines use their native
# recipe (Adam/wd=0, per the Super-Tuning paper) while OCFDA uses B1's AdamW/wd=0.
# Weight decay travels with the optimizer recipe and is checked per-method.
FROZEN_ARGV = {
    "--models": MODEL_ID,
    "--model_revision": MODEL_REVISION,
    "--tokenizer_revision": MODEL_REVISION,
    "--target_modules": TARGET_MODULES,
    "--batch_size": "16",
    "--micro_batch_size": "16",
    "--num_epochs": "3",
    "--cutoff_len": "256",
    "--val_set_size": "120",
    "--warmup_steps": "100",
    "--ocfda_k": str(OCFDA_K),
    "--generation_max_new_tokens": "256",
    "--generation_num_beams": "4",
    "--val_split_seed": "42",
}

TRAIN_END_MARKER = "LR tuning validation ppl:"


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return validate_manifest(manifest)


def validate_manifest(manifest: dict) -> dict:
    if manifest.get("protocol") != GATE_PROTOCOL:
        raise ValueError(f"manifest protocol must be {GATE_PROTOCOL!r}")
    frozen = manifest["frozen_model"]
    if frozen["model_id"] != MODEL_ID or frozen["model_revision"] != MODEL_REVISION:
        raise ValueError("gate manifest model does not match the frozen B1 model")
    training = manifest["frozen_training"]
    if ",".join(training["target_modules"]) != TARGET_MODULES or int(training["lora_r"]) != 8:
        raise ValueError("gate manifest training budget does not match the frozen B1 budget")
    if list(manifest["frozen_eval"]["benchmarks"]) != BENCHMARKS:
        raise ValueError("gate manifest benchmark set does not match B1")
    baselines = manifest.get("baselines", {})
    if not baselines:
        raise ValueError("manifest defines no baselines")
    for name, spec in baselines.items():
        parse_method(spec["method"])  # raises on unknown method
        grid = spec.get("lr_grid", [])
        if not grid or any(not math.isfinite(float(lr)) or float(lr) <= 0 for lr in grid):
            raise ValueError(f"baseline {name!r} has an empty or invalid lr_grid")
        if len(set(float(lr) for lr in grid)) != len(grid):
            raise ValueError(f"baseline {name!r} has a duplicated lr_grid")
        if not spec.get("final_train_seeds"):
            raise ValueError(f"baseline {name!r} has no final_train_seeds")
        if spec.get("support_seed") is not None:
            raise ValueError(f"baseline {name!r} must not set a support seed")
    return manifest


def manifest_sha(path: str) -> str:
    return sha256_file(path)


def mask_seed_of(method: str, train_seed: int, calibration_seed: int) -> tuple[int | None, str]:
    """Separate mask randomness from the train seed; never conflate the two."""
    adapter, mask_choice, _ = parse_method(method)
    if adapter in {"base", "full"}:
        raise ValueError(f"gate does not run {method!r}")
    if adapter == "lora":
        return None, "n/a (dense low-rank update, no mask)"
    if mask_choice == "random" or "rand" in method:
        return int(train_seed), "train-seed RNG (stochastic mask)"
    return int(calibration_seed), "calibration-deterministic mask (data + calibration seed)"


def selection_specs(manifest: dict) -> list[dict]:
    specs = []
    for name, baseline in manifest["baselines"].items():
        for lr in baseline["lr_grid"]:
            specs.append(
                {
                    "phase": "selection",
                    "baseline": name,
                    "method": baseline["method"],
                    "lr": float(lr),
                    "train_seed": int(baseline["selection_train_seed"]),
                }
            )
    return specs


def final_specs(manifest: dict, selections: dict[str, dict]) -> list[dict]:
    specs = []
    for name, baseline in manifest["baselines"].items():
        selected_lr = float(selections[name]["selected_lr"])
        for train_seed in baseline["final_train_seeds"]:
            specs.append(
                {
                    "phase": "final",
                    "baseline": name,
                    "method": baseline["method"],
                    "lr": selected_lr,
                    "train_seed": int(train_seed),
                }
            )
    return specs


def spec_run_id(spec: dict) -> str:
    """Run IDs must equal the pipeline's own RunSpec IDs (single source of truth)."""
    return RunSpec(
        seed=spec["train_seed"],
        model=MODEL_ID,
        lora_r=8,
        lr=spec["lr"],
        method=spec["method"],
        support_seed=None,
    ).run_id


def phase_dirs(gate_dir: str, phase: str, method: str) -> tuple[str, str]:
    out_dir = os.path.join(gate_dir, phase, method, "results")
    checkpoint_dir = os.path.join(gate_dir, phase, method, "checkpoints")
    return out_dir, checkpoint_dir


def build_argv(
    manifest: dict,
    spec: dict,
    heldout_dir: str,
    gate_dir: str,
    b1_manifest_path: str,
    train_data: str,
) -> list[str]:
    """One spec per invocation (timed, isolated); B1-common flags reused verbatim."""
    baseline = manifest["baselines"][spec["baseline"]]
    out_dir, checkpoint_dir = phase_dirs(gate_dir, spec["phase"], spec["method"])
    fake_args = argparse.Namespace(
        output_dir=gate_dir,
        train_data=train_data,
        parallel_eval_workers=1,
    )
    command = [sys.executable, "-m", "fine_tuning.math_experiment_tables"]
    _append_common_command(command, fake_args, heldout_dir)
    command[command.index("PLACEHOLDER_OUT")] = out_dir
    command[command.index("PLACEHOLDER_CHECKPOINT")] = checkpoint_dir
    calibration = baseline.get("calibration", {})
    optimizer = baseline.get("optimizer", {"name": "adamw", "weight_decay": 0.0})
    _replace_flag(command, "--artifact_manifest_path", b1_manifest_path)
    _replace_flag(command, "--calibration_data", str(calibration.get("data", "none")))
    _replace_flag(command, "--optimizer_name", str(optimizer.get("name", "adamw")))
    _replace_flag(command, "--weight_decay", str(optimizer.get("weight_decay", 0.0)))
    command.extend(["--calibration_nsamples", str(calibration.get("nsamples", 128))])
    command.extend(["--calibration_seed", str(calibration.get("seed", 228))])
    command.extend(
        [
            "--methods", spec["method"],
            "--lrs", f"{spec['lr']:g}",
            "--seeds", str(spec["train_seed"]),
            "--save_adapters",
            "--eval_all_lrs",
            "--stop_on_error",
        ]
    )
    if spec["phase"] == "selection":
        command.append("--skip_accuracy_eval")
    return command


def _replace_flag(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


def assert_method_flags(command: list[str], baseline: dict) -> None:
    """Baseline-native training flags (optimizer recipe) must match the manifest."""
    optimizer = baseline.get("optimizer", {"name": "adamw", "weight_decay": 0.0})
    for flag, expected in (
        ("--optimizer_name", str(optimizer.get("name", "adamw"))),
        ("--weight_decay", str(optimizer.get("weight_decay", 0.0))),
        ("--calibration_data", str(baseline.get("calibration", {}).get("data", "none"))),
    ):
        if command[command.index(flag) + 1] != expected:
            raise RuntimeError(f"baseline flag drifted for {flag}: expected {expected}")


def assert_evaluator_match(command: list[str]) -> None:
    """The generation-affecting invocation must equal the validated B1 evaluator."""
    for flag, expected in FROZEN_ARGV.items():
        if flag not in command:
            raise RuntimeError(f"evaluator invocation lost B1 flag {flag}")
        if command[command.index(flag) + 1] != expected:
            raise RuntimeError(f"evaluator invocation drifted for {flag}")
    for flag, expected in (("--generation_batch_size", "1"), ("--lora_rs", "8")):
        if flag in command and command[command.index(flag) + 1] != expected:
            raise RuntimeError(f"evaluator invocation drifted for {flag}")
    if "--skip_lr_tuning_metric" in command:
        raise RuntimeError("LR selection metric must not be skipped")


def b1_paths(b1_dir: str) -> dict[str, str]:
    return {
        "manifest": os.path.join(b1_dir, "artifact_manifest.json"),
        "heldout": os.path.join(b1_dir, "heldout_dataset"),
        "confirmatory_rows": os.path.join(b1_dir, "confirmatory", "results", "run_results.jsonl"),
        "exploratory": os.path.join(b1_dir, "pilot", "winner_evaluation.json"),
    }


def check_dirs(b1_dir: str, gate_dir: str, manifest_path: str) -> dict[str, str]:
    b1_dir, gate_dir = os.path.abspath(b1_dir), os.path.abspath(gate_dir)
    paths = b1_paths(b1_dir)
    if not os.path.isfile(paths["manifest"]):
        raise RuntimeError(f"B1 artifact manifest not found (read-only reference): {paths['manifest']}")
    with open(paths["manifest"], "r", encoding="utf-8") as handle:
        if json.load(handle).get("protocol") != "B1-OCFDA":
            raise RuntimeError("B1 manifest is not for the frozen B1 OCFDA protocol")
    if gate_dir == b1_dir or gate_dir.startswith(b1_dir + os.sep) or b1_dir.startswith(gate_dir + os.sep):
        raise RuntimeError("gate directory overlaps the read-only B1 directory; use a disjoint --gate_output_dir")
    if os.path.exists(gate_dir):
        frozen_path = os.path.join(gate_dir, "gate_manifest.json")
        if not os.path.isfile(frozen_path):
            raise RuntimeError(f"gate directory exists but is not a gate run: {gate_dir}")
        with open(frozen_path, "r", encoding="utf-8") as handle:
            frozen = json.load(handle)
        if frozen.get("manifest_sha256") != manifest_sha(manifest_path):
            raise RuntimeError("gate manifest changed after freezing; use a new --gate_output_dir")
    return paths


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()


def atomic_write_json(path: str, value: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def record_paths(gate_dir: str, run_id: str) -> tuple[str, str, str]:
    run_dir = os.path.join(gate_dir, "runs", run_id)
    return run_dir, os.path.join(run_dir, "record.json"), os.path.join(run_dir, "DONE")


def read_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dir_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def read_amd_sysfs_vram_used() -> int | None:
    """Sum used VRAM across AMD cards from kernel sysfs (read-only, no GPU context)."""
    total, found = 0, False
    for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                total += int(handle.read().strip())
            found = True
        except (OSError, ValueError):
            continue
    return total if found else None


def read_nvidia_smi_used() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    total, found = 0, False
    for line in output.splitlines():
        line = line.strip()
        if line.isdigit():
            total += int(line) * 1024 * 1024
            found = True
    return total if found else None


def sample_device_memory() -> tuple[int | None, str]:
    """Read-only device memory poll. Never allocates, never touches training state."""
    for backend, reader in (("amd-sysfs", read_amd_sysfs_vram_used), ("nvidia-smi", read_nvidia_smi_used)):
        try:
            value = reader()
        except Exception:  # noqa: BLE001 - sampling must never kill a run
            value = None
        if value is not None:
            return value, backend
    return None, "no readable accelerator memory backend (no AMD sysfs VRAM, no nvidia-smi)"


class MemorySampler(threading.Thread):
    """Poll device memory from the parent while the training child is alive.

    External observation only: no CUDA context, no model code, no training
    changes. `read_fn` returns (bytes_or_None, backend) and exists for tests.
    """

    def __init__(self, interval_sec: float = 1.0, read_fn=None) -> None:
        super().__init__(daemon=True)
        self.interval_sec = interval_sec
        self.read_fn = read_fn or sample_device_memory
        self.backend: str | None = None
        self.series: list[tuple[float, int | None]] = []
        self._stop_event = threading.Event()
        self._start = 0.0

    def run(self) -> None:
        self._start = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                value, backend = self.read_fn()
            except Exception:  # noqa: BLE001
                value, backend = None, "sampler-error"
            if value is not None and self.backend is None:
                self.backend = backend
            self.series.append((time.perf_counter() - self._start, value))
            self._stop_event.wait(self.interval_sec)

    def stop(self) -> dict[str, object]:
        self._stop_event.set()
        self.join(timeout=10)
        peaks = [value for _, value in self.series if value is not None]
        return {
            "peak_bytes": max(peaks) if peaks else None,
            "backend": self.backend or "none",
            "samples": len(self.series),
            "reason": None if peaks else sample_device_memory()[1],
        }


def load_ocfda_reference(manifest: dict, confirmatory_path: str) -> dict:
    ref = manifest["ocfda_reference"]
    wanted = {(int(support), int(train)) for support, train in ref["pairs"]}
    found = {}
    for row in read_jsonl(confirmatory_path):
        if row.get("method") != ref["method"]:
            continue
        key = (int(row["support_seed"]), int(row["seed"]))
        if key in wanted:
            if key in found:
                raise RuntimeError(f"duplicate OCFDA reference run for pair {key}")
            found[key] = row
    if set(found) != wanted:
        raise RuntimeError(f"OCFDA reference pairs incomplete: wanted {sorted(wanted)}, got {sorted(found)}")
    scores = []
    for support, train in sorted(wanted):
        row = found[(support, train)]
        ownership = row.get("ocfda_ownership", {})
        if not (
            ownership.get("host", {}).get("match") is True
            and ownership.get("detach", {}).get("detached") is True
            and ownership.get("optimizer", {}).get("only_ocfda") is True
        ):
            raise RuntimeError(f"OCFDA reference run failed integrity: {row.get('run_id')}")
        if int(row.get("trainable_params", -1)) != OCFDA_TRAINABLE:
            raise RuntimeError(f"OCFDA reference budget mismatch: {row.get('run_id')}")
        scores.append(float(row["accuracy"]["Average"]))
    rounded = [round(score, 4) for score in scores]
    if rounded != [float(value) for value in ref["expected_scores_4dp"]]:
        raise RuntimeError(f"OCFDA reference scores differ from the frozen manifest: {rounded}")
    mean = float(sum(scores) / len(scores))
    if abs(mean - float(ref["expected_mean_2dp"])) > 0.005:
        raise RuntimeError(f"OCFDA reference mean drifted: {mean}")
    return {"pairs": sorted(wanted), "scores": scores, "mean": mean, "run_ids": [found[key]["run_id"] for key in sorted(wanted)]}


def load_exploratory(manifest: dict, exploratory_path: str) -> dict:
    if not os.path.exists(exploratory_path):
        return {"status": "absent", "path": exploratory_path}
    with open(exploratory_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "status": "exploratory - never a comparator",
        "path": exploratory_path,
        "aligned_accuracy": float(payload["aligned"]["accuracy_average"]),
    }


def select_lr(manifest: dict, method: str, method_rows: list[dict]) -> dict:
    """Select one validation-NLL LR with the exact B1 0.5%-prefer-lower-LR rule."""
    rule = manifest["selection_rule"]
    selection = select_lr_per_method(method_rows, methods=[method])[method]
    return {
        "selected_lr": selection["selected_lr"],
        "selected_nll": selection["selected_nll"],
        "candidates": selection["candidates"],
        "invalid_candidates": selection["invalid_candidates"],
        "rule": {"metric": rule["metric"], "scope": rule["scope"], "tie": rule["tie"]},
    }


def verify_row(manifest: dict, spec: dict, row: dict, b1_manifest_sha: str, out_dir: str) -> dict:
    checks: dict[str, object] = {}
    checks["spec_match"] = (
        row.get("method") == spec["method"]
        and math.isclose(float(row["lr"]), spec["lr"], rel_tol=0.0, abs_tol=1e-12)
        and int(row["seed"]) == spec["train_seed"]
    )
    tuning = row.get("lr_tuning", {})
    checks["selection_nll_finite"] = math.isfinite(float(tuning.get("nll", float("nan"))))
    checks["selection_nll_examples"] = int(tuning.get("examples", -1)) == 120
    checks["manifest_pinned"] = row.get("artifact_manifest_sha256") == b1_manifest_sha
    expected_optimizer = manifest["baselines"][spec["baseline"]].get("optimizer", {"name": "adamw", "weight_decay": 0.0})
    checks["optimizer"] = row.get("optimizer_name") == expected_optimizer.get("name") and math.isclose(
        float(row.get("weight_decay", float("nan"))), float(expected_optimizer.get("weight_decay", 0.0)), rel_tol=0.0, abs_tol=1e-12
    )
    adapter, _, _ = parse_method(spec["method"])
    reference = int(row.get("reference_lora_params", 0))
    if adapter == "lora":
        checks["budget_exact"] = int(row.get("trainable_params", -1)) == reference
    else:
        checks["budget_within_tolerance"] = abs(float(row.get("trainable_budget_error_pct", 99.0))) <= 3.0
    if spec["phase"] == "final":
        progress_dir = row.get("eval_progress_dir", "")
        progress = load_eval_progress(progress_dir)
        validate_eval_progress(progress, row.get("run_id", "gate"))
        accuracy = row.get("accuracy", {})
        recomputed = {
            dataset: 100.0 * math.fsum(item["flag"] for item in progress[dataset]) / EXPECTED_HELDOUT[dataset]
            for dataset in BENCHMARKS
        }
        checks["accuracy_recomputed"] = all(
            math.isclose(float(accuracy.get(dataset, float("nan"))), recomputed[dataset], rel_tol=0.0, abs_tol=1e-12)
            for dataset in BENCHMARKS
        )
        checks["accuracy_recomputed"] = bool(checks["accuracy_recomputed"]) and math.isclose(
            float(accuracy.get("Average", float("nan"))),
            math.fsum(recomputed.values()) / len(recomputed),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    else:
        checks["accuracy_skipped_by_design"] = row.get("accuracy_eval_skipped") is True
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if value is not True)
        raise RuntimeError(f"integrity checks failed for {row.get('run_id')}: {failed}")
    return checks


def enrich_record(
    manifest: dict,
    spec: dict,
    row: dict,
    timing: dict,
    memory: dict,
    run_dir: str,
    b1_manifest_sha: str,
    manifest_path: str,
    out_dir: str,
    checkpoint_parent: str,
) -> dict:
    baseline = manifest["baselines"][spec["baseline"]]
    calibration = baseline.get("calibration", {})
    mask_seed, mask_kind = mask_seed_of(spec["method"], spec["train_seed"], int(calibration.get("seed", 228)))
    accuracy = row.get("accuracy", {})
    return {
        "protocol": GATE_PROTOCOL,
        "phase": spec["phase"],
        "baseline": spec["baseline"],
        "method": spec["method"],
        "git_commit": git_commit(),
        "gate_manifest_sha256": manifest_sha(manifest_path),
        "b1_manifest_sha256": b1_manifest_sha,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "lr": spec["lr"],
        "train_seed": spec["train_seed"],
        "support_seed": None,
        "mask_seed": mask_seed,
        "mask_seed_kind": mask_kind,
        "calibration": calibration,
        "trainable_params": row.get("trainable_params"),
        "reference_lora_params": row.get("reference_lora_params"),
        "trainable_budget_error_pct": row.get("trainable_budget_error_pct"),
        "optimizer_name": row.get("optimizer_name"),
        "weight_decay": row.get("weight_decay"),
        "wall_sec": timing["wall_sec"],
        "wall_train_sec": timing["wall_train_sec"],
        "wall_eval_sec": (
            timing["wall_sec"] - timing["wall_train_sec"] if timing["wall_train_sec"] is not None else None
        ),
        "wall_split_how": (
            "parent-side stdout timestamps: train ends at the pipeline's 'LR tuning validation ppl' line; "
            "eval is the remainder (generation + benchmark NLL). No training code touched."
            if timing["wall_train_sec"] is not None
            else "train/eval split unavailable: pipeline marker line missing; wall_sec is the whole run"
        ),
        "console_log": os.path.join(run_dir, "console.log"),
        "lr_tuning_nll": row.get("lr_tuning", {}).get("nll"),
        "lr_tuning_examples": row.get("lr_tuning", {}).get("examples"),
        "accuracy": {dataset: accuracy.get(dataset) for dataset in BENCHMARKS} if accuracy else None,
        "macro_accuracy": accuracy.get("Average") if accuracy else None,
        "peak_device_memory": {
            **memory,
            "scope": "full run wall (train+eval); phase attribution via console.log timestamps + memory_trace.csv",
            "trace": os.path.join(run_dir, "memory_trace.csv"),
        },
        "checkpoint_bytes": dir_bytes(os.path.join(checkpoint_parent, row["run_id"])),
        "eval_progress_bytes": dir_bytes(row.get("eval_progress_dir", os.path.join(out_dir, "missing"))),
        "integrity_checks": verify_row(manifest, spec, row, b1_manifest_sha, out_dir),
        "pipeline_run_id": row.get("run_id"),
        "checkpoint_dir": os.path.join(checkpoint_parent, row["run_id"]),
        "eval_progress_dir": row.get("eval_progress_dir"),
    }


def execute_child(command: list[str], run_dir: str, sampler: MemorySampler) -> tuple[int, float, float | None]:
    """Run one spec with timestamped stdout tee + memory sampling.

    Observational only: the child is the unmodified pipeline invocation; the
    parent only timestamps output lines and polls device memory externally.
    """
    os.makedirs(run_dir, exist_ok=True)
    console_path = os.path.join(run_dir, "console.log")
    sampler.start()
    started = time.perf_counter()
    train_end: float | None = None
    with subprocess.Popen(
        command, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as proc:
        assert proc.stdout is not None
        with open(console_path, "w", encoding="utf-8") as log:
            for line in proc.stdout:
                elapsed = time.perf_counter() - started
                log.write(f"{elapsed:10.3f} {line}")
                print(line, end="")
                if train_end is None and TRAIN_END_MARKER in line:
                    train_end = elapsed
        returncode = proc.wait()
    wall_sec = time.perf_counter() - started
    return returncode, wall_sec, train_end


def run_one_spec(
    manifest: dict,
    manifest_path: str,
    spec: dict,
    heldout_dir: str,
    gate_dir: str,
    b1_manifest_path: str,
    b1_manifest_sha: str,
    train_data: str,
    dry_run: bool = False,
) -> str:
    run_id = spec_run_id(spec)
    run_dir, record_path, done_path = record_paths(gate_dir, run_id)
    if os.path.exists(done_path):
        print(f"already DONE: {run_id}")
        return "skipped"
    out_dir, checkpoint_dir = phase_dirs(gate_dir, spec["phase"], spec["method"])
    command = build_argv(manifest, spec, heldout_dir, gate_dir, b1_manifest_path, train_data)
    assert_evaluator_match(command)
    assert_method_flags(command, manifest["baselines"][spec["baseline"]])
    print("$", " ".join(command))
    if dry_run:
        return "dry-run"
    sampler = MemorySampler()
    returncode, wall_sec, train_end = execute_child(command, run_dir, sampler)
    memory = sampler.stop()
    with open(os.path.join(run_dir, "memory_trace.csv"), "w", encoding="utf-8") as trace:
        trace.write("elapsed_sec,used_bytes\n")
        for elapsed, value in sampler.series:
            trace.write(f"{elapsed:.3f},{'' if value is None else value}\n")
    if returncode != 0:
        failure = {
            "run_id": run_id,
            "spec": spec,
            "returncode": returncode,
            "wall_sec": wall_sec,
            "command": command,
        }
        atomic_write_json(os.path.join(run_dir, "failure.json"), failure)
        print(f"FAILED (artifacts of other runs untouched): {run_id}")
        return "failed"
    rows = {row["run_id"]: row for row in read_jsonl(os.path.join(out_dir, "run_results.jsonl"))}
    if run_id not in rows:
        atomic_write_json(os.path.join(run_dir, "failure.json"), {"run_id": run_id, "spec": spec, "error": "row missing after exit 0"})
        return "failed"
    timing = {"wall_sec": wall_sec, "wall_train_sec": train_end}
    record = enrich_record(manifest, spec, rows[run_id], timing, memory, run_dir, b1_manifest_sha, manifest_path, out_dir, checkpoint_dir)
    atomic_write_json(record_path, record)
    with open(done_path, "x", encoding="utf-8") as handle:
        handle.write(sha256_file(record_path) + "\n")
    print(f"DONE: {run_id}")
    return "done"


def freeze_gate_dir(gate_dir: str, manifest_path: str, b1_manifest_sha: str) -> None:
    os.makedirs(gate_dir, exist_ok=True)
    frozen_path = os.path.join(gate_dir, "gate_manifest.json")
    if os.path.exists(frozen_path):
        return
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    atomic_write_json(
        frozen_path,
        {
            "manifest_sha256": manifest_sha(manifest_path),
            "manifest": manifest,
            "b1_manifest_sha256": b1_manifest_sha,
            "git_commit": git_commit(),
            "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def cmd_plan(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    paths = check_dirs(args.b1_output_dir, args.gate_output_dir, args.manifest)
    with open(paths["manifest"], "rb") as handle:
        b1_manifest_sha = hashlib.sha256(handle.read()).hexdigest()
    selection = selection_specs(manifest)
    print(f"protocol: {GATE_PROTOCOL}")
    print(f"b1 (read-only): {os.path.abspath(args.b1_output_dir)}")
    print(f"gate dir:       {os.path.abspath(args.gate_output_dir)}")
    print(f"b1 manifest sha: {b1_manifest_sha[:16]}...")
    print(f"gate manifest sha: {manifest_sha(args.manifest)[:16]}...")
    total = 0
    for phase, specs in (("selection (tuning-only, no generation)", selection),):
        print(f"\n## {phase}: {len(specs)} trainings")
        for spec in specs:
            mask_seed, mask_kind = mask_seed_of(
                spec["method"], spec["train_seed"], int(manifest["baselines"][spec["baseline"]].get("calibration", {}).get("seed", 228))
            )
            print(f"  {spec['baseline']:6s} {spec['method']:8s} lr={spec['lr']:g} train={spec['train_seed']} mask={mask_seed} [{mask_kind}] -> {spec_run_id(spec)}")
            total += 1
    print("\n## final (full generation eval): 3 seeds x selected LR per baseline (LR pending selection)")
    for name, baseline in manifest["baselines"].items():
        optimizer = baseline.get("optimizer", {})
        print(f"  {name:6s} {baseline['method']:8s} lr=<selected> trains={list(baseline['final_train_seeds'])} opt={optimizer.get('name')}/wd={optimizer.get('weight_decay')} cal={baseline.get('calibration', {}).get('data')} -> 3 trainings")
        total += len(list(baseline["final_train_seeds"]))
    print(f"\nplanned trainings total: {total} (+ cheap evaluator reruns only)")
    probe = {"phase": "selection", "baseline": next(iter(manifest["baselines"])), "method": manifest["baselines"][next(iter(manifest["baselines"]))]["method"], "lr": float(manifest["baselines"][next(iter(manifest["baselines"]))]["lr_grid"][0]), "train_seed": 0}
    command = build_argv(manifest, probe, paths["heldout"], args.gate_output_dir, paths["manifest"], manifest_probe_train_data(args))
    assert_evaluator_match(command)
    print("\nevaluator invocation matches the validated B1 evaluator (frozen flags verified).")
    print("no existing confirmatory artifact is touched: gate writes only under the gate dir above.")
    print("\nMI300X sequence:")
    print("  python -m fine_tuning.competitive_gate run --phase selection [...]   # 6 tuning trainings")
    print("  python -m fine_tuning.competitive_gate status [...]                   # inspect selection/selection.json")
    print("  python -m fine_tuning.competitive_gate run --phase final [...]       # 6 full trainings")
    print("  python -m fine_tuning.competitive_gate summarize [...]")
    print("  python -m fine_tuning.competitive_gate export [...]")
    return 0


def manifest_probe_train_data(args: argparse.Namespace) -> str:
    if args.train_data:
        return args.train_data
    return os.path.join(REPO_DIR, "fine_tuning", "ft-training_set", "math_17k.json")


def cmd_status(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    check_dirs(args.b1_output_dir, args.gate_output_dir, args.manifest)
    selection = selection_specs(manifest)
    selection_path = os.path.join(args.gate_output_dir, "selection.json")
    selections = {}
    if os.path.exists(selection_path):
        with open(selection_path, "r", encoding="utf-8") as handle:
            selections = json.load(handle)
    pending_final = final_specs(manifest, selections) if selections else []
    incomplete = 0
    for spec in selection + pending_final:
        run_id = spec_run_id(spec)
        _, _, done_path = record_paths(args.gate_output_dir, run_id)
        state = "DONE" if os.path.exists(done_path) else "missing"
        if state == "missing":
            incomplete += 1
        extra = ""
        if spec["phase"] == "final":
            extra = f" lr={spec['lr']:g}"
        print(f"[{state:7s}] {spec['phase']:9s} {spec['baseline']:6s} train={spec['train_seed']}{extra} {run_id}")
    if not selections:
        print("selection: pending (run --phase selection)")
    else:
        for name, selection_row in selections.items():
            print(f"selection {name}: lr={float(selection_row['selected_lr']):g} nll={float(selection_row['selected_nll']):.4f}")
    print(f"incomplete: {incomplete}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    paths = check_dirs(args.b1_output_dir, args.gate_output_dir, args.manifest)
    train_data = manifest_probe_train_data(args)
    with open(paths["manifest"], "rb") as handle:
        b1_manifest_sha = hashlib.sha256(handle.read()).hexdigest()
    if not args.dry_run:
        require_rocm(args.allow_cuda)
        freeze_gate_dir(args.gate_output_dir, args.manifest, b1_manifest_sha)
    phases = {"selection": ("selection",), "final": ("final",), "all": ("selection", "final")}[args.phase]
    results: dict[str, int] = {}
    if "selection" in phases:
        for spec in selection_specs(manifest):
            state = run_one_spec(manifest, args.manifest, spec, paths["heldout"], args.gate_output_dir, paths["manifest"], b1_manifest_sha, train_data, args.dry_run)
            results[state] = results.get(state, 0) + 1
    if "final" in phases:
        selections = compute_selections(args.gate_output_dir, manifest)
        for spec in final_specs(manifest, selections):
            state = run_one_spec(manifest, args.manifest, spec, paths["heldout"], args.gate_output_dir, paths["manifest"], b1_manifest_sha, train_data, args.dry_run)
            results[state] = results.get(state, 0) + 1
    print(f"run states: {results}")
    return 1 if results.get("failed") else 0


def compute_selections(gate_dir: str, manifest: dict) -> dict[str, dict]:
    selection_path = os.path.join(gate_dir, "selection.json")
    if os.path.exists(selection_path):
        with open(selection_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    selections = {}
    for name, baseline in manifest["baselines"].items():
        out_dir, _ = phase_dirs(gate_dir, "selection", baseline["method"])
        rows = [row for row in read_jsonl(os.path.join(out_dir, "run_results.jsonl")) if row.get("method") == baseline["method"]]
        expected = {(baseline["method"], float(lr)) for lr in baseline["lr_grid"]}
        actual = {(row.get("method"), float(row.get("lr"))) for row in rows}
        if actual != expected or len(rows) != len(expected):
            raise RuntimeError(f"selection grid incomplete for {name}: expected {sorted(expected)}, got {sorted(actual)}")
        selections[name] = select_lr(manifest, baseline["method"], rows)
    atomic_write_json(selection_path, selections)
    return selections


def optimizer_state_footprint_bytes(trainable_params: int) -> int:
    """Adam/AdamW fp32 m+v states dominate optimizer memory: 8 bytes per scalar."""
    return int(trainable_params) * 8


def gate_verdict(
    ocfda_mean: float,
    baseline_means: dict[str, float],
    ocfda_footprint: int,
    baseline_footprints: dict[str, int],
    ocfda_peak: int | None,
    baseline_peaks: dict[str, int | None],
    tolerance_pp: float = 2.0,
    rescue_fraction: float = 0.25,
) -> dict[str, object]:
    """Apply the predeclared gate rule. Pure function of recorded numbers only."""
    best_name = max(baseline_means, key=lambda name: baseline_means[name])
    best_mean = baseline_means[best_name]
    gap = best_mean - ocfda_mean
    verdict: dict[str, object] = {
        "primary": "mean macro accuracy",
        "tolerance_pp": tolerance_pp,
        "best_baseline": best_name,
        "best_baseline_mean": best_mean,
        "ocfda_mean": ocfda_mean,
        "gap_pp": gap,
    }
    if gap <= tolerance_pp:
        verdict.update({"outcome": "PASS", "route": "methods-paper"})
        return verdict
    rescue_evidence: dict[str, object] = {}
    for name, peak in baseline_peaks.items():
        if ocfda_peak is not None and peak is not None and peak > 0:
            rescue_evidence[f"peak_vs_{name}"] = (peak - ocfda_peak) / peak
    for name, footprint in baseline_footprints.items():
        if footprint > 0:
            rescue_evidence[f"footprint_vs_{name}"] = (footprint - ocfda_footprint) / footprint
    verdict["rescue_evidence"] = rescue_evidence
    measured = [value for value in rescue_evidence.values() if isinstance(value, float)]
    if not measured:
        verdict.update({"outcome": "FAIL-accuracy", "route": "undecided-missing-data"})
    elif max(measured) >= rescue_fraction:
        verdict.update({"outcome": "FAIL-accuracy", "route": "tradeoff-paper"})
    else:
        verdict.update({"outcome": "FAIL-accuracy", "route": "stop-peft-route"})
    return verdict


def cmd_summarize(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    paths = check_dirs(args.b1_output_dir, args.gate_output_dir, args.manifest)
    selections = compute_selections(args.gate_output_dir, manifest)
    ocfda = load_ocfda_reference(manifest, paths["confirmatory_rows"])
    exploratory = load_exploratory(manifest, paths["exploratory"])
    summary: dict[str, object] = {
        "protocol": GATE_PROTOCOL,
        "gate_manifest_sha256": manifest_sha(args.manifest),
        "ocfda_aligned_reference": {**ocfda, "label": "read-only reference, 3 matched pairs"},
        "exploratory": exploratory,
        "methods_are_independent": "seeds are not matched across methods; report means side by side, never paired cross-method deltas",
        "baselines": {},
    }
    incomplete = []
    baseline_means: dict[str, float] = {}
    baseline_footprints: dict[str, int] = {}
    baseline_peaks: dict[str, int | None] = {}
    for name, baseline in manifest["baselines"].items():
        out_dir, _ = phase_dirs(args.gate_output_dir, "final", baseline["method"])
        rows = {row["run_id"]: row for row in read_jsonl(os.path.join(out_dir, "run_results.jsonl")) if row.get("method") == baseline["method"]}
        selected_lr = float(selections[name]["selected_lr"])
        scores, run_ids = [], []
        for train_seed in baseline["final_train_seeds"]:
            run_id = spec_run_id({"method": baseline["method"], "lr": selected_lr, "train_seed": int(train_seed)})
            if run_id not in rows:
                incomplete.append(run_id)
                continue
            scores.append(float(rows[run_id]["accuracy"]["Average"]))
            run_ids.append(run_id)
        entry: dict[str, object] = {
            "selected_lr": selected_lr,
            "selection_candidates": selections[name]["candidates"],
            "individual_scores": scores,
            "run_ids": run_ids,
        }
        if len(scores) == len(list(baseline["final_train_seeds"])):
            entry["mean"] = float(sum(scores) / len(scores))
            entry["delta_vs_ocfda_mean_pp"] = float(entry["mean"]) - ocfda["mean"]  # independent means, not paired
            baseline_means[name] = float(entry["mean"])
            trainable = [int(rows[run_id].get("trainable_params", 0)) for run_id in run_ids]
            baseline_footprints[name] = optimizer_state_footprint_bytes(max(trainable))
            peaks = []
            for run_id in run_ids:
                _, record_path, _ = record_paths(args.gate_output_dir, run_id)
                if os.path.exists(record_path):
                    with open(record_path, "r", encoding="utf-8") as handle:
                        peak = json.load(handle).get("peak_device_memory", {}).get("peak_bytes")
                    if isinstance(peak, int):
                        peaks.append(peak)
            baseline_peaks[name] = max(peaks) if peaks else None
        summary["baselines"][name] = entry  # type: ignore[index]
    if incomplete:
        print(f"refusing: {len(incomplete)} final runs missing: {incomplete}")
        return 1
    ocfda_trainable = OCFDA_TRAINABLE
    verdict = gate_verdict(
        ocfda["mean"],
        baseline_means,
        optimizer_state_footprint_bytes(ocfda_trainable),
        baseline_footprints,
        None,  # OCFDA peaks predate instrumentation; retraining OCFDA is forbidden
        baseline_peaks,
        tolerance_pp=float(manifest["gate_rule"]["tolerance_pp"]),
        rescue_fraction=float(manifest["gate_rule"]["rescue_fraction"]),
    )
    verdict["ocfda_peak_note"] = (
        "OCFDA peak memory unevaluable from records (runs predate instrumentation); "
        "do not retrain OCFDA. If the accuracy route fails narrowly, a 2-batch "
        "memory-profiling addendum (no eval, no artifact impact) can supply OCFDA peaks."
    )
    summary["verdict"] = verdict
    atomic_write_json(os.path.join(args.gate_output_dir, "summary.json"), summary)
    print(f"OCFDA aligned reference (read-only): {['%.2f' % score for score in ocfda['scores']]} mean={ocfda['mean']:.2f}")
    for name, entry in summary["baselines"].items():  # type: ignore[union-attr]
        print(f"{name}: lr={float(entry['selected_lr']):g} scores={['%.2f' % score for score in entry['individual_scores']]} mean={float(entry['mean']):.2f} (delta vs OCFDA mean {float(entry['delta_vs_ocfda_mean_pp']):+.2f}pp, independent samples)")
    print(f"exploratory (excluded from comparison): {exploratory}")
    print(f"verdict: {verdict['outcome']} -> {verdict['route']} (gap {float(verdict['gap_pp']):+.2f}pp, tolerance {float(verdict['tolerance_pp']):.1f}pp)")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    load_manifest(args.manifest)
    check_dirs(args.b1_output_dir, args.gate_output_dir, args.manifest)
    destination = args.destination or (os.path.abspath(args.gate_output_dir.rstrip(os.sep)) + ".tar.gz")
    base = os.path.dirname(os.path.abspath(args.gate_output_dir.rstrip(os.sep)))
    name = os.path.basename(os.path.abspath(args.gate_output_dir.rstrip(os.sep)))
    archive = shutil.make_archive(destination[: -len(".tar.gz")] if destination.endswith(".tar.gz") else destination, "gztar", base, name)
    print(f"exported: {archive}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "status", "run", "summarize", "export"])
    parser.add_argument("--manifest", default=os.path.join(REPO_DIR, "fine_tuning", "competitive_gate_manifest.json"))
    parser.add_argument("--b1_output_dir", default="")
    parser.add_argument("--gate_output_dir", default="runs/competitive-gate")
    parser.add_argument("--train_data", default="")
    parser.add_argument("--phase", choices=["selection", "final", "all"], default="selection")
    parser.add_argument("--allow_cuda", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--destination", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in {"run", "summarize", "export"} and not args.b1_output_dir:
        raise SystemExit("--b1_output_dir (read-only B1 run) is required")
    if args.command == "plan" and not args.b1_output_dir:
        raise SystemExit("--b1_output_dir is required to verify the read-only reference")
    commands = {"plan": cmd_plan, "status": cmd_status, "run": cmd_run, "summarize": cmd_summarize, "export": cmd_export}
    raise SystemExit(commands[args.command](args))


if __name__ == "__main__":
    main()
