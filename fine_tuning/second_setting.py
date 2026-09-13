"""Frozen second-setting driver: Llama-3.2-3B, same Math17K/heldout evaluation.

Question: was the OCFDA geometry/competitive result specific to Llama-3.2-1B?

Commands
    plan       validate the frozen manifest + budget arithmetic, print the 9-run
               matrix and the exact commands; no execution, no GPU.
    run        prepare artifacts on first use, then execute specs with resume,
               per-run isolation, atomic records, and resource instrumentation.
    status     completion state of the frozen matrix.
    summarize  geometry-transfer and competitiveness readouts from records.

The only intended change relative to the 1B competitive gate is the host model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

try:
    from . import competitive_gate as gate
    from .artifacts import sha256_file, sha256_tree
    from .b1_statistics import B1_BENCHMARKS, load_eval_progress
    from .data_integrity import audit_overlap, write_heldout
    from .math_experiment_tables import RunSpec
    from .run_b1 import (
        EXPECTED_BENCHMARK_SHA256,
        EXPECTED_HELDOUT,
        TRAIN_BLOB_SHA,
        TRAIN_BYTES,
        code_artifact_paths,
        git_blob_sha,
        require_rocm,
        validate_eval_progress,
    )
except ImportError:
    import competitive_gate as gate
    from artifacts import sha256_file, sha256_tree
    from b1_statistics import B1_BENCHMARKS, load_eval_progress
    from data_integrity import audit_overlap, write_heldout
    from math_experiment_tables import RunSpec
    from run_b1 import (
        EXPECTED_BENCHMARK_SHA256,
        EXPECTED_HELDOUT,
        TRAIN_BLOB_SHA,
        TRAIN_BYTES,
        code_artifact_paths,
        git_blob_sha,
        require_rocm,
        validate_eval_progress,
    )

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO_DIR, "fine_tuning", "second_setting_manifest.json")
SETTING_PROTOCOL = "second-setting-v1"
ARTIFACT_PROTOCOL = "B2-OCFDA"
TARGET_MODULES_STR = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
TARGET_MODULES_LIST = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
OCFDA_TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]
MODEL_FLAGS = {"--models", "--model_revision", "--tokenizer_revision", "--ocfda_k"}


# --------------------------------------------------------------------------- manifest


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return validate_manifest(manifest)


def validate_manifest(manifest: dict) -> dict:
    if manifest.get("protocol") != SETTING_PROTOCOL:
        raise ValueError(f"manifest protocol must be {SETTING_PROTOCOL!r}")
    model = manifest["model"]
    for key in ("model_id", "model_revision", "tokenizer_revision"):
        if not model.get(key):
            raise ValueError(f"manifest model.{key} must be set")
    arch = manifest["architecture"]
    for key in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim"):
        if int(arch[key]) <= 0:
            raise ValueError(f"manifest architecture.{key} must be positive")
    budget = manifest["budget"]
    for key in ("reference_lora_params", "target_dense_params_all7", "ocfda_k", "ocfda_params", "super_expected_params"):
        if int(budget[key]) <= 0:
            raise ValueError(f"manifest budget.{key} must be positive")
    if float(budget["tolerance_pct"]) <= 0:
        raise ValueError("manifest budget.tolerance_pct must be positive")
    if parser_target_modules(manifest) != TARGET_MODULES_STR:
        raise ValueError("manifest fixed_training.target_modules drifted from the shared target set")
    if list(manifest["frozen_eval"]["benchmarks"]) != list(B1_BENCHMARKS):
        raise ValueError("manifest frozen_eval.benchmarks drifted from the shared held-out suite")
    if not math.isclose(float(manifest["runs"]["lr"]), 5e-4, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("manifest runs.lr must be the transferred 5e-4")
    runs = manifest["runs"]
    for name in ("ocfda-aligned", "ocfda-independent"):
        entry = runs.get(name)
        if entry is None or entry.get("adapter") != "ocfda" or not entry.get("pairs"):
            raise ValueError(f"manifest runs.{name} is incomplete")
        if entry["geometry"] != name.removeprefix("ocfda-"):
            raise ValueError(f"manifest runs.{name} geometry mismatch")
        for pair in entry["pairs"]:
            if len(pair) != 2:
                raise ValueError(f"manifest runs.{name} pairs must be (support, train)")
    entry = runs.get("super-bottom")
    if entry is None or entry.get("adapter") != "super" or not entry.get("train_seeds"):
        raise ValueError("manifest runs.super-bottom is incomplete")
    if entry["calibration"] != {"data": "c4", "nsamples": 128, "seed": 228}:
        raise ValueError("manifest runs.super-bottom calibration drifted from the canonical policy")
    if entry["optimizer"] != {"name": "adam", "weight_decay": 0.0}:
        raise ValueError("manifest runs.super-bottom optimizer drifted")
    return manifest


def parser_target_modules(manifest: dict) -> str:
    return ",".join(manifest["fixed_training"]["target_modules"])


def repo_path(relative: str) -> str:
    return os.path.join(REPO_DIR, relative) if not os.path.isabs(relative) else relative


# --------------------------------------------------------------------------- budget arithmetic


def module_shapes(arch: dict) -> list[tuple[int, int]]:
    hidden = int(arch["hidden_size"])
    intermediate = int(arch["intermediate_size"])
    kv_out = int(arch["num_key_value_heads"]) * int(arch["head_dim"])
    per_layer = [
        (hidden, hidden),
        (kv_out, hidden),
        (kv_out, hidden),
        (hidden, hidden),
        (intermediate, hidden),
        (intermediate, hidden),
        (hidden, intermediate),
    ]
    return per_layer * int(arch["num_hidden_layers"])


def reference_lora_params(arch: dict, r: int) -> int:
    return sum(r * (out_features + in_features) for out_features, in_features in module_shapes(arch))


def ocfda_params(arch: dict, k: int) -> int:
    return 3 * int(arch["num_hidden_layers"]) * k * int(arch["hidden_size"])


def super_params(arch: dict, reference: int) -> int:
    shapes = module_shapes(arch)
    rate = reference / sum(out_features * in_features for out_features, in_features in shapes)
    return sum(
        min(int(rate * out_features * in_features) + 1, out_features * in_features)
        for out_features, in_features in shapes
    )


def verify_budget(manifest: dict) -> dict:
    arch = manifest["architecture"]
    budget = manifest["budget"]
    reference = reference_lora_params(arch, int(budget["reference_lora_r"]))
    ocfda = ocfda_params(arch, int(budget["ocfda_k"]))
    super_expected = super_params(arch, reference)
    details = {
        "reference_lora_params": reference,
        "manifest_reference_lora_params": int(budget["reference_lora_params"]),
        "ocfda_params": ocfda,
        "manifest_ocfda_params": int(budget["ocfda_params"]),
        "ocfda_error_pct": 100.0 * (ocfda - reference) / reference,
        "super_expected_params": super_expected,
        "manifest_super_expected_params": int(budget["super_expected_params"]),
        "dense_all7": sum(out_features * in_features for out_features, in_features in module_shapes(arch)),
    }
    tolerance = float(budget["tolerance_pct"])
    details["ok"] = (
        reference == details["manifest_reference_lora_params"]
        and ocfda == details["manifest_ocfda_params"]
        and super_expected == details["manifest_super_expected_params"]
        and abs(details["ocfda_error_pct"]) <= tolerance
        and abs(100.0 * (super_expected - reference) / reference) <= tolerance
    )
    close_k = min(
        range(1, int(arch["intermediate_size"]) + 1),
        key=lambda candidate: abs(ocfda_params(arch, candidate) - reference),
    )
    details["closest_k"] = close_k
    details["ok"] = bool(details["ok"] and close_k == int(budget["ocfda_k"]))
    details["ocfda_k_candidates_pct"] = {
        str(candidate): 100.0 * (ocfda_params(arch, candidate) - reference) / reference
        for candidate in range(int(budget["ocfda_k"]) - 3, int(budget["ocfda_k"]) + 4)
    }
    return details


# --------------------------------------------------------------------------- matrix


def build_specs(manifest: dict) -> list[dict]:
    specs = []
    lr = float(manifest["runs"]["lr"])
    for name in ("ocfda-aligned", "ocfda-independent"):
        entry = manifest["runs"][name]
        for support_seed, train_seed in entry["pairs"]:
            specs.append(
                {
                    "method": name,
                    "adapter": entry["adapter"],
                    "geometry": entry["geometry"],
                    "lr": lr,
                    "train_seed": int(train_seed),
                    "support_seed": int(support_seed),
                    "optimizer": entry["optimizer"],
                    "calibration": {"data": "none", "nsamples": 128, "seed": 228},
                }
            )
    entry = manifest["runs"]["super-bottom"]
    for train_seed in entry["train_seeds"]:
        specs.append(
            {
                "method": entry["method"],
                "adapter": "super",
                "geometry": None,
                "lr": lr,
                "train_seed": int(train_seed),
                "support_seed": None,
                "optimizer": entry["optimizer"],
                "calibration": entry["calibration"],
            }
        )
    return specs


def spec_run_id(manifest: dict, spec: dict) -> str:
    return RunSpec(
        seed=spec["train_seed"],
        model=manifest["model"]["model_id"],
        lora_r=8,
        lr=spec["lr"],
        method=spec["method"],
        support_seed=spec["support_seed"],
    ).run_id


# --------------------------------------------------------------------------- commands


def frozen_flags(manifest: dict) -> dict[str, str]:
    model = manifest["model"]
    return {
        "--models": model["model_id"],
        "--model_revision": model["model_revision"],
        "--tokenizer_revision": model["tokenizer_revision"],
        "--target_modules": TARGET_MODULES_STR,
        "--batch_size": "16",
        "--micro_batch_size": "16",
        "--num_epochs": "3",
        "--cutoff_len": "256",
        "--val_set_size": "120",
        "--warmup_steps": "100",
        "--ocfda_k": str(manifest["budget"]["ocfda_k"]),
        "--generation_max_new_tokens": "256",
        "--generation_num_beams": "4",
        "--val_split_seed": "42",
    }


def build_command(manifest: dict, spec: dict, run_dir: str, artifact_manifest_path: str, heldout_dir: str) -> list[str]:
    method_dir = os.path.join(run_dir, spec["method"])
    out_dir = os.path.join(method_dir, "results")
    checkpoint_dir = os.path.join(method_dir, "checkpoints")
    model = manifest["model"]
    optimizer = spec["optimizer"]
    calibration = spec["calibration"]
    command = [
        sys.executable,
        "-m",
        "fine_tuning.math_experiment_tables",
        "--models",
        model["model_id"],
        "--model_revision",
        model["model_revision"],
        "--tokenizer_revision",
        model["tokenizer_revision"],
        "--artifact_manifest_path",
        artifact_manifest_path,
        "--train_data",
        repo_path(manifest["paths"]["train_data"]),
        "--dataset_dir",
        heldout_dir,
        "--target_modules",
        TARGET_MODULES_STR,
        "--calibration_data",
        str(calibration["data"]),
        "--lora_rs",
        "8",
        "--batch_size",
        "16",
        "--micro_batch_size",
        "16",
        "--num_epochs",
        "3",
        "--cutoff_len",
        "256",
        "--val_set_size",
        "120",
        "--warmup_steps",
        "100",
        "--weight_decay",
        str(optimizer["weight_decay"]),
        "--optimizer_name",
        str(optimizer["name"]),
        "--ocfda_k",
        str(manifest["budget"]["ocfda_k"]),
        "--generation_max_new_tokens",
        "256",
        "--generation_num_beams",
        "4",
        "--val_split_seed",
        "42",
        "--out_dir",
        out_dir,
        "--checkpoint_dir",
        checkpoint_dir,
        "--stop_on_error",
        "--no_resume_eval_progress",
        "--parallel_eval_workers",
        "2",
        "--bf16",
        "--calibration_nsamples",
        str(calibration.get("nsamples", 128)),
        "--calibration_seed",
        str(calibration.get("seed", 228)),
        "--methods",
        spec["method"],
        "--lrs",
        f"{spec['lr']:g}",
        "--seeds",
        str(spec["train_seed"]),
    ]
    if spec["support_seed"] is not None:
        command.extend(["--support_seeds", str(spec["support_seed"])])
    command.extend(["--save_adapters", "--eval_all_lrs", "--stop_on_error"])
    return command


def assert_frozen_flags(command: list[str], manifest: dict) -> None:
    for flag, expected in frozen_flags(manifest).items():
        if flag not in command:
            raise RuntimeError(f"command lost frozen flag {flag}")
        if command[command.index(flag) + 1] != expected:
            raise RuntimeError(f"command drifted for {flag}: expected {expected!r}")
    # Evaluator parity: every non-model flag must equal the validated 1B evaluator.
    for flag, expected in gate.FROZEN_ARGV.items():
        if flag in MODEL_FLAGS:
            continue
        if command[command.index(flag) + 1] != expected:
            raise RuntimeError(f"evaluator parity drift for {flag}: expected {expected!r}")


def assert_method_flags(command: list[str], spec: dict) -> None:
    optimizer = spec["optimizer"]
    for flag, expected in (
        ("--optimizer_name", str(optimizer["name"])),
        ("--weight_decay", str(optimizer["weight_decay"])),
        ("--calibration_data", str(spec["calibration"]["data"])),
    ):
        if command[command.index(flag) + 1] != expected:
            raise RuntimeError(f"method flag drifted for {flag}: expected {expected!r}")


# --------------------------------------------------------------------------- artifacts


def prepare_artifacts(manifest: dict, run_dir: str) -> str:
    model = manifest["model"]
    train_data = repo_path(manifest["paths"]["train_data"])
    benchmark_dir = repo_path(manifest["paths"]["benchmark_dir"])
    report, heldout = audit_overlap(train_data, benchmark_dir)
    observed = {row["dataset"]: row["heldout_records"] for row in report}
    if observed != EXPECTED_HELDOUT:
        raise RuntimeError(f"Unexpected decontaminated benchmark sizes: {observed}")
    heldout_dir = os.path.join(run_dir, "heldout_dataset")
    write_heldout(heldout, heldout_dir)
    clean_report, _ = audit_overlap(train_data, heldout_dir)
    if any(row["overlapping_records"] for row in clean_report):
        raise RuntimeError(f"Materialized evaluation subsets are not disjoint: {clean_report}")
    benchmark_files = sha256_tree([benchmark_dir])
    for dataset, expected_hash in EXPECTED_BENCHMARK_SHA256.items():
        path = os.path.abspath(os.path.join(benchmark_dir, dataset, "test.json"))
        if benchmark_files.get(path) != expected_hash:
            raise RuntimeError(f"Benchmark artifact hash mismatch for {dataset}: {benchmark_files.get(path)}")
    train_metadata = {
        "path": os.path.abspath(train_data),
        "bytes": os.path.getsize(train_data),
        "git_blob_sha1": git_blob_sha(train_data),
        "sha256": sha256_file(train_data),
    }
    if train_metadata["bytes"] != TRAIN_BYTES or train_metadata["git_blob_sha1"] != TRAIN_BLOB_SHA:
        raise RuntimeError(f"Math17K training artifact does not match the frozen snapshot: {train_metadata}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to hash the pinned model snapshot") from exc
    snapshot_path = snapshot_download(repo_id=model["model_id"], revision=model["model_revision"])
    artifact_manifest = {
        "protocol": ARTIFACT_PROTOCOL,
        "setting_protocol": SETTING_PROTOCOL,
        "question": manifest["question"],
        "model": model["model_id"],
        "model_revision": model["model_revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "ocfda_k": int(manifest["budget"]["ocfda_k"]),
        "budget": dict(manifest["budget"]),
        "train_data": train_metadata,
        "heldout_dataset": {
            "path": os.path.abspath(heldout_dir),
            "files": sha256_tree([heldout_dir]),
            "record_counts": observed,
        },
        "benchmark_source": {"path": os.path.abspath(benchmark_dir), "files": benchmark_files},
        "model_snapshot": {"path": os.path.abspath(snapshot_path), "files": sha256_tree([snapshot_path])},
        "code_artifacts": {
            "repository_commit": gate.git_commit(),
            "files": sha256_tree(code_artifact_paths(REPO_DIR)),
        },
    }
    gate.atomic_write_json(os.path.join(run_dir, "artifact_manifest.json"), artifact_manifest)
    return heldout_dir


def load_artifact_manifest(manifest: dict, run_dir: str) -> tuple[str, dict]:
    path = os.path.join(run_dir, "artifact_manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        artifact_manifest = json.load(handle)
    model = manifest["model"]
    if (
        artifact_manifest.get("protocol") != ARTIFACT_PROTOCOL
        or artifact_manifest.get("model") != model["model_id"]
        or artifact_manifest.get("model_revision") != model["model_revision"]
    ):
        raise RuntimeError("second-setting artifact manifest does not match the frozen model")
    heldout_dir = os.path.join(run_dir, "heldout_dataset")
    if sha256_tree([heldout_dir]) != artifact_manifest["heldout_dataset"]["files"]:
        raise RuntimeError("second-setting held-out dataset changed after preparation")
    return path, artifact_manifest


def freeze_run_dir(run_dir: str, manifest_path: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    frozen_path = os.path.join(run_dir, "second_setting_frozen.json")
    manifest_hash = sha256_file(manifest_path)
    if os.path.exists(frozen_path):
        with open(frozen_path, "r", encoding="utf-8") as handle:
            frozen = json.load(handle)
        if frozen.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("second-setting manifest changed after freezing; use a new --run_dir")
        return
    gate.atomic_write_json(
        frozen_path,
        {
            "manifest_sha256": manifest_hash,
            "git_commit": gate.git_commit(),
            "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


# --------------------------------------------------------------------------- row verification


def _mask_seed(spec: dict) -> tuple[int | None, str]:
    if spec["adapter"] == "ocfda":
        return int(spec["support_seed"]), "support seed (deterministic coordinate draw)"
    if spec["adapter"] == "super":
        return int(spec["calibration"]["seed"]), "calibration-deterministic mask (C4 + calibration seed)"
    return None, "n/a"


def verify_row(manifest: dict, spec: dict, row: dict, artifact_manifest_sha: str) -> dict:
    import math as _math

    checks: dict[str, object] = {}
    checks["spec_match"] = (
        row.get("method") == spec["method"]
        and _math.isclose(float(row["lr"]), spec["lr"], rel_tol=0.0, abs_tol=1e-12)
        and int(row["seed"]) == spec["train_seed"]
        and int(row.get("support_seed") or -1) == int(spec["support_seed"] or -1)
    )
    tuning = row.get("lr_tuning", {})
    checks["lr_tuning_finite"] = _math.isfinite(float(tuning.get("nll", float("nan"))))
    checks["lr_tuning_examples"] = int(tuning.get("examples", -1)) == 120
    checks["artifact_pinned"] = row.get("artifact_manifest_sha256") == artifact_manifest_sha
    optimizer = spec["optimizer"]
    checks["optimizer"] = (
        row.get("optimizer_name") == optimizer["name"]
        and _math.isclose(float(row.get("weight_decay", float("nan"))), float(optimizer["weight_decay"]), rel_tol=0.0, abs_tol=1e-12)
    )
    if spec["adapter"] == "ocfda":
        checks["budget_exact"] = int(row.get("trainable_params", -1)) == int(manifest["budget"]["ocfda_params"])
        checks["target_modules"] = row.get("target_modules") == OCFDA_TARGET_MODULES
        ownership = row.get("ocfda_ownership", {})
        checks["ownership"] = (
            ownership.get("host", {}).get("match") is True
            and ownership.get("detach", {}).get("detached") is True
            and ownership.get("optimizer", {}).get("only_ocfda") is True
        )
        layers = int(manifest["architecture"]["num_hidden_layers"])
        k = int(manifest["budget"]["ocfda_k"])
        supports = row.get("ocfda_supports")
        structure_ok = isinstance(supports, dict) and set(supports) == {str(index) for index in range(layers)}
        if structure_ok:
            for layer in range(layers):
                layer_supports = supports[str(layer)]
                if set(layer_supports) != set(OCFDA_TARGET_MODULES):
                    structure_ok = False
                    break
                for support in layer_supports.values():
                    if (
                        not isinstance(support, list)
                        or len(support) != k
                        or len(set(support)) != k
                        or min(support) < 0
                        or max(support) >= int(manifest["architecture"]["intermediate_size"])
                    ):
                        structure_ok = False
                        break
                if spec["geometry"] == "aligned" and len({tuple(layer_supports[name]) for name in layer_supports}) != 1:
                    structure_ok = False
                if not structure_ok:
                    break
        checks["supports_structure"] = bool(structure_ok)
    else:
        checks["budget_within_tolerance"] = abs(float(row.get("trainable_budget_error_pct", 99.0))) <= float(
            manifest["budget"]["tolerance_pct"]
        )
    progress = load_eval_progress(row.get("eval_progress_dir", ""))
    validate_eval_progress(progress, row.get("run_id", "second-setting"))
    accuracy = row.get("accuracy", {})
    recomputed = {
        dataset: 100.0 * _math.fsum(item["flag"] for item in progress[dataset]) / EXPECTED_HELDOUT[dataset]
        for dataset in B1_BENCHMARKS
    }
    checks["accuracy_recomputed"] = all(
        _math.isclose(float(accuracy.get(dataset, float("nan"))), recomputed[dataset], rel_tol=0.0, abs_tol=1e-12)
        for dataset in B1_BENCHMARKS
    ) and _math.isclose(
        float(accuracy.get("Average", float("nan"))),
        _math.fsum(recomputed.values()) / len(recomputed),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    failed = sorted(key for key, value in checks.items() if value is not True)
    if failed:
        raise RuntimeError(f"integrity checks failed for {row.get('run_id')}: {failed}")
    return checks


def enrich_record(manifest: dict, spec: dict, row: dict, timing: dict, memory: dict, run_dir: str, artifact_manifest_sha: str, artifact_manifest_path: str, setting_manifest_path: str) -> dict:
    model = manifest["model"]
    mask_seed, mask_kind = _mask_seed(spec)
    accuracy = row.get("accuracy", {})
    checkpoint_dir = os.path.join(run_dir, spec["method"], "checkpoints", row["run_id"])
    return {
        "protocol": SETTING_PROTOCOL,
        "artifact_protocol": ARTIFACT_PROTOCOL,
        "baseline": spec["method"],
        "method": spec["method"],
        "adapter": spec["adapter"],
        "geometry": spec.get("geometry"),
        "git_commit": gate.git_commit(),
        "setting_manifest_sha256": sha256_file(setting_manifest_path),
        "artifact_manifest_sha256": artifact_manifest_sha,
        "artifact_manifest_path": artifact_manifest_path,
        "model": model["model_id"],
        "model_revision": model["model_revision"],
        "lr": spec["lr"],
        "train_seed": spec["train_seed"],
        "support_seed": spec["support_seed"],
        "mask_seed": mask_seed,
        "mask_seed_kind": mask_kind,
        "calibration": spec["calibration"],
        "optimizer_name": row.get("optimizer_name"),
        "weight_decay": row.get("weight_decay"),
        "trainable_params": row.get("trainable_params"),
        "reference_lora_params": row.get("reference_lora_params"),
        "wall_sec": timing["wall_sec"],
        "wall_train_sec": timing["wall_train_sec"],
        "wall_eval_sec": (timing["wall_sec"] - timing["wall_train_sec"]) if timing["wall_train_sec"] is not None else None,
        "wall_split_how": (
            "parent-side stdout timestamps: train ends at the pipeline's 'LR tuning validation ppl' line; eval is the remainder."
            if timing["wall_train_sec"] is not None
            else "train/eval split unavailable: marker line missing; wall_sec is the whole run"
        ),
        "console_log": os.path.join(gate.record_paths(run_dir, row["run_id"])[0], "console.log"),
        "lr_tuning_nll": row.get("lr_tuning", {}).get("nll"),
        "accuracy": {dataset: accuracy.get(dataset) for dataset in B1_BENCHMARKS} if accuracy else None,
        "macro_accuracy": accuracy.get("Average") if accuracy else None,
        "peak_device_memory": {
            **memory,
            "scope": "full run wall (train+eval); phase attribution via console.log timestamps + memory_trace.csv",
            "trace": os.path.join(gate.record_paths(run_dir, row["run_id"])[0], "memory_trace.csv"),
        },
        "checkpoint_bytes": gate.dir_bytes(checkpoint_dir),
        "integrity_checks": verify_row(manifest, spec, row, artifact_manifest_sha),
        "pipeline_run_id": row["run_id"],
        "checkpoint_dir": checkpoint_dir,
        "eval_progress_dir": row.get("eval_progress_dir"),
    }


# --------------------------------------------------------------------------- execution


def run_one_spec(manifest: dict, spec: dict, run_dir: str, artifact_manifest_path: str, heldout_dir: str, artifact_manifest_sha: str, setting_manifest_path: str, dry_run: bool = False) -> str:
    run_id = spec_run_id(manifest, spec)
    record_dir, record_path, done_path = gate.record_paths(run_dir, run_id)
    if os.path.exists(done_path):
        print(f"already DONE: {run_id}")
        return "skipped"
    command = build_command(manifest, spec, run_dir, artifact_manifest_path, heldout_dir)
    assert_frozen_flags(command, manifest)
    assert_method_flags(command, spec)
    print("$", " ".join(command))
    if dry_run:
        return "dry-run"
    sampler = gate.MemorySampler()
    returncode, wall_sec, train_end = gate.execute_child(command, record_dir, sampler)
    memory = sampler.stop()
    with open(os.path.join(record_dir, "memory_trace.csv"), "w", encoding="utf-8") as trace:
        trace.write("elapsed_sec,used_bytes\n")
        for elapsed, value in sampler.series:
            trace.write(f"{elapsed:.3f},{'' if value is None else value}\n")
    if returncode != 0:
        gate.atomic_write_json(
            os.path.join(record_dir, "failure.json"),
            {"run_id": run_id, "spec": spec, "returncode": returncode, "wall_sec": wall_sec, "command": command},
        )
        print(f"FAILED (other runs untouched): {run_id}")
        return "failed"
    results_path = os.path.join(run_dir, spec["method"], "results", "run_results.jsonl")
    rows = {row["run_id"]: row for row in gate.read_jsonl(results_path) if row.get("method") == spec["method"]}
    if run_id not in rows:
        gate.atomic_write_json(os.path.join(record_dir, "failure.json"), {"run_id": run_id, "spec": spec, "error": "row missing after exit 0"})
        return "failed"
    timing = {"wall_sec": wall_sec, "wall_train_sec": train_end}
    record = enrich_record(manifest, spec, rows[run_id], timing, memory, run_dir, artifact_manifest_sha, artifact_manifest_path, setting_manifest_path)
    gate.atomic_write_json(record_path, record)
    with open(done_path, "x", encoding="utf-8") as handle:
        handle.write(sha256_file(record_path) + "\n")
    print(f"DONE: {run_id}")
    return "done"


# --------------------------------------------------------------------------- verdicts


def geometry_transfer_verdict(aligned_scores: list[float], independent_scores: list[float], wins_needed: int = 2) -> dict:
    if len(aligned_scores) != len(independent_scores) or not aligned_scores:
        raise ValueError("geometry verdict requires matched aligned/independent score lists")
    deltas = [float(a) - float(b) for a, b in zip(aligned_scores, independent_scores)]
    mean_delta = sum(deltas) / len(deltas)
    wins = sum(1 for delta in deltas if delta > 0)
    return {
        "mean_delta_pp": mean_delta,
        "per_pair_delta_pp": deltas,
        "wins": wins,
        "pair_count": len(deltas),
        "passed": bool(mean_delta > 0.0 and wins >= wins_needed),
        "strong_replication": wins == len(deltas),
    }


def competitiveness_verdict(aligned_mean: float, super_mean: float, tolerance_pp: float = 2.0) -> dict:
    delta = float(aligned_mean) - float(super_mean)
    return {
        "aligned_mean": float(aligned_mean),
        "super_mean": float(super_mean),
        "delta_pp": delta,
        "tolerance_pp": tolerance_pp,
        "passed": bool(delta >= -tolerance_pp),
    }


# --------------------------------------------------------------------------- commands


def cmd_plan(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    budget = verify_budget(manifest)
    run_dir = os.path.abspath(args.run_dir or repo_path(manifest["paths"]["run_dir"]))
    print(f"SECOND SETTING (FROZEN): {manifest['model']['model_id']} @ {manifest['model']['model_revision']}")
    print(f"question: {manifest['question']}")
    print(f"run_dir: {run_dir}")
    print(
        "budget: reference_lora_r8={reference_lora_params} | ocfda k={k} -> {ocfda_params} ({err:+.4f}%) | super ~{super_params}".format(
            reference_lora_params=budget["reference_lora_params"],
            k=manifest["budget"]["ocfda_k"],
            ocfda_params=budget["ocfda_params"],
            err=budget["ocfda_error_pct"],
            super_params=budget["super_expected_params"],
        )
    )
    if not budget["ok"]:
        raise RuntimeError(f"budget verification failed: {json.dumps(budget, indent=2)}")
    print(f"k closest to reference: {budget['closest_k']} (frozen: {manifest['budget']['ocfda_k']})")
    print()
    specs = build_specs(manifest)
    print(f"matrix: {len(specs)} full trainings")
    for index, spec in enumerate(specs, 1):
        support = spec["support_seed"] if spec["support_seed"] is not None else "-"
        print(
            f"{index:2d}. {spec['method']:18s} train={spec['train_seed']} support={support} lr={spec['lr']:g} "
            f"opt={spec['optimizer']['name']}/wd{spec['optimizer']['weight_decay']} cal={spec['calibration']['data']} -> {spec_run_id(manifest, spec)}"
        )
    print()
    if os.path.exists(os.path.join(run_dir, "second_setting_frozen.json")):
        freeze_run_dir(run_dir, args.manifest)  # raises on manifest drift
        print("frozen manifest matches this run dir.")
    print("exact commands:")
    for spec in specs:
        command = build_command(manifest, spec, run_dir, os.path.join(run_dir, "artifact_manifest.json"), os.path.join(run_dir, "heldout_dataset"))
        assert_frozen_flags(command, manifest)
        assert_method_flags(command, spec)
        print("$", " ".join(command))
    print()
    expectations = manifest["resource_expectations"]
    print("resource expectations (from 1B measurements, 3x FLOPs / 2.16x params):")
    print(f"  train/run: {expectations['train_sec_per_run']} s | eval/run serial: {expectations['eval_sec_per_run_serial']} s | OCFDA parallel-2: {expectations['eval_sec_per_run_ocfda_parallel2']} s")
    print(f"  per-run: OCFDA {expectations['per_ocfda_run_minutes']} min, Super {expectations['per_super_run_minutes']} min | total block: {expectations['total_block_hours']} h")
    print(f"  peak VRAM estimate: {expectations['peak_vram_gb_estimate']} GB (recommend >= {expectations['recommended_vram_gb']} GB)")
    print("NO GPU EXECUTION: plan only.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    run_dir = os.path.abspath(args.run_dir or repo_path(manifest["paths"]["run_dir"]))
    missing = 0
    for spec in build_specs(manifest):
        run_id = spec_run_id(manifest, spec)
        _, _, done_path = gate.record_paths(run_dir, run_id)
        state = "DONE" if os.path.exists(done_path) else "missing"
        if state == "missing":
            missing += 1
        print(f"[{state:7s}] {spec['method']:18s} train={spec['train_seed']} support={spec['support_seed']} {run_id}")
    print(f"missing: {missing}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    budget = verify_budget(manifest)
    if not budget["ok"]:
        raise RuntimeError(f"budget verification failed: {json.dumps(budget, indent=2)}")
    run_dir = os.path.abspath(args.run_dir or repo_path(manifest["paths"]["run_dir"]))
    if not args.dry_run:
        require_rocm(args.allow_cuda)
        freeze_run_dir(run_dir, args.manifest)
    os.makedirs(run_dir, exist_ok=True)
    try:
        artifact_manifest_path, _ = load_artifact_manifest(manifest, run_dir)
    except FileNotFoundError:
        if args.dry_run:
            artifact_manifest_path = os.path.join(run_dir, "artifact_manifest.json")
        else:
            print("preparing frozen artifacts (data audit + model snapshot hash)...")
            prepare_artifacts(manifest, run_dir)
            artifact_manifest_path, _ = load_artifact_manifest(manifest, run_dir)
    artifact_manifest_sha = sha256_file(artifact_manifest_path) if os.path.isfile(artifact_manifest_path) else None
    heldout_dir = os.path.join(run_dir, "heldout_dataset")
    results: dict[str, int] = {}
    for spec in build_specs(manifest):
        state = run_one_spec(manifest, spec, run_dir, artifact_manifest_path, heldout_dir, artifact_manifest_sha, args.manifest, args.dry_run)
        results[state] = results.get(state, 0) + 1
    print(f"run states: {results}")
    return 1 if results.get("failed") else 0


def cmd_summarize(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    run_dir = os.path.abspath(args.run_dir or repo_path(manifest["paths"]["run_dir"]))
    records: dict[str, dict] = {}
    missing = []
    for spec in build_specs(manifest):
        run_id = spec_run_id(manifest, spec)
        _, record_path, done_path = gate.record_paths(run_dir, run_id)
        if not os.path.exists(done_path) or not os.path.exists(record_path):
            missing.append(run_id)
            continue
        with open(record_path, "r", encoding="utf-8") as handle:
            records[spec["method"] + "|" + str(spec["train_seed"]) + "|" + str(spec["support_seed"])] = json.load(handle)
    if missing:
        print(f"refusing: {len(missing)} runs incomplete: {missing}")
        return 1
    aligned = [
        records[f"ocfda-aligned|{train}|{support}"]["macro_accuracy"]
        for support, train in manifest["runs"]["ocfda-aligned"]["pairs"]
    ]
    independent = [
        records[f"ocfda-independent|{train}|{support}"]["macro_accuracy"]
        for support, train in manifest["runs"]["ocfda-independent"]["pairs"]
    ]
    super_scores = [records[f"super-bottom|{train}|None"]["macro_accuracy"] for train in manifest["runs"]["super-bottom"]["train_seeds"]]
    aligned_mean = sum(aligned) / len(aligned)
    independent_mean = sum(independent) / len(independent)
    super_mean = sum(super_scores) / len(super_scores)
    geometry = geometry_transfer_verdict(aligned, independent)
    competitiveness = competitiveness_verdict(aligned_mean, super_mean, tolerance_pp=2.0)
    summary = {
        "protocol": SETTING_PROTOCOL,
        "model": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "ocfda_k": manifest["budget"]["ocfda_k"],
        "lr": manifest["runs"]["lr"],
        "aligned_scores": aligned,
        "independent_scores": independent,
        "super_scores": super_scores,
        "aligned_mean": aligned_mean,
        "independent_mean": independent_mean,
        "super_mean": super_mean,
        "geometry_transfer": geometry,
        "competitiveness": competitiveness,
        "setting1_reference": {
            "aligned_mean_1b": 52.103629886409436,
            "aligned_pairs_1b": [46.338871734967356, 54.40442926536295, 55.567588658898],
            "super_bottomk_mean_1b": 47.953899732913584,
            "super_scores_1b": [52.25246936082598, 39.43697062021777, 52.172259217697],
            "note": "read-only setting-1 results, not part of this run",
        },
    }
    gate.atomic_write_json(os.path.join(run_dir, "second_setting_summary.json"), summary)
    print(f"aligned:     {['%.2f' % value for value in aligned]} mean={aligned_mean:.2f}")
    print(f"independent: {['%.2f' % value for value in independent]} mean={independent_mean:.2f}")
    print(f"super:       {['%.2f' % value for value in super_scores]} mean={super_mean:.2f}")
    print(
        f"geometry transfer: {'PASS' if geometry['passed'] else 'FAIL'} "
        f"(mean delta {geometry['mean_delta_pp']:+.2f}pp, wins {geometry['wins']}/{geometry['pair_count']}"
        f"{', strong replication' if geometry['strong_replication'] else ''})"
    )
    print(
        f"competitiveness: {'PASS' if competitiveness['passed'] else 'FAIL'} "
        f"(aligned {aligned_mean:.2f} vs super {super_mean:.2f}, delta {competitiveness['delta_pp']:+.2f}pp, tolerance {competitiveness['tolerance_pp']:.1f}pp)"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "status", "summarize"])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--allow_cuda", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = {"plan": cmd_plan, "run": cmd_run, "status": cmd_status, "summarize": cmd_summarize}
    raise SystemExit(commands[args.command](args))


if __name__ == "__main__":
    main()
