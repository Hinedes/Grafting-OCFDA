"""Execute the frozen B1 OCFDA experiment in its prescribed order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

try:
    from .artifacts import sha256_file, sha256_tree
    from .b1_statistics import (
        B1_BOOTSTRAP_REPLICATES,
        B1_BOOTSTRAP_SEED,
        B1_PILOT_SEED,
        B1_SUPPORT_SEEDS,
        B1_TRAINING_SEEDS,
        hierarchical_paired_bootstrap,
        load_eval_progress,
        select_shared_lr,
        stratified_example_bootstrap,
        write_json,
    )
    from .data_integrity import audit_overlap, write_heldout
except ImportError:
    from artifacts import sha256_file, sha256_tree
    from b1_statistics import (
        B1_BOOTSTRAP_REPLICATES,
        B1_BOOTSTRAP_SEED,
        B1_PILOT_SEED,
        B1_SUPPORT_SEEDS,
        B1_TRAINING_SEEDS,
        hierarchical_paired_bootstrap,
        load_eval_progress,
        select_shared_lr,
        stratified_example_bootstrap,
        write_json,
    )
    from data_integrity import audit_overlap, write_heldout


MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_REVISION = "5d853ed7d16ac794afa8f5c9c7f59f4e9c950954"
SUPERTUNING_COMMIT = "3e961f0bb7ca49417f3804d7a61b24af58fab21d"
TRAIN_BLOB_SHA = "e72c024ec9957e8f7e67d2478450ac8851b666a7"
TRAIN_BYTES = 12_098_055
TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
OCFDA_K = 57
PILOT_LRS = "1e-4,5e-4,1e-3"
PILOT_LR_VALUES = (1e-4, 5e-4, 1e-3)
PRIMARY_LR = 5e-4
EXPECTED_HELDOUT = {
    "AddSub": 79,
    "MultiArith": 109,
    "SingleEq": 102,
    "gsm8k": 264,
    "AQuA": 51,
    "SVAMP": 200,
}
EXPECTED_BENCHMARK_SHA256 = {
    "AddSub": "62a8662b1ad40879953c4df0e61883a749d03aa3fc5e71f1cdb842b816e139ff",
    "MultiArith": "a40f426807c8a563802c2a5b66bda3bcd1eba9c099038fd1199f0dd57de5fd28",
    "SingleEq": "712b15edb851c97aab42a5de106d94008f94d917366c92537ed5400a8aacb207",
    "gsm8k": "2cc616a1e0b23ea1df29370476365cde71c4e0aa823fb06cc194c5a8a9381abe",
    "AQuA": "de677f5f0139340009eb01f17c0db781b4161912cd1f4efa77292f2fcf3478ee",
    "SVAMP": "ffed015784f738f317c5603177861d044c4fc94e09211d78b25e96d46c656e2c",
}
B1_ALLOWED_TREE_CHANGES = {
    "CONTRIBUTING.md",
    "README.md",
    "pyproject.toml",
    "fine_tuning/artifacts.py",
    "fine_tuning/b1_statistics.py",
    "fine_tuning/checkpoints.py",
    "fine_tuning/data_integrity.py",
    "fine_tuning/evaluate.py",
    "fine_tuning/evaluate_checkpoint.py",
    "fine_tuning/finetune.py",
    "fine_tuning/launch_math_methods.py",
    "fine_tuning/math_experiment_tables.py",
    "fine_tuning/ocfda.py",
    "fine_tuning/profile_efficiency.py",
    "fine_tuning/run_b1.py",
    "tests/test_b1_pipeline.py",
    "tests/test_b1_statistics.py",
    "tests/test_checkpoint_metadata.py",
    "tests/test_data_integrity.py",
    "tests/test_evaluate_batching.py",
    "tests/test_ocfda.py",
    "tests/test_run_b1.py",
}


def code_artifact_paths(repo_dir: str) -> list[str]:
    paths = [os.path.join(repo_dir, "README.md"), os.path.join(repo_dir, "pyproject.toml")]
    paths.extend(
        os.path.join(repo_dir, name)
        for name in os.listdir(repo_dir)
        if name.endswith(".py")
    )
    for source_dir in (os.path.join(repo_dir, "fine_tuning"), os.path.join(repo_dir, "src"), os.path.join(repo_dir, "tests")):
        if not os.path.isdir(source_dir):
            continue
        for root, directories, names in os.walk(source_dir):
            directories[:] = [directory for directory in directories if directory != "__pycache__"]
            paths.extend(os.path.join(root, name) for name in names if name.endswith(".py"))
    return sorted(paths)


def working_tree_paths(repo_dir: str) -> set[str]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_dir,
        text=True,
    )
    paths = set()
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path.replace("\\", "/"))
    return paths


def committed_tree_paths(repo_dir: str) -> set[str]:
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{SUPERTUNING_COMMIT}...HEAD"],
        cwd=repo_dir,
        text=True,
    )
    return {path.replace("\\", "/") for path in changed.splitlines() if path}


def git_blob_sha(path: str) -> str:
    with open(path, "rb") as input_file:
        value = input_file.read()
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def require_rocm() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("B1 requires the project PyTorch environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("B1 requires a GPU; refusing to run the protocol on CPU")
    if getattr(getattr(torch, "version", None), "hip", None) is None:
        raise RuntimeError("B1 requires a ROCm/HIP PyTorch build")


def prepare_artifacts(args: argparse.Namespace) -> str:
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        text=True,
    ).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", SUPERTUNING_COMMIT, actual_commit],
        cwd=repo_dir,
        check=False,
    ).returncode:
        raise RuntimeError(f"B1 must descend from Super-Tuning commit {SUPERTUNING_COMMIT}, got {actual_commit}")
    unexpected_changes = (
        committed_tree_paths(repo_dir) | working_tree_paths(repo_dir)
    ) - B1_ALLOWED_TREE_CHANGES
    if unexpected_changes:
        raise RuntimeError(f"Unexpected changes outside the B1 implementation: {sorted(unexpected_changes)}")
    os.makedirs(args.output_dir, exist_ok=True)
    heldout_dir = os.path.join(args.output_dir, "heldout_dataset")
    report, heldout = audit_overlap(args.train_data, args.benchmark_dir)
    observed = {row["dataset"]: row["heldout_records"] for row in report}
    if observed != EXPECTED_HELDOUT:
        raise RuntimeError(f"Unexpected decontaminated benchmark sizes: {observed}")
    write_heldout(heldout, heldout_dir)
    clean_report, _ = audit_overlap(args.train_data, heldout_dir)
    if any(row["overlapping_records"] for row in clean_report):
        raise RuntimeError(f"Materialized evaluation subsets are not disjoint: {clean_report}")

    benchmark_source_files = sha256_tree([args.benchmark_dir])
    for dataset, expected_hash in EXPECTED_BENCHMARK_SHA256.items():
        path = os.path.abspath(os.path.join(args.benchmark_dir, dataset, "test.json"))
        if benchmark_source_files.get(path) != expected_hash:
            raise RuntimeError(f"Benchmark artifact hash mismatch for {dataset}: {benchmark_source_files.get(path)}")

    artifact_manifest = {
        "protocol": "B1-OCFDA",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "supertuning_commit": SUPERTUNING_COMMIT,
        "train_data": {
            "path": os.path.abspath(args.train_data),
            "bytes": os.path.getsize(args.train_data),
            "git_blob_sha1": git_blob_sha(args.train_data),
            "sha256": sha256_file(args.train_data),
        },
        "heldout_dataset": {
            "path": os.path.abspath(heldout_dir),
            "files": sha256_tree([heldout_dir]),
            "record_counts": observed,
        },
        "benchmark_source": {
            "path": os.path.abspath(args.benchmark_dir),
            "files": benchmark_source_files,
        },
        "code_artifacts": {
            "scaffold_commit": SUPERTUNING_COMMIT,
            "repository_commit": actual_commit,
            "files": sha256_tree(code_artifact_paths(repo_dir)),
        },
    }
    train_metadata = artifact_manifest["train_data"]
    if train_metadata["bytes"] != TRAIN_BYTES or train_metadata["git_blob_sha1"] != TRAIN_BLOB_SHA:
        raise RuntimeError(
            "The Math17K training artifact does not match the frozen Super-Tuning blob "
            f"({train_metadata['bytes']} bytes, {train_metadata['git_blob_sha1']})"
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to hash the pinned model snapshot") from exc
    snapshot_path = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    artifact_manifest["model_snapshot"] = {
        "path": os.path.abspath(snapshot_path),
        "files": sha256_tree([snapshot_path]),
    }

    write_json(os.path.join(args.output_dir, "artifact_manifest.json"), artifact_manifest)
    return heldout_dir


def _append_common_command(command: list[str], args: argparse.Namespace, heldout_dir: str) -> None:
    command.extend(
        [
            "--models",
            MODEL_ID,
            "--model_revision",
            MODEL_REVISION,
            "--tokenizer_revision",
            MODEL_REVISION,
            "--artifact_manifest_path",
            os.path.join(args.output_dir, "artifact_manifest.json"),
            "--train_data",
            args.train_data,
            "--dataset_dir",
            heldout_dir,
            "--target_modules",
            TARGET_MODULES,
            "--calibration_data",
            "none",
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
            "0",
            "--optimizer_name",
            "adamw",
            "--ocfda_k",
            str(OCFDA_K),
            "--generation_max_new_tokens",
            "256",
            "--generation_num_beams",
            "4",
            "--val_split_seed",
            "42",
            "--out_dir",
            "PLACEHOLDER_OUT",
            "--checkpoint_dir",
            "PLACEHOLDER_CHECKPOINT",
            "--stop_on_error",
            "--no_resume_eval_progress",
        ]
    )
    command.append("--bf16")


def validate_input_artifacts(args: argparse.Namespace, heldout_dir: str) -> None:
    manifest_path = os.path.join(args.output_dir, "artifact_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("protocol") != "B1-OCFDA":
        raise RuntimeError("Artifact manifest is not for the frozen B1 OCFDA protocol")
    if sha256_file(args.train_data) != manifest["train_data"]["sha256"]:
        raise RuntimeError("Math17K training data changed after the artifact manifest was written")
    if sha256_tree([heldout_dir]) != manifest["heldout_dataset"]["files"]:
        raise RuntimeError("Decontaminated benchmark data changed after the artifact manifest was written")
    if sha256_tree([args.benchmark_dir]) != manifest["benchmark_source"]["files"]:
        raise RuntimeError("Benchmark source data changed after the artifact manifest was written")
    if sha256_tree([manifest["model_snapshot"]["path"]]) != manifest["model_snapshot"]["files"]:
        raise RuntimeError("Pinned model snapshot changed after the artifact manifest was written")
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sha256_tree(code_artifact_paths(repo_dir)) != manifest["code_artifacts"]["files"]:
        raise RuntimeError("B1 source code changed after the artifact manifest was written")


def run_phase(
    args: argparse.Namespace,
    heldout_dir: str,
    name: str,
    methods: str,
    lrs: str,
    training_seeds: str,
    support_seeds: str,
    *,
    eval_all_lrs: bool = False,
    skip_accuracy_eval: bool = False,
    skip_lr_tuning_metric: bool = False,
    max_steps: int | None = None,
    ppl_max_examples: int | None = None,
    eval_step: int | None = None,
    save_step: int | None = None,
    max_runs: int | None = None,
) -> str:
    validate_input_artifacts(args, heldout_dir)
    out_dir = os.path.join(args.output_dir, name, "results")
    checkpoint_dir = os.path.join(args.output_dir, name, "checkpoints")
    command = [sys.executable, "-m", "fine_tuning.math_experiment_tables"]
    _append_common_command(command, args, heldout_dir)
    command[command.index("PLACEHOLDER_OUT")] = out_dir
    command[command.index("PLACEHOLDER_CHECKPOINT")] = checkpoint_dir
    command.extend(
        [
            "--methods",
            methods,
            "--lrs",
            lrs,
            "--seeds",
            training_seeds,
            "--support_seeds",
            support_seeds,
            "--save_adapters",
        ]
    )
    if eval_all_lrs:
        command.append("--eval_all_lrs")
    if skip_accuracy_eval:
        command.append("--skip_accuracy_eval")
    if skip_lr_tuning_metric:
        command.append("--skip_lr_tuning_metric")
    if max_steps is not None:
        command.extend(["--max_steps", str(max_steps)])
    if ppl_max_examples is not None:
        command.extend(["--ppl_max_examples", str(ppl_max_examples)])
    if eval_step is not None:
        command.extend(["--eval_step", str(eval_step)])
    if save_step is not None:
        command.extend(["--save_step", str(save_step)])
    if max_runs is not None:
        command.extend(["--max_runs", str(max_runs)])
    print("Running", name)
    subprocess.run(command, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), check=True)
    return out_dir


def read_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_eval_progress(progress: dict[str, list[dict]], label: str) -> None:
    for dataset, expected_count in EXPECTED_HELDOUT.items():
        rows = progress.get(dataset, [])
        indices = [row.get("idx") for row in rows]
        if (
            len(rows) != expected_count
            or set(indices) != set(range(expected_count))
            or any(not isinstance(row.get("flag"), bool) for row in rows)
        ):
            raise RuntimeError(
                f"{label} evaluation progress for {dataset} is incomplete or duplicated: "
                f"expected {expected_count} rows, got {len(rows)}"
            )


def validate_ocfda_rows(
    rows: list[dict],
    expected_pairs: set[tuple[int, int]],
    selected_lr: float,
    artifact_manifest_sha256: str,
    label: str,
) -> None:
    expected = {
        (method, support_seed, training_seed)
        for support_seed, training_seed in expected_pairs
        for method in ("ocfda-aligned", "ocfda-independent")
    }
    actual = {
        (row.get("method"), int(row["support_seed"]), int(row["seed"]))
        for row in rows
    }
    if len(rows) != len(expected) or actual != expected:
        raise RuntimeError(f"{label} OCFDA runs are incomplete or duplicated: expected {expected}, got {actual}")

    for row in rows:
        method = row["method"]
        geometry = "aligned" if method == "ocfda-aligned" else "independent"
        if not math.isclose(float(row["lr"]), selected_lr, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"{label} run {row.get('run_id')} used an unexpected learning rate")
        if (
            row.get("model") != MODEL_ID
            or row.get("model_revision") != MODEL_REVISION
            or row.get("tokenizer_revision") != MODEL_REVISION
            or row.get("optimizer_name") != "adamw"
            or row.get("weight_decay") != 0.0
            or row.get("ocfda_k") != OCFDA_K
            or row.get("ocfda_geometry") != geometry
            or row.get("target_modules") != ["gate_proj", "up_proj", "down_proj"]
            or row.get("artifact_manifest_sha256") != artifact_manifest_sha256
        ):
            raise RuntimeError(f"{label} run metadata does not match the frozen OCFDA protocol: {row.get('run_id')}")

        supports = row.get("ocfda_supports")
        if not isinstance(supports, dict) or set(supports) != {str(index) for index in range(16)}:
            raise RuntimeError(f"{label} run has an invalid OCFDA support record: {row.get('run_id')}")
        for layer in range(16):
            layer_supports = supports[str(layer)]
            if set(layer_supports) != {"gate_proj", "up_proj", "down_proj"}:
                raise RuntimeError(f"{label} run has incomplete OCFDA supports: {row.get('run_id')}")
            for support in layer_supports.values():
                if (
                    not isinstance(support, list)
                    or len(support) != OCFDA_K
                    or len(set(support)) != OCFDA_K
                    or min(support) < 0
                    or max(support) >= 8192
                ):
                    raise RuntimeError(f"{label} run has invalid OCFDA coordinates: {row.get('run_id')}")
            if geometry == "aligned" and len({tuple(layer_supports[name]) for name in layer_supports}) != 1:
                raise RuntimeError(f"{label} aligned supports differ: {row.get('run_id')}")

        ownership = row.get("ocfda_ownership", {})
        if not (
            ownership.get("host", {}).get("match") is True
            and ownership.get("detach", {}).get("detached") is True
            and ownership.get("optimizer", {}).get("only_ocfda") is True
        ):
            raise RuntimeError(f"{label} run failed an OCFDA integrity check: {row.get('run_id')}")

        progress = load_eval_progress(row.get("eval_progress_dir", ""))
        validate_eval_progress(progress, f"{label} {row.get('run_id')}")
        accuracy = row.get("accuracy", {})
        computed_accuracy = {
            dataset: 100.0 * math.fsum(row["flag"] for row in progress[dataset]) / EXPECTED_HELDOUT[dataset]
            for dataset in EXPECTED_HELDOUT
        }
        computed_average = math.fsum(computed_accuracy.values()) / len(computed_accuracy)
        if any(
            dataset not in accuracy
            or not math.isfinite(float(accuracy[dataset]))
            or not math.isclose(float(accuracy[dataset]), computed_accuracy[dataset], rel_tol=0.0, abs_tol=1e-12)
            for dataset in EXPECTED_HELDOUT
        ) or not math.isfinite(float(accuracy.get("Average", float("nan")))) or not math.isclose(
            float(accuracy["Average"]), computed_average, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"{label} run has incomplete accuracy results: {row.get('run_id')}")
        for metric in ("ppl", "nll"):
            values = row.get(metric, {})
            if any(
                dataset not in values or not math.isfinite(float(values[dataset]))
                for dataset in EXPECTED_HELDOUT
            ) or not math.isfinite(float(values.get("Average", float("nan")))):
                raise RuntimeError(f"{label} run has incomplete {metric} results: {row.get('run_id')}")
        examples = row.get("ppl_examples", {})
        if set(examples) != set(EXPECTED_HELDOUT) or any(
            int(examples[dataset]) != expected_count for dataset, expected_count in EXPECTED_HELDOUT.items()
        ):
            raise RuntimeError(f"{label} run has incomplete PPL example counts: {row.get('run_id')}")


def run_lora_sentinel(args: argparse.Namespace, heldout_dir: str) -> dict[str, object]:
    out_dir = run_phase(
        args,
        heldout_dir,
        "sentinel",
        "base,lora",
        f"{PRIMARY_LR:g}",
        str(B1_PILOT_SEED),
        str(B1_PILOT_SEED),
        eval_all_lrs=True,
    )
    rows = read_jsonl(os.path.join(out_dir, "run_results.jsonl"))
    by_method = {row["method"]: row for row in rows}
    if len(rows) != 2 or set(by_method) != {"base", "lora"}:
        raise RuntimeError(f"Expected Base and LoRA sentinel results, got {sorted(by_method)}")
    base_progress = load_eval_progress(by_method["base"]["eval_progress_dir"])
    lora_progress = load_eval_progress(by_method["lora"]["eval_progress_dir"])
    validate_eval_progress(base_progress, "Base")
    validate_eval_progress(lora_progress, "LoRA")
    sentinel = stratified_example_bootstrap(
        base_progress,
        lora_progress,
        repetitions=B1_BOOTSTRAP_REPLICATES,
        seed=B1_BOOTSTRAP_SEED,
    )
    write_json(os.path.join(args.output_dir, "sentinel", "lora_sentinel.json"), sentinel)
    if not sentinel["passes"]:
        raise SystemExit("Clean LoRA sentinel failed; stopping before Graft interpretation.")
    return sentinel


def run_pilot(args: argparse.Namespace, heldout_dir: str) -> dict[str, object]:
    out_dir = run_phase(
        args,
        heldout_dir,
        "pilot",
        "ocfda-aligned,ocfda-independent",
        PILOT_LRS,
        str(B1_PILOT_SEED),
        str(B1_PILOT_SEED),
        max_runs=6,
    )
    rows = read_jsonl(os.path.join(out_dir, "tuning_results.jsonl"))
    expected = {
        (method, lr)
        for method in ("ocfda-aligned", "ocfda-independent")
        for lr in PILOT_LR_VALUES
    }
    actual = {(row.get("method"), float(row.get("lr"))) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise RuntimeError(f"Pilot grid is incomplete or duplicated: expected {expected}, got {actual}")
    if any(int(row.get("seed")) != B1_PILOT_SEED or int(row.get("support_seed")) != B1_PILOT_SEED for row in rows):
        raise RuntimeError("Pilot must use training and support seed 9001")
    invalid = [
        row.get("run_id")
        for row in rows
        if not math.isfinite(float(row.get("lr_tuning", {}).get("nll", float("nan"))))
        or int(row.get("lr_tuning", {}).get("examples", -1)) != 120
    ]
    if invalid:
        raise RuntimeError(f"Pilot contains non-finite or incomplete validation NLL results: {invalid}")
    selection = select_shared_lr(rows)
    write_json(os.path.join(args.output_dir, "pilot", "lr_selection.json"), selection)
    return selection


def paired_confirmatory_rows(rows: list[dict]) -> list[dict]:
    by_pair = {}
    seen_methods = set()
    for row in rows:
        key = (int(row["support_seed"]), int(row["seed"]))
        method_key = (key, row["method"])
        if method_key in seen_methods:
            raise RuntimeError(f"Duplicate confirmatory run: {method_key}")
        seen_methods.add(method_key)
        by_pair.setdefault(key, {})[row["method"]] = row
    expected = {(support, training) for support in B1_SUPPORT_SEEDS for training in B1_TRAINING_SEEDS}
    if set(by_pair) != expected:
        raise RuntimeError(f"Confirmatory pairing is incomplete: expected {expected}, got {set(by_pair)}")

    paired = []
    for support_seed, training_seed in sorted(expected):
        methods = by_pair[(support_seed, training_seed)]
        if set(methods) != {"ocfda-aligned", "ocfda-independent"}:
            raise RuntimeError(f"Missing geometry in pair {(support_seed, training_seed)}")
        paired.append(
            {
                "support_seed": support_seed,
                "training_seed": training_seed,
                "aligned_accuracy": methods["ocfda-aligned"]["accuracy"]["Average"],
                "independent_accuracy": methods["ocfda-independent"]["accuracy"]["Average"],
            }
        )
    return paired


def validate_sanity_rows(rows: list[dict]) -> None:
    expected = {
        ("ocfda-aligned", 1001, 2001),
        ("ocfda-independent", 1001, 2001),
    }
    actual = {
        (row.get("method"), int(row["support_seed"]), int(row["seed"]))
        for row in rows
    }
    if len(rows) != len(expected) or actual != expected:
        raise RuntimeError(f"Sanity pair is incomplete or duplicated: expected {expected}, got {actual}")


def run(args: argparse.Namespace) -> None:
    require_rocm()
    args.output_dir = os.path.abspath(args.output_dir)
    args.train_data = os.path.abspath(args.train_data)
    args.benchmark_dir = os.path.abspath(args.benchmark_dir)
    if os.path.exists(args.output_dir):
        if not os.path.isdir(args.output_dir):
            raise RuntimeError(f"B1 output path is not a directory: {args.output_dir}")
        with os.scandir(args.output_dir) as entries:
            if any(entries):
                raise RuntimeError(f"B1 requires a fresh output directory: {args.output_dir}")
    heldout_dir = prepare_artifacts(args)
    run_phase(
        args,
        heldout_dir,
        "smoke_2batch",
        "ocfda-aligned,ocfda-independent",
        f"{PRIMARY_LR:g}",
        str(B1_PILOT_SEED),
        str(B1_PILOT_SEED),
        eval_all_lrs=True,
        skip_accuracy_eval=True,
        skip_lr_tuning_metric=True,
        max_steps=2,
        ppl_max_examples=2,
        eval_step=1,
        save_step=1,
        max_runs=2,
    )
    run_lora_sentinel(args, heldout_dir)
    selection = run_pilot(args, heldout_dir)
    selected_lr = float(selection["selected_lr"])
    sanity_dir = run_phase(
        args,
        heldout_dir,
        "sanity",
        "ocfda-aligned,ocfda-independent",
        f"{selected_lr:g}",
        "2001",
        "1001",
        eval_all_lrs=True,
    )
    sanity_rows = read_jsonl(os.path.join(sanity_dir, "run_results.jsonl"))
    validate_sanity_rows(sanity_rows)
    manifest_sha256 = sha256_file(os.path.join(args.output_dir, "artifact_manifest.json"))
    validate_ocfda_rows(
        sanity_rows,
        {(1001, 2001)},
        selected_lr,
        manifest_sha256,
        "Sanity",
    )
    confirmatory_dir = run_phase(
        args,
        heldout_dir,
        "confirmatory",
        "ocfda-aligned,ocfda-independent",
        f"{selected_lr:g}",
        ",".join(map(str, B1_TRAINING_SEEDS)),
        ",".join(map(str, B1_SUPPORT_SEEDS)),
        eval_all_lrs=True,
    )
    rows = read_jsonl(os.path.join(confirmatory_dir, "run_results.jsonl"))
    validate_ocfda_rows(
        rows,
        {(support, training) for support in B1_SUPPORT_SEEDS for training in B1_TRAINING_SEEDS},
        selected_lr,
        manifest_sha256,
        "Confirmatory",
    )
    paired = paired_confirmatory_rows(rows)
    statistics = hierarchical_paired_bootstrap(
        paired,
        repetitions=B1_BOOTSTRAP_REPLICATES,
        seed=B1_BOOTSTRAP_SEED,
    )
    statistics["paired_runs"] = paired
    statistics["selected_lr"] = selected_lr
    statistics["sanity_pair"] = {"support_seed": 1001, "training_seed": 2001}
    write_json(os.path.join(args.output_dir, "b1_statistics.json"), statistics)
    if not statistics["supported"]:
        raise SystemExit("B1 confirmatory criteria were not all met.")


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/b1")
    parser.add_argument("--train_data", default=str(repo_dir / "fine_tuning" / "ft-training_set" / "math_17k.json"))
    parser.add_argument("--benchmark_dir", default=str(repo_dir / "fine_tuning" / "dataset"))
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
