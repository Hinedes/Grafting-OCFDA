"""Predeclared B1 paired and sentinel statistics."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

B1_SUPPORT_SEEDS = (1001, 1002, 1003)
B1_TRAINING_SEEDS = (2001, 2002, 2003)
B1_PILOT_SEED = 9001
B1_BOOTSTRAP_REPLICATES = 50_000
B1_BOOTSTRAP_SEED = 424242
B1_BENCHMARKS = ("AddSub", "MultiArith", "SingleEq", "gsm8k", "AQuA", "SVAMP")


def select_shared_lr(
    rows: Sequence[Mapping[str, object]],
    methods: Sequence[str] = ("ocfda-aligned", "ocfda-independent"),
) -> dict[str, object]:
    """Select one LR from paired pilot NLLs using the frozen 0.5% tie rule."""

    if not methods:
        raise ValueError("At least one method is required for shared LR selection")
    by_lr: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    invalid_candidates = []
    for row in rows:
        nll = row.get("lr_tuning", {}).get("nll")
        if nll is None or not np.isfinite(float(nll)):
            if str(row.get("method")) in methods:
                invalid_candidates.append({"method": str(row["method"]), "lr": float(row["lr"])})
            continue
        method = str(row["method"])
        if method in methods:
            by_lr[float(row["lr"])][method].append(float(nll))

    candidates = {}
    for lr, method_values in by_lr.items():
        if any(method not in method_values for method in methods):
            continue
        candidates[lr] = float(np.mean([value for method in methods for value in method_values[method]]))
    if not candidates:
        raise ValueError("No learning rate has finite pilot NLLs for both OCFDA geometries")

    best_lr = min(candidates, key=candidates.get)
    best_nll = candidates[best_lr]
    tolerance = max(abs(best_nll) * 0.005, 1e-12)
    selected_lr = min(lr for lr, nll in candidates.items() if nll <= best_nll + tolerance)
    return {
        "selected_lr": float(selected_lr),
        "selected_nll": float(candidates[selected_lr]),
        "candidates": {str(lr): nll for lr, nll in sorted(candidates.items())},
        "invalid_candidates": invalid_candidates,
        "tie_rule_relative": 0.005,
    }


def select_lr_per_method(
    rows: Sequence[Mapping[str, object]],
    methods: Sequence[str] = ("ocfda-aligned", "ocfda-independent"),
) -> dict[str, dict[str, object]]:
    """Select one validation-NLL learning rate per OCFDA geometry using the frozen 0.5% tie rule."""

    selection: dict[str, dict[str, object]] = {}
    for method in methods:
        by_lr: dict[float, list[float]] = defaultdict(list)
        invalid_candidates = []
        for row in rows:
            if str(row.get("method")) != method:
                continue
            nll = row.get("lr_tuning", {}).get("nll")
            if nll is None or not np.isfinite(float(nll)):
                invalid_candidates.append({"method": method, "lr": float(row["lr"])})
                continue
            by_lr[float(row["lr"])].append(float(nll))

        candidates = {lr: float(np.mean(values)) for lr, values in by_lr.items() if values}
        if not candidates:
            raise ValueError(f"No learning rate has finite pilot NLLs for {method}")
        best_lr = min(candidates, key=candidates.get)
        best_nll = candidates[best_lr]
        tolerance = max(abs(best_nll) * 0.005, 1e-12)
        selected_lr = min(lr for lr, nll in candidates.items() if nll <= best_nll + tolerance)
        selection[method] = {
            "selected_lr": float(selected_lr),
            "selected_nll": float(candidates[selected_lr]),
            "candidates": {str(lr): nll for lr, nll in sorted(candidates.items())},
            "invalid_candidates": invalid_candidates,
            "tie_rule_relative": 0.005,
        }
    return selection


def _pair_delta(row: Mapping[str, object]) -> float:
    if row.get("delta") is not None:
        return float(row["delta"])
    return float(row["aligned_accuracy"]) - float(row["independent_accuracy"])


def hierarchical_paired_bootstrap(
    rows: Sequence[Mapping[str, object]],
    repetitions: int = B1_BOOTSTRAP_REPLICATES,
    seed: int = B1_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Bootstrap paired run deltas by support seed, then training seed."""

    if not rows:
        raise ValueError("At least one paired run is required")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    by_support: dict[int, list[float]] = defaultdict(list)
    seen_pairs = set()
    for row in rows:
        support_seed = int(row["support_seed"])
        training_seed = int(row["training_seed"])
        pair = (support_seed, training_seed)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate paired run: {pair}")
        seen_pairs.add(pair)
        delta = _pair_delta(row)
        if not np.isfinite(delta):
            raise ValueError(f"Non-finite paired delta for {pair}: {delta}")
        by_support[support_seed].append(delta)

    if any(not values for values in by_support.values()):
        raise ValueError("Every support seed must contain at least one training seed")
    supports = sorted(by_support)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(repetitions, dtype=np.float64)
    for replicate in range(repetitions):
        sampled_supports = rng.choice(supports, size=len(supports), replace=True)
        sampled_deltas = []
        for support_seed in sampled_supports:
            support_deltas = by_support[int(support_seed)]
            sampled_deltas.extend(
                support_deltas[index]
                for index in rng.integers(0, len(support_deltas), size=len(support_deltas))
            )
        bootstrap_means[replicate] = float(np.mean(sampled_deltas))

    deltas = np.asarray([_pair_delta(row) for row in rows], dtype=np.float64)
    mean_delta = float(np.mean(deltas))
    ci_lower, ci_upper = np.quantile(bootstrap_means, [0.025, 0.975]).tolist()
    favor_count = int(np.count_nonzero(deltas > 0.0))
    pair_count = len(deltas)
    return {
        "mean_delta_pp": mean_delta,
        "ci95_lower_pp": float(ci_lower),
        "ci95_upper_pp": float(ci_upper),
        "bootstrap": "hierarchical_paired_percentile",
        "bootstrap_replicates": int(repetitions),
        "bootstrap_seed": int(seed),
        "pair_count": pair_count,
        "favor_aligned": favor_count,
        "required_favor_aligned": 6,
        "supported": bool(mean_delta >= 2.0 and ci_lower > 0.0 and favor_count >= 6),
    }


def _load_progress(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as progress_file:
        for line in progress_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_eval_progress(directory: str, datasets: Iterable[str] = B1_BENCHMARKS) -> dict[str, list[dict]]:
    result = {}
    for dataset in datasets:
        path = os.path.join(directory, f"{dataset}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        result[dataset] = _load_progress(path)
    return result


def stratified_example_bootstrap(
    base_results: Mapping[str, Sequence[Mapping[str, object]]],
    lora_results: Mapping[str, Sequence[Mapping[str, object]]],
    datasets: Sequence[str] = B1_BENCHMARKS,
    repetitions: int = B1_BOOTSTRAP_REPLICATES,
    seed: int = B1_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Bootstrap paired correctness differences equally across benchmarks."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    differences = {}
    base_accuracy = {}
    lora_accuracy = {}
    for dataset in datasets:
        base_by_idx = {int(row["idx"]): bool(row["flag"]) for row in base_results[dataset]}
        lora_by_idx = {int(row["idx"]): bool(row["flag"]) for row in lora_results[dataset]}
        if (
            len(base_by_idx) != len(base_results[dataset])
            or len(lora_by_idx) != len(lora_results[dataset])
            or set(base_by_idx) != set(lora_by_idx)
            or not base_by_idx
        ):
            raise ValueError(f"Base and LoRA progress rows do not match for {dataset}")
        differences[dataset] = np.asarray(
            [int(lora_by_idx[index]) - int(base_by_idx[index]) for index in sorted(base_by_idx)],
            dtype=np.float64,
        )
        base_accuracy[dataset] = 100.0 * float(np.mean([base_by_idx[index] for index in sorted(base_by_idx)]))
        lora_accuracy[dataset] = 100.0 * float(np.mean([lora_by_idx[index] for index in sorted(lora_by_idx)]))

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(repetitions, dtype=np.float64)
    for replicate in range(repetitions):
        benchmark_means = []
        for dataset in datasets:
            values = differences[dataset]
            indices = rng.integers(0, len(values), size=len(values))
            benchmark_means.append(float(np.mean(values[indices])))
        bootstrap_means[replicate] = 100.0 * float(np.mean(benchmark_means))

    deltas = {dataset: lora_accuracy[dataset] - base_accuracy[dataset] for dataset in datasets}
    macro_delta = float(np.mean(list(deltas.values())))
    ci_lower, ci_upper = np.quantile(bootstrap_means, [0.025, 0.975]).tolist()
    improved = sum(delta > 0.0 for delta in deltas.values())
    return {
        "base_accuracy": base_accuracy,
        "lora_accuracy": lora_accuracy,
        "benchmark_delta_pp": deltas,
        "macro_delta_pp": macro_delta,
        "ci95_lower_pp": float(ci_lower),
        "ci95_upper_pp": float(ci_upper),
        "bootstrap": "stratified_example_percentile",
        "bootstrap_replicates": int(repetitions),
        "bootstrap_seed": int(seed),
        "improved_benchmarks": int(improved),
        "benchmark_count": len(datasets),
        "passes": bool(macro_delta >= 10.0 and ci_lower > 0.0 and improved >= 4),
    }


def write_json(path: str, value: Mapping[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
