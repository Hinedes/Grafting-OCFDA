from fine_tuning.b1_statistics import (
    B1_BENCHMARKS,
    B1_PILOT_SEED,
    B1_SUPPORT_SEEDS,
    B1_TRAINING_SEEDS,
    hierarchical_paired_bootstrap,
    select_shared_lr,
    stratified_example_bootstrap,
)


def test_protocol_seed_sets_are_disjoint() -> None:
    assert len(set(B1_SUPPORT_SEEDS)) == 3
    assert len(set(B1_TRAINING_SEEDS)) == 3
    assert B1_PILOT_SEED not in B1_SUPPORT_SEEDS
    assert B1_PILOT_SEED not in B1_TRAINING_SEEDS
    assert not set(B1_SUPPORT_SEEDS) & set(B1_TRAINING_SEEDS)


def test_shared_lr_uses_lower_candidate_within_tie_band() -> None:
    rows = []
    for method in ("ocfda-aligned", "ocfda-independent"):
        rows.extend(
            [
                {"method": method, "lr": 1e-4, "lr_tuning": {"nll": 1.0}},
                {"method": method, "lr": 5e-4, "lr_tuning": {"nll": 0.998}},
            ]
        )
    assert select_shared_lr(rows)["selected_lr"] == 1e-4


def test_shared_lr_marks_nonfinite_candidates_invalid() -> None:
    rows = [
        {"method": "ocfda-aligned", "lr": 1e-4, "lr_tuning": {"nll": 1.0}},
        {"method": "ocfda-independent", "lr": 1e-4, "lr_tuning": {"nll": 1.0}},
        {"method": "ocfda-aligned", "lr": 5e-4, "lr_tuning": {"nll": float("nan")}},
        {"method": "ocfda-independent", "lr": 5e-4, "lr_tuning": {"nll": 1.0}},
    ]
    result = select_shared_lr(rows)
    assert result["selected_lr"] == 1e-4
    assert len(result["invalid_candidates"]) == 1


def test_hierarchical_bootstrap_reports_paired_effect() -> None:
    rows = [
        {"support_seed": 1001, "training_seed": 2001, "delta": 2.0},
        {"support_seed": 1001, "training_seed": 2002, "delta": 3.0},
        {"support_seed": 1002, "training_seed": 2001, "delta": 4.0},
        {"support_seed": 1002, "training_seed": 2002, "delta": 5.0},
    ]
    result = hierarchical_paired_bootstrap(rows, repetitions=200, seed=4)
    assert result["mean_delta_pp"] == 3.5
    assert result["favor_aligned"] == 4
    assert result["ci95_lower_pp"] > 0


def test_stratified_bootstrap_uses_equal_benchmark_weight() -> None:
    base = {dataset: [{"idx": 0, "flag": False}, {"idx": 1, "flag": False}] for dataset in B1_BENCHMARKS}
    lora = {dataset: [{"idx": 0, "flag": True}, {"idx": 1, "flag": False}] for dataset in B1_BENCHMARKS}
    result = stratified_example_bootstrap(base, lora, repetitions=200, seed=4)
    assert result["macro_delta_pp"] == 50.0
    assert result["improved_benchmarks"] == 6
    assert result["passes"] is True
