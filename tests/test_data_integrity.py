import os

from fine_tuning.data_integrity import DEFAULT_DATASET_DIR, audit_overlap, write_heldout
from fine_tuning.evaluate import extract_answer_letter, load_data

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(REPO_DIR, "fine_tuning", "ft-training_set")


def report_by_dataset(train_file: str) -> dict[str, dict]:
    report, _ = audit_overlap(os.path.join(TRAINING_DIR, train_file), DEFAULT_DATASET_DIR)
    return {row["dataset"]: row for row in report}


def test_math7k_is_disjoint_from_all_benchmarks() -> None:
    report = report_by_dataset("math_7k.json")
    assert all(row["overlapping_records"] == 0 for row in report.values())


def test_math17k_overlap_is_explicit_and_stable() -> None:
    report = report_by_dataset("math_17k.json")
    expected = {
        "AddSub": (395, 316, 79),
        "MultiArith": (600, 491, 109),
        "SingleEq": (508, 406, 102),
        "gsm8k": (1319, 1055, 264),
        "AQuA": (254, 203, 51),
        "SVAMP": (1000, 800, 200),
    }
    assert {
        dataset: (
            row["total_records"],
            row["overlapping_records"],
            row["heldout_records"],
        )
        for dataset, row in report.items()
    } == expected


def test_materialized_math17k_heldout_subsets_are_disjoint(tmp_path) -> None:
    train_path = os.path.join(TRAINING_DIR, "math_17k.json")
    _, heldout = audit_overlap(train_path, DEFAULT_DATASET_DIR)
    write_heldout(heldout, str(tmp_path))

    clean_report, _ = audit_overlap(train_path, str(tmp_path))
    assert all(row["overlapping_records"] == 0 for row in clean_report)
    assert len(load_data("MultiArith", dataset_dir=str(tmp_path))) == 109


def test_aqua_answer_extraction_prefers_explicit_answer() -> None:
    assert extract_answer_letter("The options are A, B, C, D, E. Answer: (D)") == "D"
    assert extract_answer_letter("D") == "D"
