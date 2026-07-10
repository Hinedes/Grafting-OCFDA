"""Audit instruction overlap between fine-tuning and evaluation data."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Iterable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_DATA = os.path.join(SCRIPT_DIR, "ft-training_set", "math_17k.json")
DEFAULT_DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")


def normalize_instruction(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def load_records(path: str) -> list[dict]:
    with open(path, "r") as input_file:
        records = json.load(input_file)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return records


def benchmark_paths(dataset_dir: str) -> Iterable[tuple[str, str]]:
    for name in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, name, "test.json")
        if os.path.isfile(path):
            yield name, path


def audit_overlap(train_data: str, dataset_dir: str = DEFAULT_DATASET_DIR) -> tuple[list[dict], dict[str, list[dict]]]:
    train_records = load_records(train_data)
    train_instructions = {
        normalize_instruction(record.get("instruction", ""))
        for record in train_records
    }

    report = []
    heldout_by_dataset = {}
    for dataset, path in benchmark_paths(dataset_dir):
        records = load_records(path)
        overlapping = [
            record
            for record in records
            if normalize_instruction(record.get("instruction", "")) in train_instructions
        ]
        heldout = [
            record
            for record in records
            if normalize_instruction(record.get("instruction", "")) not in train_instructions
        ]
        report.append(
            {
                "dataset": dataset,
                "total_records": len(records),
                "overlapping_records": len(overlapping),
                "heldout_records": len(heldout),
            }
        )
        heldout_by_dataset[dataset] = heldout
    return report, heldout_by_dataset


def write_heldout(heldout_by_dataset: dict[str, list[dict]], output_dir: str) -> None:
    for dataset, records in heldout_by_dataset.items():
        directory = os.path.join(output_dir, dataset)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "test.json"), "w") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--write_heldout_dir",
        default="",
        help="Optionally write benchmark subsets whose normalized instructions do not occur in the training file.",
    )
    parser.add_argument(
        "--require_disjoint",
        action="store_true",
        help="Exit with status 1 when any benchmark record overlaps the training file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, heldout = audit_overlap(args.train_data, args.dataset_dir)
    print(json.dumps({"train_data": os.path.abspath(args.train_data), "benchmarks": report}, indent=2))
    if args.write_heldout_dir:
        write_heldout(heldout, args.write_heldout_dir)
        print("Wrote disjoint benchmark subsets to", args.write_heldout_dir)
    if args.require_disjoint and any(row["overlapping_records"] for row in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
