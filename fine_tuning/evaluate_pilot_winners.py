"""Evaluate the two pilot winner checkpoints with the fresh two-process B=1 evaluator.

Reads the per-geometry selection produced by the pilot phase (validation NLL only),
resolves each winner's saved checkpoint, verifies seed/parameter invariants, and runs
the validated two-process batch=1 evaluator over all heldout benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os

try:
    from .b1_statistics import write_json
    from .parallel_eval import evaluate_checkpoint_parallel
except ImportError:
    from b1_statistics import write_json
    from parallel_eval import evaluate_checkpoint_parallel


B1_BENCHMARKS = ("AddSub", "MultiArith", "SingleEq", "gsm8k", "AQuA", "SVAMP")
PILOT_SEED = 9001
EXPECTED_TRAINABLE_SCALARS = 5_603_328
MODEL_NAME = "Llama-3_2-1B"
LORA_R = 8


def find_winner_checkpoint(output_dir: str, method: str, lr: float, seed: int, support_seed: int) -> str:
    lr_text = f"{lr:g}".replace("-", "m").replace(".", "p")
    support = f"_support{support_seed}" if method.startswith(("ocfda-", "graft-")) else ""
    run_id = f"{MODEL_NAME}_r{LORA_R}_{method}_lr{lr_text}_seed{seed}{support}"
    checkpoint_dir = os.path.join(output_dir, "pilot", "checkpoints", run_id)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Winner checkpoint not found: {checkpoint_dir}")
    return checkpoint_dir


def validate_winner_metadata(checkpoint_dir: str, expected_geometry: str) -> dict:
    metadata_path = os.path.join(checkpoint_dir, "supertuning_config.json")
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("protocol") != "B1-OCFDA":
        raise RuntimeError(f"Winner checkpoint is not a B1 OCFDA checkpoint: {checkpoint_dir}")
    if metadata.get("geometry") != expected_geometry:
        raise RuntimeError(f"Winner checkpoint geometry mismatch: {metadata.get('geometry')!r}")
    if int(metadata.get("support_seed", -1)) != PILOT_SEED or int(metadata.get("training_seed", -1)) != PILOT_SEED:
        raise RuntimeError(f"Winner checkpoint does not use pilot seed {PILOT_SEED}: {checkpoint_dir}")
    if int(metadata.get("ocfda_trainable_scalars", -1)) != EXPECTED_TRAINABLE_SCALARS:
        raise RuntimeError(f"Winner checkpoint trainable scalars do not match the frozen budget: {checkpoint_dir}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/b1-pilot")
    parser.add_argument("--datasets", default=",".join(B1_BENCHMARKS))
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_path = os.path.join(args.output_dir, "pilot", "lr_selection.json")
    with open(selection_path, "r", encoding="utf-8") as selection_file:
        selection = json.load(selection_file)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    heldout_dir = os.path.join(args.output_dir, "heldout_dataset")

    summary: dict[str, dict] = {}
    for method, geometry in (("ocfda-aligned", "aligned"), ("ocfda-independent", "independent")):
        if method not in selection:
            raise RuntimeError(f"Pilot selection is missing {method}: {selection_path}")
        selected_lr = float(selection[method]["selected_lr"])
        checkpoint_dir = find_winner_checkpoint(args.output_dir, method, selected_lr, PILOT_SEED, PILOT_SEED)
        validate_winner_metadata(checkpoint_dir, geometry)
        eval_dir = os.path.join(args.output_dir, "pilot", "winner_eval", geometry)
        scores = evaluate_checkpoint_parallel(
            checkpoint_dir,
            datasets,
            heldout_dir,
            eval_dir,
            workers=args.workers,
        )
        summary[geometry] = {
            "method": method,
            "selected_lr": selected_lr,
            "selected_nll": selection[method].get("selected_nll"),
            "checkpoint_dir": checkpoint_dir,
            "eval_progress_dir": eval_dir,
            "accuracy": {dataset: float(scores[dataset]) * 100.0 for dataset in datasets},
            "accuracy_average": float(scores["Average"]) * 100.0,
        }
        print(
            f"{geometry}: lr={selected_lr:g} average={summary[geometry]['accuracy_average']:.4f}",
            flush=True,
        )
    output_path = os.path.join(args.output_dir, "pilot", "winner_evaluation.json")
    write_json(output_path, summary)
    print("WROTE", output_path)


if __name__ == "__main__":
    main()
