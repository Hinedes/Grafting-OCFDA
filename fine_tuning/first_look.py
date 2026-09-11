"""Time-boxed exploratory first look: one OCFDA full training + full parallel evaluation.

Reuses the frozen B1 artifact preparation and phase runner; no new training logic.
Intended for platform probes (for example a single CUDA GPU) before the six-run pilot.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from .run_b1 import (
        prepare_artifacts,
        read_jsonl,
        require_rocm,
        run_phase,
        validate_input_artifacts,
    )
except ImportError:
    from run_b1 import (
        prepare_artifacts,
        read_jsonl,
        require_rocm,
        run_phase,
        validate_input_artifacts,
    )


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="ocfda-aligned")
    parser.add_argument("--lr", default="5e-4")
    parser.add_argument("--phase_name", default="")
    parser.add_argument("--output_dir", default=str(repo_dir / "runs" / "b1-first-look"))
    parser.add_argument("--parallel_eval_workers", type=int, default=2)
    parser.add_argument("--allow_cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = args.phase_name or "first_look_" + args.method.split("-")[-1]
    run_args = argparse.Namespace(
        output_dir=os.path.abspath(os.path.expanduser(args.output_dir)),
        train_data=str(Path(__file__).resolve().parents[1] / "fine_tuning" / "ft-training_set" / "math_17k.json"),
        benchmark_dir=str(Path(__file__).resolve().parents[1] / "fine_tuning" / "dataset"),
        parallel_eval_workers=args.parallel_eval_workers,
    )
    require_rocm(allow_cuda=args.allow_cuda)
    heldout = os.path.join(run_args.output_dir, "heldout_dataset")
    if os.path.exists(os.path.join(run_args.output_dir, "artifact_manifest.json")):
        validate_input_artifacts(run_args, heldout)
    else:
        heldout = prepare_artifacts(run_args)

    out_dir = run_phase(
        run_args,
        heldout,
        name,
        args.method,
        args.lr,
        "9001",
        "9001",
        eval_all_lrs=True,
        max_runs=1,
    )
    rows = read_jsonl(os.path.join(out_dir, "run_results.jsonl"))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one first-look result row, got {len(rows)}")
    row = rows[0]
    print(
        json.dumps(
            {
                "method": row.get("method"),
                "lr": row.get("lr"),
                "support_seed": row.get("support_seed"),
                "seed": row.get("seed"),
                "trainable_params": row.get("trainable_params"),
                "accuracy": row.get("accuracy"),
                "lr_tuning": row.get("lr_tuning"),
                "ownership": {
                    key: row.get("ocfda_ownership", {}).get(key, {})
                    for key in ("host", "detach", "optimizer")
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("FIRST_LOOK_PASS", args.method, args.lr, name)


if __name__ == "__main__":
    main()
