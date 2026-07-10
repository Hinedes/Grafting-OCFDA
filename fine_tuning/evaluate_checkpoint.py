"""Evaluate a saved Super-Tuning checkpoint on arithmetic benchmarks."""

from __future__ import annotations

import argparse
import json
import os

from .checkpoints import load_checkpoint
from .evaluate import eval_model

BENCHMARKS = ("AddSub", "MultiArith", "SingleEq", "gsm8k", "AQuA", "SVAMP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_model", default="", help="Override the base model saved in checkpoint metadata.")
    parser.add_argument("--datasets", default=",".join(BENCHMARKS))
    parser.add_argument(
        "--dataset_dir",
        default="",
        help="Directory containing <dataset>/test.json files; defaults to the packaged benchmark snapshots.",
    )
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_merge_rosa", dest="merge_rosa", action="store_false")
    parser.set_defaults(merge_rosa=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.checkpoint, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer, metadata = load_checkpoint(
        args.checkpoint,
        base_model=args.base_model or None,
        dtype=args.dtype,
        merge_rosa=args.merge_rosa,
    )

    scores = {}
    for dataset in (item.strip() for item in args.datasets.split(",") if item.strip()):
        scores[dataset] = 100.0 * eval_model(
            dataset,
            model,
            tokenizer,
            dataset_dir=args.dataset_dir or None,
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            verbose=args.verbose,
            progress_path=os.path.join(output_dir, f"{dataset}.jsonl"),
        )
    scores["Average"] = sum(scores.values()) / len(scores)

    result = {"checkpoint": args.checkpoint, "metadata": metadata, "accuracy": scores}
    result_path = os.path.join(output_dir, "accuracy.json")
    with open(result_path, "w") as result_file:
        json.dump(result, result_file, indent=2, sort_keys=True)
    print(json.dumps(scores, indent=2, sort_keys=True))
    print("Wrote", result_path)


if __name__ == "__main__":
    main()
