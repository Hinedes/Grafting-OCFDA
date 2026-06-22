import argparse
import glob
import json
import math
import os
import re
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "lora": "LoRA",
    "rosa": "RoSA",
    "sift-topk": "SIFT (TopK)",
    "sift-rand": "SIFT (RandK)",
    "super-rand": "Sparse RandK",
    "super-wanda": "Super (TopK)",
    "super-wanda-bottom": "Super (BottomK)",
    "supra-0.3": "Supra (TopK, lambda=0.3)",
    "supra-0.5": "Supra (TopK, lambda=0.5)",
    "supra-0.8": "Supra (TopK, lambda=0.8)",
    "supra-0.3-bottom": "Supra (BottomK, lambda=0.3)",
    "supra-0.5-bottom": "Supra (BottomK, lambda=0.5)",
    "supra-0.8-bottom": "Supra (BottomK, lambda=0.8)",
}


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(value: str) -> str:
    value = value.split("/")[-1]
    value = value.replace(".", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def iter_curve_files(roots: Iterable[str]):
    for root in roots:
        if os.path.isfile(root):
            yield root
            continue
        for pattern in [
            os.path.join(root, "curves", "*.jsonl"),
            os.path.join(root, "*.jsonl"),
            os.path.join(root, "**", "curves", "*.jsonl"),
        ]:
            yield from glob.glob(pattern, recursive=True)


def load_curves(roots: List[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(set(iter_curve_files(roots))):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("event") != "log" or row.get("loss") is None:
                    continue
                row["curve_file"] = path
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df[df["loss"].notna() & df["step"].notna()].copy()
    df["step"] = df["step"].astype(int)
    df["ppl"] = df["loss"].clip(upper=20).map(math.exp)
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    return df


def plot_one(df: pd.DataFrame, output_path: str, metric: str, max_step: int, smooth_window: int) -> None:
    if max_step > 0:
        df = df[df["step"] <= max_step].copy()
    if df.empty:
        raise ValueError("No rows left after filtering.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.figure(figsize=(7.0, 4.2))
    grouped = df.groupby(["method", "method_label", "lr", "step"], as_index=False)[metric].mean()
    for (_, label, lr), group in grouped.groupby(["method", "method_label", "lr"], sort=False):
        group = group.sort_values("step")
        y = group[metric]
        if smooth_window > 1:
            y = y.rolling(window=smooth_window, min_periods=1).mean()
        plt.plot(group["step"], y, linewidth=1.8, label=f"{label}, lr={lr:g}")

    ylabel = "Training perplexity" if metric == "ppl" else "Training loss"
    plt.xlabel("Optimizer step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot optimization curves saved by run_optimization_curves.py.")
    parser.add_argument("--curve_roots", required=True, help="Comma-separated run roots or JSONL curve files.")
    parser.add_argument("--output_dir", default="optimization_curve_plots")
    parser.add_argument("--models", default="", help="Optional comma-separated model filter.")
    parser.add_argument("--methods", default="", help="Optional comma-separated method filter.")
    parser.add_argument("--metric", choices=["loss", "ppl", "both"], default="both")
    parser.add_argument("--max_step", type=int, default=-1)
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--formats", default="pdf,png")
    return parser.parse_args()


def main(args) -> None:
    df = load_curves(parse_csv_list(args.curve_roots))
    if df.empty:
        raise SystemExit("No curve rows found.")
    if args.models:
        df = df[df["model"].isin(parse_csv_list(args.models))]
    if args.methods:
        df = df[df["method"].isin(parse_csv_list(args.methods))]
    if df.empty:
        raise SystemExit("No curve rows left after filters.")

    metrics = ["loss", "ppl"] if args.metric == "both" else [args.metric]
    formats = parse_csv_list(args.formats)
    for model, model_df in df.groupby("model", sort=False):
        model_name = safe_name(model)
        for metric in metrics:
            for fmt in formats:
                output_path = os.path.join(args.output_dir, f"optimization_curve_{model_name}_{metric}.{fmt}")
                plot_one(model_df, output_path, metric, args.max_step, args.smooth_window)
                print("Saved", output_path)


if __name__ == "__main__":
    main(parse_args())
