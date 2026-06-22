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
    "supra-0.3": r"Supra (TopK, $\lambda=0.3$)",
    "supra-0.5": r"Supra (TopK, $\lambda=0.5$)",
    "supra-0.8": r"Supra (TopK, $\lambda=0.8$)",
    "supra-0.3-bottom": r"Supra (BottomK, $\lambda=0.3$)",
    "supra-0.5-bottom": r"Supra (BottomK, $\lambda=0.5$)",
    "supra-0.8-bottom": r"Supra (BottomK, $\lambda=0.8$)",
}

METHOD_MARKERS = {
    "lora": "o",
    "rosa": "s",
    "sift-topk": "^",
    "sift-rand": "v",
    "super-rand": "P",
    "super-wanda": "D",
    "super-wanda-bottom": "D",
    "supra-0.3": "X",
    "supra-0.5": "X",
    "supra-0.8": "X",
    "supra-0.3-bottom": "X",
    "supra-0.5-bottom": "X",
    "supra-0.8-bottom": "X",
}


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(value: str) -> str:
    value = value.split("/")[-1]
    value = value.replace(".", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def latex_float(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-2 or abs(value) >= 1e3:
        mantissa, exponent = f"{value:.0e}".split("e")
        exponent = int(exponent)
        if mantissa == "1":
            return rf"10^{{{exponent}}}"
        return rf"{mantissa}\times 10^{{{exponent}}}"
    return f"{value:g}"


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
                if row.get("event") != "log" or (row.get("loss") is None and row.get("eval_loss") is None):
                    continue
                row["curve_file"] = path
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for column in ["loss", "eval_loss"]:
        if column not in df:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df[df["step"].notna() & (df["loss"].notna() | df["eval_loss"].notna())].copy()
    df["step"] = df["step"].astype(int)
    df["ppl"] = df["loss"].clip(upper=20).map(math.exp)
    df["eval_ppl"] = df["eval_loss"].clip(upper=20).map(math.exp)
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    return df


def plot_one(
    df: pd.DataFrame,
    output_path: str,
    metric: str,
    max_step: int,
    smooth_window: int,
    yscale: str,
    mark_every: int,
    marker_size: float,
) -> None:
    if max_step > 0:
        df = df[df["step"] <= max_step].copy()
    df = df[df[metric].notna()].copy()
    if df.empty:
        raise ValueError("No rows left after filtering.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.figure(figsize=(7.0, 4.2))
    grouped = df.groupby(["method", "method_label", "lr", "step"], as_index=False)[metric].mean()
    for (method, label, lr), group in grouped.groupby(["method", "method_label", "lr"], sort=False):
        group = group.sort_values("step")
        y = group[metric]
        if smooth_window > 1:
            y = y.rolling(window=smooth_window, min_periods=1).mean()
        plt.plot(
            group["step"],
            y,
            linewidth=1.8,
            marker=METHOD_MARKERS.get(method, "o"),
            markersize=marker_size,
            markevery=mark_every if mark_every > 0 else None,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=rf"{label}, $\eta={latex_float(lr)}$",
        )

    ylabels = {
        "loss": "Training loss",
        "ppl": "Training perplexity",
        "eval_loss": "Validation loss",
        "eval_ppl": "Validation perplexity",
    }
    plt.xlabel("Optimizer step")
    plt.ylabel(ylabels[metric])
    plt.yscale(yscale)
    plt.grid(True, which="both", alpha=0.25)
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
    parser.add_argument("--metric", choices=["loss", "ppl", "eval_loss", "eval_ppl", "both", "all"], default="both")
    parser.add_argument("--max_step", type=int, default=-1)
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--yscale", choices=["linear", "log"], default="linear")
    parser.add_argument("--mark_every", type=int, default=12)
    parser.add_argument("--marker_size", type=float, default=4.0)
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

    if args.metric == "both":
        metrics = ["loss", "ppl"]
    elif args.metric == "all":
        metrics = ["loss", "ppl", "eval_loss", "eval_ppl"]
    else:
        metrics = [args.metric]
    formats = parse_csv_list(args.formats)
    for model, model_df in df.groupby("model", sort=False):
        model_name = safe_name(model)
        for metric in metrics:
            if metric not in model_df or model_df[metric].notna().sum() == 0:
                print(f"Skipping {model_name} {metric}: no rows")
                continue
            for fmt in formats:
                output_path = os.path.join(args.output_dir, f"optimization_curve_{model_name}_{metric}.{fmt}")
                plot_one(
                    model_df,
                    output_path,
                    metric,
                    args.max_step,
                    args.smooth_window,
                    args.yscale,
                    args.mark_every,
                    args.marker_size,
                )
                print("Saved", output_path)


if __name__ == "__main__":
    main(parse_args())
