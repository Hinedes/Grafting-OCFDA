import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(1, REPO_DIR)

from src.datasets_loader import get_loaders  # noqa: E402
from src.mask import prepare_super_mask  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity-check Wanda calibration samples and statistics.")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--calibration_data", default="ft-training_set/math_10k.json")
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=228)
    parser.add_argument("--sparse_rate", type=float, default=0.005962171052631579)
    parser.add_argument("--max_layers", type=int, default=4)
    parser.add_argument("--show_samples", type=int, default=2)
    parser.add_argument("--sample_tokens", type=int, default=160)
    parser.add_argument("--out_json", default="")
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser.parse_args()


def print_sample_windows(tokenizer, calibration_data, nsamples, seed, seqlen, show_samples, sample_tokens):
    trainloader, _ = get_loaders(
        calibration_data,
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
    )
    print(f"Loaded {len(trainloader)} calibration windows from {calibration_data}")
    for idx, (input_ids, _) in enumerate(trainloader[:show_samples]):
        ids = input_ids[0, :sample_tokens]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        text = text.replace("\n", "\\n")
        print(f"\n--- sample {idx} first {sample_tokens} tokens ---")
        print(text[:1200])


def summarize_and_validate(stats):
    failures = []
    if stats["actual_nsamples"] != stats["requested_nsamples"]:
        failures.append(
            f"captured {stats['actual_nsamples']} samples, expected {stats['requested_nsamples']}"
        )
    if not stats["layers"]:
        failures.append("no layer statistics were collected")

    print("\nWanda statistic summary:")
    for layer in stats["layers"]:
        selected_ok = layer["selected_unique_count"] == layer["train_num"]
        print(
            f"layer={layer['layer']:02d} {layer['name']:<12} "
            f"shape={layer['weight_shape']} k={layer['train_num']} "
            f"scaler_mean={layer['scaler_mean']:.6g} "
            f"scaler_zero={layer['scaler_zero_frac']:.4f} "
            f"metric_mean={layer['metric_mean']:.6g} "
            f"metric_max={layer['metric_max']:.6g} "
            f"selected_min={layer['selected_metric_min']:.6g} "
            f"finite={layer['scaler_finite'] and layer['metric_finite']} "
            f"unique={selected_ok}"
        )

        if not layer["scaler_finite"]:
            failures.append(
                f"non-finite activation scaler in layer {layer['layer']} {layer['name']} "
                f"(nan={layer['scaler_nan_count']}, inf={layer['scaler_inf_count']})"
            )
        if not layer["metric_finite"]:
            failures.append(
                f"non-finite Wanda metric in layer {layer['layer']} {layer['name']} "
                f"(nan={layer['metric_nan_count']}, inf={layer['metric_inf_count']})"
            )
        if layer["metric_max"] <= 0.0:
            failures.append(f"non-positive Wanda metric in layer {layer['layer']} {layer['name']}")
        if layer["selected_unique_count"] != layer["train_num"]:
            failures.append(
                f"duplicate selected indices in layer {layer['layer']} {layer['name']} "
                f"({layer['selected_unique_count']} unique for k={layer['train_num']})"
            )

    if failures:
        print("\nFAILED calibration sanity check:")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(1)

    print("\nPASSED: calibration samples and Wanda statistics are finite, nonzero, and internally consistent.")


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=False,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map={"": 0} if device.type == "cuda" else None,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    model.seqlen = model.config.max_position_embeddings

    print("model:", args.model)
    print("calibration_data:", args.calibration_data)
    print("nsamples:", args.nsamples)
    print("seed:", args.seed)
    print("model.seqlen:", model.seqlen)
    print("sparse_rate:", args.sparse_rate)
    print("device:", device)

    print_sample_windows(
        tokenizer=tokenizer,
        calibration_data=args.calibration_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        show_samples=args.show_samples,
        sample_tokens=args.sample_tokens,
    )

    stats = prepare_super_mask(
        model=model,
        tokenizer=tokenizer,
        dev=model.device,
        sparse_rate=args.sparse_rate,
        nsamples=args.nsamples,
        seed=args.seed,
        calibration_data=args.calibration_data,
        collect_stats=True,
        max_layers=args.max_layers,
    )

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        print("wrote:", args.out_json)

    summarize_and_validate(stats)


if __name__ == "__main__":
    main()
