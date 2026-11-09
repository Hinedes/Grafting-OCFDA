"""Six-endpoint fixed-prefix whole-model post-hoc analysis (no generation).

For each frozen nonlinear endpoint g*:
  view 1: f_nl(x; g*)   = ordinary nonlinear forward with g*
  view 3: f_post(x; g*) = f(x; 0) + J_g(x)|_{g=0} g*   (whole-model fixed-anchor)

on identical fixed held-out token sequences (all 805 examples):
  - gold-continuation NLL under each view;
  - whole-logit divergence between views (relative L2, cosine, mean/max abs);
  - per-pair aligned-vs-independent geometry contrast for each view.
No autoregressive generation, no optimizer.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "/shared-docker/grafting-research")
sys.path.insert(0, "/shared-docker/b1-logs/posthoc")


WORKTREE = "/shared-docker/ocfda-b1-rerun"
ENDPOINTS = os.path.join(WORKTREE, "runs/b1-nonlinear-rerun/confirmatory")
HELDOUT = os.path.join(WORKTREE, "runs/b1-nonlinear-rerun/heldout_dataset")
OUT_DIR = "/shared-docker/b1-logs/posthoc/fixedprefix"
DATASETS = ("AddSub", "MultiArith", "SingleEq", "gsm8k", "AQuA", "SVAMP")
PAIRS = ((1001, 2001), (1002, 2002), (1003, 2003))
METHODS = ("ocfda-aligned", "ocfda-independent")
MAX_LENGTH = 256
BATCH_SIZE = 16


def build_fixed_batches(tokenizer, device):
    from fine_tuning.evaluate import generate_prompt

    sequences = []
    for dataset in DATASETS:
        with open(os.path.join(HELDOUT, dataset, "test.json"), "r", encoding="utf-8") as handle:
            records = json.load(handle)
        for index, record in enumerate(records):
            prompt = generate_prompt(record.get("instruction", ""), record.get("input"))
            answer = str(record.get("output", ""))
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"][:MAX_LENGTH]
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            sequences.append({"dataset": dataset, "idx": index, "ids": full_ids, "labels": labels})
    batches = []
    for start in range(0, len(sequences), BATCH_SIZE):
        chunk = sequences[start : start + BATCH_SIZE]
        target = max(len(item["ids"]) for item in chunk)
        target += (-target) % 8
        input_ids, attention, labels = [], [], []
        for item in chunk:
            pad = target - len(item["ids"])
            input_ids.append([0] * pad + item["ids"])
            attention.append([0] * pad + [1] * len(item["ids"]))
            labels.append([-100] * pad + item["labels"])
        batches.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
        })
    return batches


@torch.no_grad()
def endpoint_metrics(model, batches, view: str):
    sum_nll = 0.0
    label_tokens = 0
    for batch in batches:
        kwargs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"], "labels": batch["labels"]}
        if view == "nl":
            loss = model.ocfda_original_forward(**kwargs).loss
        else:
            loss = model(**kwargs).loss
        n_labels = int((batch["labels"] != -100).sum().item())
        sum_nll += float(loss.item()) * n_labels
        label_tokens += n_labels
    return {"nll_gold": sum_nll / label_tokens if label_tokens else float("nan"), "label_tokens": label_tokens}


@torch.no_grad()
def divergence(model, batches):
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    sum_abs = 0.0
    max_abs = 0.0
    count = 0
    for batch in batches:
        ids, mask = batch["input_ids"], batch["attention_mask"]
        a = model.ocfda_original_forward(input_ids=ids, attention_mask=mask).logits.float()
        b = model(input_ids=ids, attention_mask=mask).logits.float()
        valid = mask.bool()
        a = a[valid]
        b = b[valid]
        diff = (b - a).abs()
        dot += float((a * b).sum().item())
        norm_a += float((a * a).sum().item())
        norm_b += float((b * b).sum().item())
        sum_abs += float(diff.sum().item())
        max_abs = max(max_abs, float(diff.max().item()))
        count += a.shape[0]
    return {
        "relative_l2": (max(norm_a + norm_b - 2 * dot, 0.0) ** 0.5) / (norm_a ** 0.5),
        "cosine": dot / ((norm_a ** 0.5) * (norm_b ** 0.5)),
        "mean_abs": sum_abs / count,
        "max_abs": max_abs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", revision="5d853ed7d16ac794afa8f5c9c7f59f4e9c950954", trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    device = torch.device("cuda", 0)
    batches = build_fixed_batches(tokenizer, device)
    print(f"fixed batches={len(batches)}", flush=True)

    results = {}
    for method in METHODS:
        for support, train in PAIRS:
            run = run_id(method, support, train)
            model = build_endpoint_model(os.path.join(ENDPOINTS, "checkpoints", run))
            nl = endpoint_metrics(model, batches, "nl")
            post = endpoint_metrics(model, batches, "post")
            div = divergence(model, batches)
            results[run] = {"method": method, "support_seed": support, "training_seed": train, "nll_nl": nl["nll_gold"], "nll_post": post["nll_gold"], "nll_delta": post["nll_gold"] - nl["nll_gold"], "divergence": div}
            print(f"{run} nll_nl={nl['nll_gold']:.4f} nll_post={post['nll_gold']:.4f} d={post['nll_gold']-nl['nll_gold']:+.4f} relL2={div['relative_l2']:.4f} cos={div['cosine']:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()

    contrasts = {}
    for support, train in PAIRS:
        a = results[run_id("ocfda-aligned", support, train)]
        i = results[run_id("ocfda-independent", support, train)]
        contrasts[str(support)] = {
            "nll_contrast_independent_minus_aligned_nl": i["nll_nl"] - a["nll_nl"],
            "nll_contrast_independent_minus_aligned_post": i["nll_post"] - a["nll_post"],
        }
    summary = {
        "endpoints": results,
        "pair_contrasts": contrasts,
        "mean_contrast_nl": sum(v["nll_contrast_independent_minus_aligned_nl"] for v in contrasts.values()) / len(contrasts),
        "mean_contrast_post": sum(v["nll_contrast_independent_minus_aligned_post"] for v in contrasts.values()) / len(contrasts),
    }
    with open(os.path.join(args.output_dir, "whole_posthoc_fixedprefix.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({"mean_contrast_nl": summary["mean_contrast_nl"], "mean_contrast_post": summary["mean_contrast_post"]}, indent=2), flush=True)
    print("FIXEDPREFIX_WHOLE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
