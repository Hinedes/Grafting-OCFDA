"""Run the G0 fixed-prefix NLL check.

Modes:
  tag     build each endpoint with the b1-evidence-v1 get_ocfda_model path
  readme  build each endpoint with grafting.attach_graft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

torch.cuda.set_per_process_memory_fraction(0.9)

import exact_interventions as ex
import fixedprefix_whole as fpw
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = "meta-llama/Llama-3.2-1B"
REVISION = "5d853ed7d16ac794afa8f5c9c7f59f4e9c950954"
ENDPOINTS = {
    "aligned": "Llama-3_2-1B_r8_ocfda-aligned_lr0p0005_seed2001_support1001",
    "independent": "Llama-3_2-1B_r8_ocfda-independent_lr0p0005_seed2001_support1001",
}
EXPECTED_NLL = {
    "aligned": 0.29259259259259257,
    "independent": 0.3300037411148522,
}
EXPECTED_CANONICAL = {
    "aligned": "f526a8cc0b3f1de8a961855fc012b2449e488fd1cb2ce308e94f8e4bb69f2e5d",
    "independent": "52c0f9bb812a1b3d5ac5227ae82ae04ae793af5d95d68c8eae63e2015e2d32c0",
}
PREFIX_ENTRIES = 805
PREFIX_LABELS = 53_460


def read_prefix_manifest(path: Path) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != PREFIX_ENTRIES:
        raise SystemExit(
            f"prefix manifest has {len(rows)} entries, expected {PREFIX_ENTRIES}"
        )
    labels = sum(int(row["valid_length"]) for row in rows)
    if labels != PREFIX_LABELS:
        raise SystemExit(
            f"prefix manifest valid_length sum is {labels}, expected {PREFIX_LABELS}"
        )
    print(f"prefix_manifest entries={len(rows)} valid_length_sum={labels}", flush=True)


def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        HOST, revision=REVISION, trust_remote_code=True
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(mode: str, endpoint: Path):
    if mode == "tag":
        return ex.build_model(str(endpoint))

    from grafting import attach_graft

    model = AutoModelForCausalLM.from_pretrained(
        HOST,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": 0},
    )
    attach_graft(model, str(endpoint))
    model.eval()
    model.config.use_cache = False
    return model, {}


def endpoint_nll(model, batches) -> tuple[float, int]:
    total = 0.0
    count = 0
    for batch in batches:
        logits = ex.model_logits(model, batch)
        loss, labels = ex.ce_from_logits(logits, batch["labels"])
        total += loss
        count += labels
    return total / count, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["tag", "readme"], required=True)
    parser.add_argument("--endpoints-dir", required=True)
    parser.add_argument("--heldout-dir", required=True)
    parser.add_argument("--prefix-manifest", required=True)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--revision", default=REVISION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.host != HOST or args.revision != REVISION:
        raise SystemExit("G0 only supports the frozen Llama-3.2-1B revision")
    if args.mode == "tag" and (args.host != ex.MODEL or args.revision != ex.REV):
        raise SystemExit("tag mode must use exact_interventions.MODEL and REV")

    read_prefix_manifest(Path(args.prefix_manifest))
    fpw.HELDOUT = args.heldout_dir
    ex.fpw.HELDOUT = args.heldout_dir

    tokenizer = load_tokenizer()
    batches = fpw.build_fixed_batches(tokenizer, torch.device("cuda"))
    print(f"batches={len(batches)}", flush=True)

    failed = False
    for name, endpoint_name in ENDPOINTS.items():
        endpoint = Path(args.endpoints_dir) / endpoint_name
        model, metadata = load_model(args.mode, endpoint)
        canonical = ex.canonical(ex.graft_named(model))
        value, labels = endpoint_nll(model, batches)
        expected = EXPECTED_NLL[name]
        nll_ok = f"{value:.8f}" == f"{expected:.8f}"
        canonical_ok = canonical == EXPECTED_CANONICAL[name]
        labels_ok = labels == PREFIX_LABELS
        print(
            f"{name} nll={value:.17g} nll8={value:.8f} "
            f"expected={expected:.17g} match={nll_ok} labels={labels} "
            f"canonical={canonical} canonical_match={canonical_ok} "
            f"geometry={metadata.get('geometry')} k={metadata.get('ocfda_k')} "
            f"support_seed={metadata.get('support_seed')}",
            flush=True,
        )
        failed = failed or not nll_ok or not canonical_ok or not labels_ok
        del model
        torch.cuda.empty_cache()

    print("G0_PASS" if not failed else "G0_FAIL", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
