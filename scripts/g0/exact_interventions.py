"""Exact finite-change decomposition and nonlinear interaction interventions.

Per intermediate coordinate with s = W_g x, u = W_u x, delta_g = dW_g x,
delta_u = dW_u x, and down delta dW_d (columns on the down support), the exact
grafted coordinate is (d + dW_d) phi(s + delta_g) (u + delta_u).  Relative to
the host term d phi(s) u, the finite change decomposes exactly into seven terms:

  T_G   = d [phi(s+dg) - phi(s)] u
  T_U   = d phi(s) du
  T_D   = dd phi(s) u
  T_GU  = d [phi(s+dg) - phi(s)] du
  T_GD  = dd [phi(s+dg) - phi(s)] u
  T_UD  = dd phi(s) du
  T_GUD = dd [phi(s+dg) - phi(s)] du

Interactions exist only where the corresponding deltas overlap at the same
SwiGLU coordinate (full-size zero-padded coordinate vectors; no packed pairing).
Interventions subtract exactly one interaction family from every grafted FFN,
preserving the full nonlinear Transformer and exact F.silu computation.

Phases:
  validate - fp64 algebra checks on real host weights and both support geometries
  run      - NL and NL - {GU, GD, UD, GUD} on six endpoints x 805 fixed prefixes
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = "/home/iclab/grafting-resume/resume-stage/droplet"
POSTHOC = os.path.join(ROOT, "b1-logs", "posthoc")
sys.path.insert(0, ROOT)
sys.path.insert(0, POSTHOC)

import fixedprefix_whole as fpw  # noqa: E402

fpw.HELDOUT = os.path.join(ROOT, "heldout_dataset")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "meta-llama/Llama-3.2-1B"
REV = "5d853ed7d16ac794afa8f5c9c7f59f4e9c950954"
PAIRS = ((1001, 2001), (1002, 2002), (1003, 2003))
METHODS = ("ocfda-aligned", "ocfda-independent")
VARIANTS = ("GU", "GD", "UD", "GUD")
NL_MEAN_CONTRAST = 0.04045859611028635
OUT_DIR = "/home/iclab/grafting-resume/exact-interventions"


def run_id(method, support, train):
    return f"Llama-3_2-1B_r8_{method}_lr0p0005_seed{train}_support{support}"


def canonical(named):
    digest = hashlib.sha256()
    digest.update(b"ocfda-graft-endpoint-v1")
    for name in sorted(named):
        tensor = named[name].detach().to("cpu", copy=True).contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def graft_named(model):
    return {name: p for name, p in model.named_parameters() if "graft_delta" in name}


def build_model(endpoint_dir, device="cuda", dtype=torch.bfloat16, load_state=True):
    from safetensors.torch import load_file

    from fine_tuning.ocfda import get_ocfda_model

    with open(os.path.join(endpoint_dir, "supertuning_config.json"), "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    device_map = {"": 0} if device == "cuda" else None
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REV, torch_dtype=dtype, attn_implementation="eager", device_map=device_map
    )
    model = get_ocfda_model(
        base,
        geometry=metadata["geometry"],
        support_seed=int(metadata["support_seed"]),
        k=int(metadata["ocfda_k"]),
        supports=metadata["ocfda_supports"],
    )
    if load_state:
        state = load_file(os.path.join(endpoint_dir, "model.safetensors"))
        named = dict(model.named_parameters())
        with torch.no_grad():
            for name, tensor in state.items():
                if "graft_delta" in name:
                    named[name].copy_(tensor.to(named[name].dtype))
    model.eval()
    model.config.use_cache = False
    return model, metadata


def _scatter(selected, support, size):
    out = torch.zeros(*selected.shape[:-1], size, dtype=selected.dtype, device=selected.device)
    out.scatter_(-1, support.expand(selected.shape[:-1] + (support.numel(),)), selected)
    return out


def _parts(mlp, x):
    gate, up = mlp.gate_proj, mlp.up_proj
    s = F.linear(x, gate.weight, gate.bias)
    u = F.linear(x, up.weight, up.bias)
    compute_dtype = torch.float32 if s.dtype == torch.bfloat16 else s.dtype
    xc = x.to(compute_dtype)
    dg = _scatter(torch.matmul(xc, gate.graft_delta.to(compute_dtype).transpose(0, 1)), gate.graft_support, s.shape[-1]).to(s.dtype)
    du = _scatter(torch.matmul(xc, up.graft_delta.to(compute_dtype).transpose(0, 1)), up.graft_support, u.shape[-1]).to(s.dtype)
    return s, u, dg, du


def family_output(mlp, x, family):
    down = mlp.down_proj
    s, u, dg, du = _parts(mlp, x)
    compute_dtype = torch.float32 if s.dtype == torch.bfloat16 else s.dtype
    phi_s = F.silu(s)
    delta_phi = F.silu(s + dg) - phi_s
    support = down.graft_support.to(x.device)
    if family == "NONE":
        return F.linear(phi_s * 0.0, down.weight, None)
    if family == "GU":
        return F.linear(delta_phi * du, down.weight, None)
    if family == "GUD":
        hidden = (delta_phi * du).to(compute_dtype).index_select(-1, support)
        return F.linear(hidden, down.graft_delta.to(compute_dtype)).to(s.dtype)
    if family == "GD":
        hidden = (delta_phi * u).to(compute_dtype).index_select(-1, support)
        return F.linear(hidden, down.graft_delta.to(compute_dtype)).to(s.dtype)
    if family == "UD":
        hidden = (phi_s * du).to(compute_dtype).index_select(-1, support)
        return F.linear(hidden, down.graft_delta.to(compute_dtype)).to(s.dtype)
    raise ValueError(family)


def attach_family(model, family):
    if not hasattr(model, "_exact_originals"):
        model._exact_originals = [(layer.mlp, layer.mlp.forward) for layer in model.model.layers]
    for mlp, original in model._exact_originals:

        def make(mlp=mlp, original=original, family=family):
            def forward(x):
                return original(x) - family_output(mlp, x, family)

            return forward

        mlp.forward = make()
    return model


def detach_family(model):
    for mlp, original in getattr(model, "_exact_originals", []):
        mlp.forward = original
    return model


def ce_from_logits(logits, labels):
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    total = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100, reduction="sum"
    )
    return float(total.item()), int((shift_labels != -100).sum().item())


@torch.no_grad()
def model_logits(model, batch):
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits


def divergence_accumulate(entry, a, b, valid):
    a = a.float()[valid]
    b = b.float()[valid]
    entry["dot"] += float((a * b).sum().item())
    entry["na"] += float((a * a).sum().item())
    entry["nb"] += float((b * b).sum().item())
    entry["sum"] += float((b - a).abs().sum().item())
    entry["max"] = max(entry["max"], float((b - a).abs().max().item()))
    entry["count"] += a.numel()


def finish_divergence(entry):
    return {
        "relative_l2": (max(entry["na"] + entry["nb"] - 2 * entry["dot"], 0.0) ** 0.5) / (entry["na"] ** 0.5),
        "cosine": entry["dot"] / ((entry["na"] ** 0.5) * (entry["nb"] ** 0.5)),
        "mean_abs": entry["sum"] / entry["count"],
        "max_abs": entry["max"],
    }


def validate(output_dir):
    report = {}
    for geometry, method in (("aligned", "ocfda-aligned"), ("independent", "ocfda-independent")):
        endpoint = os.path.join(ROOT, "confirmatory", "checkpoints", run_id(method, 1001, 2001))
        model, _ = build_model(endpoint, device="cpu", dtype=torch.float32, load_state=False)
        mlp = model.model.layers[0].mlp.to(torch.float64)
        gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        hidden = gate.weight.shape[1]
        generator = torch.Generator().manual_seed(20261005)
        x = torch.randn(2, 4, hidden, generator=generator, dtype=torch.float64)
        with torch.no_grad():
            gate.graft_delta.copy_(torch.randn(gate.graft_delta.shape, generator=generator, dtype=torch.float64))
            up.graft_delta.copy_(torch.randn(up.graft_delta.shape, generator=generator, dtype=torch.float64))
            down.graft_delta.copy_(torch.randn(down.graft_delta.shape, generator=generator, dtype=torch.float64))

        def compute(dg_scale=1.0, du_scale=1.0, dd_scale=1.0):
            s, u, dg, du = _parts(mlp, x)
            dg, du = dg * dg_scale, du * du_scale
            phi_s = F.silu(s)
            delta_phi = F.silu(s + dg) - phi_s
            support = down.graft_support
            delta_down = down.graft_delta * dd_scale
            h0 = phi_s * u
            h = F.silu(s + dg) * (u + du)

            def down_apply(mat):
                return F.linear(mat, down.weight, None)

            def dd_apply(mat):
                return F.linear(mat.index_select(-1, support), delta_down)

            terms = {
                "T_G": down_apply(delta_phi * u),
                "T_U": down_apply(phi_s * du),
                "T_D": dd_apply(phi_s * u),
                "T_GU": down_apply(delta_phi * du),
                "T_GD": dd_apply(delta_phi * u),
                "T_UD": dd_apply(phi_s * du),
                "T_GUD": dd_apply(delta_phi * du),
            }
            exact = down_apply(h) + dd_apply(h) - down_apply(h0)
            return terms, exact, (h, h0)

        terms, exact, (h, h0) = compute()
        total = sum(terms.values())
        error = (total - exact).abs().max().item()
        scale = max(exact.abs().max().item(), 1e-30)
        zero_terms, zero_exact, (zero_h, zero_h0) = compute(0.0, 0.0, 0.0)
        vanish = {
            "du_zero": compute(du_scale=0.0)[0],
            "dg_zero": compute(dg_scale=0.0)[0],
            "dd_zero": compute(dd_scale=0.0)[0],
        }
        vanish_checks = {
            "du_zero_gu_ud_gud": all(vanish["du_zero"][k].abs().max().item() == 0.0 for k in ("T_GU", "T_UD", "T_GUD")),
            "dg_zero_gu_gd_gud": all(vanish["dg_zero"][k].abs().max().item() == 0.0 for k in ("T_GU", "T_GD", "T_GUD")),
            "dd_zero_gd_ud_gud": all(vanish["dd_zero"][k].abs().max().item() == 0.0 for k in ("T_GD", "T_UD", "T_GUD")),
        }
        report[geometry] = {
            "reconstruction_max_abs_error": error,
            "reconstruction_relative_error": error / scale,
            "zero_graft_all_terms_zero": all(v.abs().max().item() == 0.0 for v in zero_terms.values()),
            "zero_graft_hidden_equals_host": bool(torch.equal(zero_h, zero_h0)),
            "zero_graft_exact_delta_zero": zero_exact.abs().max().item() == 0.0,
            "vanish_checks": vanish_checks,
            "terms_nonzero": {k: float(v.abs().max().item()) for k, v in terms.items()},
            "passed": (
                error / scale < 1e-10
                and all(v.abs().max().item() == 0.0 for v in zero_terms.values())
                and bool(torch.equal(zero_h, zero_h0))
                and all(vanish_checks.values())
            ),
        }
        del model
    with open(os.path.join(output_dir, "validation.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("VALIDATION_DONE", flush=True)
    return 0 if all(report[g]["passed"] for g in report) else 1


def run(output_dir, batches):
    results = {}
    for method in METHODS:
        for support, train in PAIRS:
            run_name = run_id(method, support, train)
            endpoint = os.path.join(ROOT, "confirmatory", "checkpoints", run_name)
            sidecar = json.load(open(os.path.join(ROOT, "confirmatory", "results", "endpoint_hashes", run_name + ".json"), encoding="utf-8"))
            model, _ = build_model(endpoint)
            canonical_before = canonical(graft_named(model))

            # no-op patch check: subtracting nothing must reproduce the nonlinear FFN exactly
            with torch.no_grad():
                reference_logits = model(
                    input_ids=batches[0]["input_ids"], attention_mask=batches[0]["attention_mask"]
                ).logits
                attach_family(model, "NONE")
                noop_logits = model(
                    input_ids=batches[0]["input_ids"], attention_mask=batches[0]["attention_mask"]
                ).logits
                detach_family(model)
            noop_ok = bool(torch.equal(reference_logits, noop_logits))

            record = {"method": method, "support_seed": support, "training_seed": train, "variants": {}, "batches": len(batches)}
            print(f"=== {run_name} loaded ===", flush=True)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            total = 0.0
            count = 0
            for batch in batches:
                detach_family(model)
                logits = model_logits(model, batch)
                loss, labels = ce_from_logits(logits, batch["labels"])
                total += loss
                count += labels
            torch.cuda.synchronize()
            record["nl_wall_sec"] = time.perf_counter() - started
            record["nl_peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
            record["nll_nl"] = total / count
            print(f"{run_name} NL nll={record['nll_nl']:.4f} wall={record['nl_wall_sec']:.1f}s", flush=True)

            for variant in VARIANTS:
                entry = {"dot": 0.0, "na": 0.0, "nb": 0.0, "sum": 0.0, "max": 0.0, "count": 0}
                total = 0.0
                count = 0
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                for batch in batches:
                    detach_family(model)
                    nl_logits = model_logits(model, batch)
                    attach_family(model, variant)
                    variant_logits = model_logits(model, batch)
                    loss, labels = ce_from_logits(variant_logits, batch["labels"])
                    total += loss
                    count += labels
                    divergence_accumulate(entry, nl_logits, variant_logits, batch["attention_mask"].bool())
                torch.cuda.synchronize()
                record["variants"][variant] = {
                    "nll": total / count,
                    "wall_sec": time.perf_counter() - started,
                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                    "divergence_nl_variant": finish_divergence(entry),
                }
                print(f"{run_name} NL-{variant} nll={record['variants'][variant]['nll']:.4f}", flush=True)

            detach_family(model)
            record["canonical_unchanged"] = canonical(graft_named(model)) == sidecar["graft_canonical_sha256"] == canonical_before
            record["noop_patch_ok"] = noop_ok
            results[run_name] = record
            del model
            torch.cuda.empty_cache()

    contrasts = {}
    for support, train in PAIRS:
        aligned = results[run_id("ocfda-aligned", support, train)]
        independent = results[run_id("ocfda-independent", support, train)]
        contrasts[str(support)] = {
            "nl": independent["nll_nl"] - aligned["nll_nl"],
            **{variant: independent["variants"][variant]["nll"] - aligned["variants"][variant]["nll"] for variant in VARIANTS},
        }
    mean_contrast = {"nl": sum(c["nl"] for c in contrasts.values()) / len(contrasts)}
    mean_contrast.update({variant: sum(c[variant] for c in contrasts.values()) / len(contrasts) for variant in VARIANTS})
    summary = {
        "endpoints": results,
        "pair_contrasts": contrasts,
        "mean_contrast": mean_contrast,
        "change_vs_nl": {variant: mean_contrast[variant] - NL_MEAN_CONTRAST for variant in VARIANTS},
        "nl_mean_contrast_reference": NL_MEAN_CONTRAST,
    }
    with open(os.path.join(output_dir, "exact_interventions.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({"mean_contrast": mean_contrast, "change_vs_nl": summary["change_vs_nl"]}, indent=2), flush=True)
    print("EXACT_RUN_DONE", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["validate", "run"], default="validate")
    parser.add_argument("--output_dir", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.phase == "validate":
        return validate(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    batches = fpw.build_fixed_batches(tokenizer, torch.device("cuda"))
    return run(args.output_dir, batches)


if __name__ == "__main__":
    raise SystemExit(main())
