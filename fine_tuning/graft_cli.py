"""Command-line inspection tools for standalone Graft artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM

try:
    from .graft_artifact import manifest_torch_dtype, read_graft_manifest
    from .graft_manage import attach_graft, list_grafts
    from .graft_train import train_graft
except ImportError:
    from graft_artifact import manifest_torch_dtype, read_graft_manifest
    from graft_manage import attach_graft, list_grafts
    from graft_train import train_graft


def _summary(path: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    supports = manifest.get("supports") or {}
    return {
        "path": os.path.abspath(path),
        "format_version": manifest.get("format_version"),
        "base_model": manifest.get("base_model"),
        "base_model_revision": manifest.get("base_model_revision"),
        "k": manifest.get("k"),
        "geometry": manifest.get("geometry"),
        "seed": manifest.get("seed"),
        "training_seed": manifest.get("training_seed"),
        "dtype": manifest.get("dtype"),
        "provenance": manifest.get("provenance") or {},
        "support_layers": len(supports),
        "host_tensors": len(manifest.get("host_tensor_hashes") or {}),
    }


def _cmd_list(args: argparse.Namespace) -> int:
    entries = list_grafts(args.directory)
    print(json.dumps(entries, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    manifest = read_graft_manifest(args.artifact)
    print(json.dumps(_summary(args.artifact, manifest), indent=2, sort_keys=True, default=str))
    return 0


def _resolve_host_source(model: str, revision: str | None) -> str:
    if os.path.isdir(model):
        return model
    if revision:
        from huggingface_hub import snapshot_download

        return snapshot_download(model, revision=revision, local_files_only=True)
    return model


def _cmd_verify(args: argparse.Namespace) -> int:
    manifest = read_graft_manifest(args.artifact)
    dtype = manifest_torch_dtype(manifest) or torch.float32
    revision = manifest.get("base_model_revision") if args.model == manifest.get("base_model") else None
    source = _resolve_host_source(args.model, revision)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    try:
        attach_graft(model, args.artifact)
    except RuntimeError as exc:
        print(f"MISMATCH: {exc}")
        return 1
    print(f"MATCH: {os.path.abspath(args.artifact)} matches {args.model}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    supports = None
    if args.supports_json:
        with open(args.supports_json, "r", encoding="utf-8") as handle:
            supports = json.load(handle)
    result = train_graft(
        model_id=args.model,
        train_data=args.train_data,
        output_dir=args.output_dir,
        revision=args.revision,
        k=args.k,
        geometry=args.geometry,
        support_seed=args.support_seed,
        supports=supports,
        learning_rate=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        max_examples=args.max_examples,
        max_length=args.max_length,
        loss_on=args.loss_on,
        dtype=args.dtype,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    summary = {key: value for key, value in result.items() if key not in {"model", "tokenizer"}}
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grafting")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List Graft artifacts in a directory")
    list_parser.add_argument("directory")
    list_parser.set_defaults(func=_cmd_list)

    inspect_parser = subparsers.add_parser("inspect", help="Print a Graft manifest summary")
    inspect_parser.add_argument("artifact")
    inspect_parser.set_defaults(func=_cmd_inspect)

    verify_parser = subparsers.add_parser("verify", help="Verify a Graft against a host model")
    verify_parser.add_argument("artifact")
    verify_parser.add_argument("--model", required=True)
    verify_parser.set_defaults(func=_cmd_verify)

    train_parser = subparsers.add_parser("train", help="Train a standalone v2 Graft artifact")
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--revision", default=None)
    train_parser.add_argument("--k", type=int, default=57)
    train_parser.add_argument("--geometry", choices=["aligned", "independent"], default="aligned")
    train_parser.add_argument("--support-seed", type=int, default=None)
    train_parser.add_argument("--supports-json", default=None)
    train_parser.add_argument("--train-data", required=True)
    train_parser.add_argument("--lr", type=float, default=5e-4)
    train_parser.add_argument("--steps", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--max-examples", type=int, default=256)
    train_parser.add_argument("--max-length", type=int, default=128)
    train_parser.add_argument("--loss-on", choices=["response", "all"], default="response")
    train_parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--local-files-only", action="store_true")
    train_parser.add_argument("--output-dir", required=True)
    train_parser.set_defaults(func=_cmd_train)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
