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
    from .graft_artifact import read_graft_manifest
    from .graft_manage import attach_graft, list_grafts
except ImportError:
    from graft_artifact import read_graft_manifest
    from graft_manage import attach_graft, list_grafts


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


def _cmd_verify(args: argparse.Namespace) -> int:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    try:
        attach_graft(model, args.artifact)
    except RuntimeError as exc:
        print(f"MISMATCH: {exc}")
        return 1
    print(f"MATCH: {os.path.abspath(args.artifact)} matches {args.model}")
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

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
