"""Standalone Graft artifact serialization.

A Graft is the OCFDA adapter state owned by a frozen host model: the
``graft_delta`` parameters plus the ``graft_support`` index buffers.  This
module stores that state independently of the B1 experiment protocol, so a
Graft can be saved and loaded without reconstructing a full Super-Tuning
checkpoint directory.

The artifact is a directory containing:

* ``graft.safetensors`` -- only the Graft tensors (``graft_delta`` and
  ``graft_support``).
* ``graft_manifest.json`` -- format version, host tensor hashes, supports,
  k/geometry/seed, and optional provenance metadata.

Format version 1 Super-Tuning checkpoints (``model.safetensors`` +
``supertuning_config.json``) are also readable.  Their metadata is mapped into
the same in-memory manifest before loading.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Iterable, Mapping, Optional

import torch
from safetensors.torch import load_file, save_file

try:
    from .ocfda import (
        OCFDA_PARAMETER_MARKER,
        OCFDA_PROJECTIONS,
        OCFDA_SUPPORT_MARKER,
        get_ocfda_model_state_dict,
        host_tensor_hashes,
        verify_host_tensor_hashes,
    )
except ImportError:
    from ocfda import (
        OCFDA_PARAMETER_MARKER,
        OCFDA_PROJECTIONS,
        OCFDA_SUPPORT_MARKER,
        get_ocfda_model_state_dict,
        host_tensor_hashes,
        verify_host_tensor_hashes,
    )

GRAFT_FORMAT_VERSION = 2
GRAFT_MANIFEST_NAME = "graft_manifest.json"
GRAFT_TENSORS_NAME = "graft.safetensors"
LEGACY_MANIFEST_NAME = "supertuning_config.json"
GRAFT_MARKERS = (OCFDA_PARAMETER_MARKER, OCFDA_SUPPORT_MARKER)


def _indices(values: Iterable[Any]) -> list[int]:
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    return sorted(int(value) for value in values)


def _normalise_supports(supports: Optional[Mapping[str, Mapping[str, Iterable[int]]]]) -> dict[str, dict[str, list[int]]]:
    if not isinstance(supports, Mapping) or not supports:
        raise ValueError("Graft supports must be a non-empty layer -> projection -> indices mapping")
    normalised: dict[str, dict[str, list[int]]] = {}
    for layer, layer_supports in supports.items():
        if not isinstance(layer_supports, Mapping) or set(layer_supports) != set(OCFDA_PROJECTIONS):
            raise ValueError(f"Graft supports for layer {layer!r} must define {OCFDA_PROJECTIONS}")
        normalised[str(layer)] = {projection: _indices(layer_supports[projection]) for projection in OCFDA_PROJECTIONS}
    return normalised


def _support_mismatch(expected: Mapping[str, Mapping[str, list[int]]], actual: Mapping[str, Mapping[str, list[int]]]) -> str:
    for layer in sorted(expected, key=lambda value: int(value)):
        if layer not in actual:
            return f"layer {layer} is missing from the host"
        for projection in OCFDA_PROJECTIONS:
            if expected[layer][projection] != actual[layer][projection]:
                return f"layer {layer} {projection}"
    return "unknown layer"


def _strict_b1_constants() -> tuple[str, str, int, set[int], set[int]]:
    try:
        from .checkpoints import (
            B1_MODEL_ID,
            B1_MODEL_REVISION,
            B1_OCFDA_K,
            B1_OCFDA_SUPPORT_SEEDS,
            B1_OCFDA_TRAINING_SEEDS,
        )
    except ImportError:
        from checkpoints import (
            B1_MODEL_ID,
            B1_MODEL_REVISION,
            B1_OCFDA_K,
            B1_OCFDA_SUPPORT_SEEDS,
            B1_OCFDA_TRAINING_SEEDS,
        )
    return B1_MODEL_ID, B1_MODEL_REVISION, B1_OCFDA_K, B1_OCFDA_SUPPORT_SEEDS, B1_OCFDA_TRAINING_SEEDS


def manifest_torch_dtype(manifest: Mapping[str, Any]) -> Optional[torch.dtype]:
    """Return the host dtype recorded by an artifact, defaulting v1 to B1's bf16."""

    name = manifest.get("dtype")
    if name:
        name = str(name).replace("torch.", "")
        dtype = getattr(torch, name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported Graft dtype: {manifest.get('dtype')!r}")
        return dtype
    if manifest.get("format_version") == 1:
        extra = manifest.get("extra_metadata") or {}
        return torch.bfloat16 if extra.get("bf16", True) else torch.float16
    return None


def _check_strict_b1(manifest: Mapping[str, Any]) -> None:
    model_id, revision, k, support_seeds, training_seeds = _strict_b1_constants()
    if manifest.get("base_model") != model_id:
        raise RuntimeError(
            f"strict_b1: base model {manifest.get('base_model')!r} does not match {model_id!r}"
        )
    if manifest.get("base_model_revision") != revision:
        raise RuntimeError(
            "strict_b1: base model revision "
            f"{manifest.get('base_model_revision')!r} does not match {revision!r}"
        )
    if int(manifest.get("k") or -1) != k:
        raise RuntimeError(f"strict_b1: k={manifest.get('k')!r} does not match {k}")
    if int(manifest.get("seed") or -1) not in support_seeds:
        raise RuntimeError(f"strict_b1: support seed {manifest.get('seed')!r} is not a frozen B1 seed")
    training_seed = manifest.get("training_seed")
    if training_seed is None or int(training_seed) not in training_seeds:
        raise RuntimeError(f"strict_b1: training seed {training_seed!r} is not a frozen B1 seed")


def _load_legacy_state(path: str) -> dict[str, torch.Tensor]:
    try:
        from .checkpoints import _load_adapter_state
    except ImportError:
        from checkpoints import _load_adapter_state
    state = _load_adapter_state(path)
    return {
        name: tensor
        for name, tensor in state.items()
        if any(marker in name for marker in GRAFT_MARKERS)
    }


def read_graft_manifest(path: str) -> dict[str, Any]:
    """Read and normalize a v1 or v2 Graft manifest without loading tensors."""

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Graft path is not a directory: {path}")

    manifest_path = os.path.join(path, GRAFT_MANIFEST_NAME)
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if manifest.get("format_version") != GRAFT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported Graft format version: {manifest.get('format_version')!r}; "
                f"expected {GRAFT_FORMAT_VERSION}"
            )
        if not os.path.isfile(os.path.join(path, GRAFT_TENSORS_NAME)):
            raise FileNotFoundError(f"Missing {GRAFT_TENSORS_NAME} in {path}")
        return manifest

    legacy_path = os.path.join(path, LEGACY_MANIFEST_NAME)
    if not os.path.isfile(legacy_path):
        raise FileNotFoundError(
            f"No {GRAFT_MANIFEST_NAME} or {LEGACY_MANIFEST_NAME} found in {path}"
        )
    with open(legacy_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format version: {metadata.get('format_version')!r}")
    if metadata.get("adapter_name") != "ocfda":
        raise ValueError("Only OCFDA checkpoints can be loaded as a Graft artifact")
    if not any(
        os.path.isfile(os.path.join(path, name))
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {path}")
    ownership = metadata.get("ocfda_ownership", {})
    host_hashes = ownership.get("host", {}).get("hashes")
    if not isinstance(host_hashes, dict) or not host_hashes:
        raise RuntimeError("Format-version-1 checkpoint is missing pretrained host tensor hashes")
    return {
        "format_version": 1,
        "base_model": metadata.get("base_model"),
        "base_model_revision": metadata.get("model_revision"),
        "host_tensor_hashes": host_hashes,
        "supports": metadata.get("ocfda_supports"),
        "k": metadata.get("ocfda_k"),
        "geometry": metadata.get("geometry"),
        "seed": metadata.get("support_seed"),
        "training_seed": metadata.get("training_seed"),
        "dtype": "bfloat16" if metadata.get("bf16", True) else "float16",
        "provenance": {"protocol": metadata["protocol"]} if metadata.get("protocol") else {},
        "extra_metadata": metadata,
    }


def _load_graft_state(path: str, manifest: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if manifest.get("format_version") == 1:
        return _load_legacy_state(path)
    tensors_path = os.path.join(path, GRAFT_TENSORS_NAME)
    if not os.path.isfile(tensors_path):
        raise FileNotFoundError(f"Missing {GRAFT_TENSORS_NAME} in {path}")
    return load_file(tensors_path)


def _read_graft(path: str) -> tuple[dict[str, Any], dict[str, torch.Tensor], int]:
    manifest = read_graft_manifest(path)
    return manifest, _load_graft_state(path, manifest), int(manifest["format_version"])


def verify_host(
    host_model: torch.nn.Module,
    manifest: Mapping[str, Any],
    path: Optional[str] = None,
) -> None:
    """Verify that the host tensors match the manifest without changing the model."""

    expected_hashes = manifest.get("host_tensor_hashes")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise RuntimeError("Graft manifest is missing pretrained host tensor hashes")
    host_state_names = {
        name
        for name, _ in list(host_model.named_parameters()) + list(host_model.named_buffers())
        if not any(marker in name for marker in GRAFT_MARKERS)
    }
    if set(expected_hashes) != host_state_names:
        raise RuntimeError("Graft manifest does not contain a complete pretrained host hash set")
    try:
        verify_host_tensor_hashes(host_model, expected_hashes, exclude=GRAFT_MARKERS)
    except RuntimeError as exc:
        prefix = f"Graft host mismatch for {path}: " if path else "Graft host mismatch: "
        raise RuntimeError(f"{prefix}{exc}") from exc


def save_graft(
    model: torch.nn.Module,
    path: str,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    legacy_checkpoint: bool = False,
) -> dict[str, Any]:
    """Save the Graft owned by ``model`` under ``path``.

    ``legacy_checkpoint=True`` reproduces the format-version-1 Super-Tuning
    checkpoint layout (``model.safetensors`` + ``supertuning_config.json``)
    used by the training pipeline.  The default writes the standalone format
    version 2 artifact (``graft.safetensors`` + ``graft_manifest.json``).
    """

    supports = getattr(model, "ocfda_supports", None)
    if supports is None:
        raise RuntimeError("save_graft requires an OCFDA-wrapped model")
    supports = _normalise_supports(supports)
    state = get_ocfda_model_state_dict(model)
    if not state:
        raise RuntimeError("save_graft found no Graft tensors on the model")

    extra = copy.deepcopy(dict(extra_metadata or {}))
    config = getattr(model, "config", None)
    base_model = extra.get("base_model") or getattr(config, "_name_or_path", None) or None
    revision = (
        extra.get("base_model_revision")
        or extra.get("model_revision")
        or getattr(config, "_commit_hash", None)
        or None
    )
    k = int(getattr(model, "ocfda_k", 0) or len(next(iter(supports.values()))["gate_proj"]))
    geometry = getattr(model, "ocfda_geometry", None) or extra.get("geometry")
    seed = getattr(model, "ocfda_support_seed", None)
    if seed is None:
        seed = extra.get("support_seed", extra.get("seed"))
    host_dtype = next(model.parameters()).dtype
    host_hashes = host_tensor_hashes(model, exclude=GRAFT_MARKERS)

    os.makedirs(path, exist_ok=True)

    if legacy_checkpoint:
        if not hasattr(model, "save_pretrained"):
            raise RuntimeError("legacy_checkpoint=True requires a model with save_pretrained()")
        had_instance_state_dict = "state_dict" in model.__dict__
        previous_state_dict = model.state_dict

        def _graft_state_dict(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            return get_ocfda_model_state_dict(model, previous_state_dict(*args, **kwargs))

        model.state_dict = _graft_state_dict
        try:
            model.save_pretrained(path)
        finally:
            if had_instance_state_dict:
                model.state_dict = previous_state_dict
            else:
                del model.state_dict

        metadata = extra
        metadata.setdefault("format_version", 1)
        metadata.setdefault("adapter_name", "ocfda")
        metadata.setdefault("base_model", base_model)
        metadata.setdefault("model_revision", revision)
        metadata.setdefault("ocfda_k", k)
        metadata.setdefault("ocfda_supports", supports)
        metadata.setdefault("support_seed", seed)
        metadata.setdefault("geometry", geometry)
        ownership = metadata.setdefault("ocfda_ownership", {})
        host = ownership.setdefault("host", {})
        host.setdefault("hashes", host_hashes)
        with open(os.path.join(path, LEGACY_MANIFEST_NAME), "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        return metadata

    manifest = {
        "format_version": GRAFT_FORMAT_VERSION,
        "kind": "graft",
        "base_model": base_model,
        "base_model_revision": revision,
        "host_tensor_hashes": host_hashes,
        "supports": supports,
        "k": k,
        "geometry": geometry,
        "seed": seed,
        "training_seed": extra.get("training_seed"),
        "dtype": str(host_dtype).replace("torch.", ""),
        "provenance": extra.get("provenance")
        or ({"protocol": extra["protocol"]} if extra.get("protocol") else {}),
        "extra_metadata": extra or None,
    }
    cpu_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
    }
    save_file(cpu_state, os.path.join(path, GRAFT_TENSORS_NAME))
    with open(os.path.join(path, GRAFT_MANIFEST_NAME), "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
    return manifest


def verify_graft(
    host_model: torch.nn.Module,
    path: str,
    strict_b1: bool = False,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Verify a Graft artifact against a wrapped host without changing the model.

    Returns the normalized manifest and the validated Graft state dictionary.
    """

    manifest, state, _version = _read_graft(path)
    if strict_b1:
        _check_strict_b1(manifest)
    verify_host(host_model, manifest, path)

    host_supports = getattr(host_model, "ocfda_supports", None)
    if host_supports is None:
        raise RuntimeError("Host model is not OCFDA-wrapped; attach OCFDA before load_graft")
    manifest_supports = _normalise_supports(manifest.get("supports"))
    host_supports = _normalise_supports(host_supports)
    if host_supports != manifest_supports:
        raise RuntimeError(
            "Graft support mismatch at "
            f"{_support_mismatch(manifest_supports, host_supports)}"
        )
    if int(getattr(host_model, "ocfda_k", -1)) != int(manifest.get("k") or -1):
        raise RuntimeError(
            f"Graft k mismatch: host has {getattr(host_model, 'ocfda_k', None)!r}, "
            f"artifact has {manifest.get('k')!r}"
        )

    host_state = host_model.state_dict()
    expected_names = {
        name
        for name in host_state
        if any(marker in name for marker in GRAFT_MARKERS)
    }
    missing = sorted(expected_names - set(state))
    unexpected = sorted(set(state) - expected_names)
    if missing:
        raise RuntimeError(f"Graft is missing {len(missing)} tensors; first: {missing[0]}")
    if unexpected:
        raise RuntimeError(f"Graft contains {len(unexpected)} unexpected tensors; first: {unexpected[0]}")
    for name in expected_names:
        if tuple(state[name].shape) != tuple(host_state[name].shape):
            raise RuntimeError(
                f"Graft tensor shape mismatch for {name}: "
                f"{tuple(state[name].shape)} != {tuple(host_state[name].shape)}"
            )
        if OCFDA_SUPPORT_MARKER in name and not torch.equal(state[name].cpu(), host_state[name].cpu()):
            raise RuntimeError(f"Graft support tensor mismatch for {name}")
    return manifest, state


def load_graft(
    host_model: torch.nn.Module,
    path: str,
    strict_b1: bool = False,
) -> dict[str, Any]:
    """Load a Graft artifact onto an already OCFDA-wrapped host model.

    The host tensor hashes are verified before any tensor is loaded, so a Graft
    produced for a different base model is refused.  ``strict_b1=True`` also
    applies the frozen B1 model/revision/k/seed checks.
    """

    manifest, state = verify_graft(host_model, path, strict_b1=strict_b1)
    host_model.load_state_dict(state, strict=False)
    return manifest
