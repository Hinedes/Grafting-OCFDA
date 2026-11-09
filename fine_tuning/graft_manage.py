"""Live Graft management for a loaded host model.

The host is wrapped once with the OCFDA structure described by an artifact
manifest; the Graft itself is the small ``graft_delta``/``graft_support`` state
that can be switched off, switched back on, or replaced without reloading the
host weights.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

try:
    from .graft_artifact import (
        load_graft,
        manifest_torch_dtype,
        read_graft_manifest,
        verify_graft,
        verify_host,
    )
    from .ocfda import get_ocfda_model, get_ocfda_model_state_dict, set_graft_enabled
except ImportError:
    from graft_artifact import (
        load_graft,
        manifest_torch_dtype,
        read_graft_manifest,
        verify_graft,
        verify_host,
    )
    from ocfda import get_ocfda_model, get_ocfda_model_state_dict, set_graft_enabled


def _wrappers(model: torch.nn.Module) -> tuple[Any, ...]:
    return tuple(getattr(model, "ocfda_wrappers", ()))


def _is_wrapped(model: torch.nn.Module) -> bool:
    return bool(_wrappers(model))


def _enabled(model: torch.nn.Module) -> bool:
    wrappers = _wrappers(model)
    if not wrappers:
        raise RuntimeError("Model has no OCFDA wrappers")
    states = {bool(wrapper.graft_enabled) for wrapper in wrappers}
    if len(states) != 1:
        raise RuntimeError("OCFDA wrappers have inconsistent graft_enabled state")
    return states.pop()


def attach_graft(host_model: torch.nn.Module, artifact_path: str) -> torch.nn.Module:
    """Attach the Graft in ``artifact_path`` to ``host_model``.

    If the host is not wrapped yet, the wrapper structure is built from the
    artifact's supports and k; no random support draw is used.  The host is
    verified before it is modified.
    """

    manifest = read_graft_manifest(artifact_path)
    supports = manifest.get("supports")
    k = manifest.get("k")
    geometry = manifest.get("geometry")
    seed = manifest.get("seed")
    if not supports or k is None or geometry is None:
        raise RuntimeError(f"Graft artifact {artifact_path} is missing supports/k/geometry")

    target_dtype = manifest_torch_dtype(manifest)
    if not _is_wrapped(host_model) and target_dtype is not None:
        current_dtype = next(host_model.parameters()).dtype
        if current_dtype != target_dtype:
            host_model.to(dtype=target_dtype)

    verify_host(host_model, manifest, artifact_path)
    if not _is_wrapped(host_model):
        get_ocfda_model(
            host_model,
            geometry=str(geometry),
            support_seed=None if seed is None else int(seed),
            k=int(k),
            supports=supports,
            verify_supports_against_seed=False,
        )
    load_graft(host_model, artifact_path)
    set_graft_enabled(host_model, True)
    return host_model


def detach_graft(model: torch.nn.Module) -> torch.nn.Module:
    """Switch the Graft off in place; the delta stays in memory."""

    set_graft_enabled(model, False)
    return model


def enable_graft(model: torch.nn.Module) -> torch.nn.Module:
    """Switch the Graft back on in place."""

    set_graft_enabled(model, True)
    return model


def swap_graft(model: torch.nn.Module, artifact_path: str) -> torch.nn.Module:
    """Replace the active Graft with ``artifact_path`` atomically.

    The new artifact is fully verified before the model is touched.  A
    different k or support layout is refused: the host must already have the
    same OCFDA structure.  The enabled/disabled state is preserved.
    """

    if not _is_wrapped(model):
        raise RuntimeError("swap_graft requires an attached Graft")
    enabled = _enabled(model)
    try:
        _manifest, new_state = verify_graft(model, artifact_path)
    except RuntimeError as exc:
        raise RuntimeError(f"swap_graft refused {artifact_path}: {exc}") from exc

    previous_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in get_ocfda_model_state_dict(model).items()
    }
    try:
        model.load_state_dict(new_state, strict=False)
    except Exception:
        model.load_state_dict(previous_state, strict=False)
        set_graft_enabled(model, enabled)
        raise
    set_graft_enabled(model, enabled)
    return model


def list_grafts(directory: str, host_model: Optional[torch.nn.Module] = None) -> list[dict[str, Any]]:
    """List v1 and v2 Graft artifacts found under ``directory``.

    Non-artifact files and directories are ignored.  If ``host_model`` is
    provided, each entry also reports whether its host tensor hashes match.
    """

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Artifact directory does not exist: {directory}")

    candidates: list[tuple[str, dict[str, Any]]] = []
    try:
        candidates.append((directory, read_graft_manifest(directory)))
    except (FileNotFoundError, ValueError, RuntimeError, OSError):
        for entry in os.scandir(directory):
            if not entry.is_dir():
                continue
            try:
                candidates.append((entry.path, read_graft_manifest(entry.path)))
            except (FileNotFoundError, ValueError, RuntimeError, OSError):
                continue

    entries = []
    for path, manifest in sorted(candidates, key=lambda item: item[0]):
        item = {
            "path": os.path.abspath(path),
            "format_version": int(manifest.get("format_version", 0)),
            "base_model": manifest.get("base_model"),
            "k": manifest.get("k"),
            "geometry": manifest.get("geometry"),
            "seed": manifest.get("seed"),
            "provenance": manifest.get("provenance") or {},
        }
        if host_model is not None:
            try:
                verify_host(host_model, manifest, path)
                item["host_match"] = True
                item["host_mismatch"] = None
            except RuntimeError as exc:
                item["host_match"] = False
                item["host_mismatch"] = str(exc)
        entries.append(item)
    return entries
