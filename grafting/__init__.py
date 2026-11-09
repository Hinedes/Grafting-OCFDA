"""Public facade for the standalone Graft framework.

The implementation remains under ``fine_tuning``; this package only re-exports
the stable API and the ``grafting`` CLI entry point.
"""

from fine_tuning.graft_artifact import (
    load_graft,
    read_graft_manifest,
    save_graft,
    verify_graft,
    verify_host,
    verify_host_identity,
)
from fine_tuning.graft_cli import main
from fine_tuning.graft_manage import (
    attach_graft,
    detach_graft,
    enable_graft,
    list_grafts,
    swap_graft,
)
from fine_tuning.graft_train import train_graft

__all__ = [
    "attach_graft",
    "detach_graft",
    "enable_graft",
    "list_grafts",
    "load_graft",
    "main",
    "read_graft_manifest",
    "save_graft",
    "swap_graft",
    "train_graft",
    "verify_graft",
    "verify_host",
    "verify_host_identity",
]
