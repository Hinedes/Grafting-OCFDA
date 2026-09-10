"""Small SHA-256 helpers for B1 provenance manifests."""

from __future__ import annotations

import hashlib
import os
from typing import Iterable


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(paths: Iterable[str]) -> dict[str, str]:
    files = []
    for path in paths:
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            for root, _, names in os.walk(path):
                files.extend(os.path.join(root, name) for name in names)
        else:
            raise FileNotFoundError(path)
    return {
        os.path.abspath(path): sha256_file(path)
        for path in sorted(files)
    }
