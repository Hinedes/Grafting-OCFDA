import json
import os
import sys
from types import SimpleNamespace

import pytest

from fine_tuning.run_b1 import (
    EXPECTED_HELDOUT,
    paired_confirmatory_rows,
    prepare_artifacts,
    require_rocm,
    validate_eval_progress,
    validate_input_artifacts,
    validate_sanity_rows,
)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _row(method: str, support_seed: int, training_seed: int) -> dict:
    return {
        "method": method,
        "support_seed": support_seed,
        "seed": training_seed,
        "accuracy": {"Average": 50.0},
    }


def test_validate_eval_progress_rejects_incomplete_or_duplicated_rows() -> None:
    progress = {
        dataset: [{"idx": index, "flag": bool(index % 2)} for index in range(count)]
        for dataset, count in EXPECTED_HELDOUT.items()
    }
    validate_eval_progress(progress, "sentinel")

    duplicated = {dataset: list(rows) for dataset, rows in progress.items()}
    duplicated["AQuA"].append(duplicated["AQuA"][0])
    with pytest.raises(RuntimeError, match="AQuA"):
        validate_eval_progress(duplicated, "sentinel")

    incomplete = {dataset: list(rows) for dataset, rows in progress.items()}
    incomplete["SVAMP"] = incomplete["SVAMP"][:-1]
    with pytest.raises(RuntimeError, match="SVAMP"):
        validate_eval_progress(incomplete, "sentinel")


def test_confirmatory_rows_require_all_paired_geometries() -> None:
    rows = [
        _row(method, support_seed, training_seed)
        for support_seed in (1001, 1002, 1003)
        for training_seed in (2001, 2002, 2003)
        for method in ("ocfda-aligned", "ocfda-independent")
    ]
    paired = paired_confirmatory_rows(rows)
    assert len(paired) == 9
    assert paired[0]["support_seed"] == 1001
    assert paired[0]["training_seed"] == 2001


def test_sanity_rows_reject_duplicates() -> None:
    rows = [_row("ocfda-aligned", 1001, 2001), _row("ocfda-independent", 1001, 2001)]
    validate_sanity_rows(rows)
    with pytest.raises(RuntimeError, match="Sanity pair"):
        validate_sanity_rows(rows + [rows[0]])


def test_b1_requires_rocm_and_accepts_the_torch_cuda_compatibility_api(monkeypatch) -> None:
    no_hip = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), version=SimpleNamespace(hip=None))
    monkeypatch.setitem(sys.modules, "torch", no_hip)
    with pytest.raises(RuntimeError, match="ROCm/HIP"):
        require_rocm()

    rocm = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), version=SimpleNamespace(hip="6.2", cuda=None))
    monkeypatch.setitem(sys.modules, "torch", rocm)
    require_rocm()


def test_prepare_artifacts_records_and_revalidates_provenance(monkeypatch, tmp_path) -> None:
    snapshot_dir = tmp_path / "model-snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text("{}")

    def fake_snapshot_download(repo_id, revision):
        assert repo_id == "meta-llama/Llama-3.2-1B"
        assert revision == "5d853ed7d16ac794afa8f5c9c7f59f4e9c950954"
        return str(snapshot_dir)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))
    args = SimpleNamespace(
        output_dir=str(tmp_path / "output"),
        train_data=os.path.join(REPO_DIR, "fine_tuning", "ft-training_set", "math_17k.json"),
        benchmark_dir=os.path.join(REPO_DIR, "fine_tuning", "dataset"),
    )

    heldout_dir = prepare_artifacts(args)
    validate_input_artifacts(args, heldout_dir)
    with open(os.path.join(args.output_dir, "artifact_manifest.json"), encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    assert manifest["protocol"] == "B1-OCFDA"
    assert manifest["code_artifacts"]["files"]
    assert manifest["model_snapshot"]["files"]
    (snapshot_dir / "config.json").write_text("tampered")
    with pytest.raises(RuntimeError, match="model snapshot"):
        validate_input_artifacts(args, heldout_dir)
