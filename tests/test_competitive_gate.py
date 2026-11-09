"""Unit/preflight tests for the competitive-gate harness (CPU-only, no training)."""

import copy
import json
import os
import sys

import pytest

from fine_tuning import competitive_gate as gate
from fine_tuning.math_experiment_tables import RunSpec

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(REPO_DIR, "fine_tuning", "competitive_gate_manifest.json")


def _manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_manifest_validates_and_matrix_is_twelve_trainings() -> None:
    manifest = gate.load_manifest(MANIFEST_PATH)
    selection = gate.selection_specs(manifest)
    assert len(selection) == 6  # 2 baselines x 3 LRs x 1 selection seed
    combos = {(spec["baseline"], spec["method"], spec["lr"], spec["train_seed"]) for spec in selection}
    assert combos == {
        ("lora", "lora", 0.0001, 9001),
        ("lora", "lora", 0.0005, 9001),
        ("lora", "lora", 0.001, 9001),
        ("super", "super-bottom", 0.0001, 9001),
        ("super", "super-bottom", 0.0005, 9001),
        ("super", "super-bottom", 0.001, 9001),
    }
    fake_selections = {"lora": {"selected_lr": 0.0005}, "super": {"selected_lr": 0.0005}}
    final = gate.final_specs(manifest, fake_selections)
    assert len(final) == 6  # 2 baselines x 3 final seeds
    assert {(spec["baseline"], spec["train_seed"]) for spec in final} == {
        ("lora", 2001), ("lora", 2002), ("lora", 2003),
        ("super", 2001), ("super", 2002), ("super", 2003),
    }
    assert len(selection) + len(final) == 12


def test_run_ids_match_pipeline_source_of_truth() -> None:
    manifest = gate.load_manifest(MANIFEST_PATH)
    for spec in gate.selection_specs(manifest):
        expected = RunSpec(seed=spec["train_seed"], model=gate.MODEL_ID, lora_r=8, lr=spec["lr"], method=spec["method"], support_seed=None).run_id
        assert gate.spec_run_id(spec) == expected
    assert "support" not in gate.spec_run_id({"method": "lora", "lr": 0.0005, "train_seed": 2001})


def test_mask_randomness_recorded_separately_from_train_seed() -> None:
    assert gate.mask_seed_of("lora", 2001, 228) == (None, "n/a (dense low-rank update, no mask)")
    seed, kind = gate.mask_seed_of("super-bottom", 2001, 228)
    assert (seed, "calibration" in kind) == (228, True)
    seed, kind = gate.mask_seed_of("super-rand", 2001, 228)
    assert (seed, "train-seed" in kind) == (2001, True)


def test_selection_uses_nll_only_and_ignores_accuracy() -> None:
    manifest = gate.load_manifest(MANIFEST_PATH)
    rows = [
        {"method": "lora", "run_id": "a", "lr": 0.0001, "lr_tuning": {"nll": 0.50}, "accuracy": {"Average": 99.0}},
        {"method": "lora", "run_id": "b", "lr": 0.0005, "lr_tuning": {"nll": 0.40}, "accuracy": {"Average": 10.0}},
        {"method": "lora", "run_id": "c", "lr": 0.001, "lr_tuning": {"nll": 0.45}, "accuracy": {"Average": 50.0}},
    ]
    selection = gate.select_lr(manifest, "lora", rows)
    assert selection["selected_lr"] == 0.0005  # best NLL, worst accuracy
    with pytest.raises(ValueError, match="No learning rate"):
        gate.select_lr(manifest, "lora", [{"method": "lora", "run_id": "d", "lr": 0.0001, "lr_tuning": {"nll": float("nan")}}])


def test_selection_uses_b1_point_five_percent_lower_lr_tie_rule() -> None:
    manifest = gate.load_manifest(MANIFEST_PATH)
    rows = [
        {"method": "lora", "run_id": "a", "lr": 0.0001, "lr_tuning": {"nll": 0.4015}},
        {"method": "lora", "run_id": "b", "lr": 0.0005, "lr_tuning": {"nll": 0.4000}},
    ]
    assert gate.select_lr(manifest, "lora", rows)["selected_lr"] == 0.0001  # within 0.5% -> lower LR
    rows[0]["lr_tuning"] = {"nll": 0.4100}
    assert gate.select_lr(manifest, "lora", rows)["selected_lr"] == 0.0005  # outside 0.5% -> best NLL


def test_baselines_use_native_adam_recipe() -> None:
    manifest = gate.load_manifest(MANIFEST_PATH)
    for name in ("lora", "super"):
        baseline = manifest["baselines"][name]
        assert baseline["optimizer"] == {"name": "adam", "weight_decay": 0.0}
        spec = {"phase": "final", "baseline": name, "method": baseline["method"], "lr": 0.0005, "train_seed": 2001}
        command = gate.build_argv(manifest, spec, "/heldout", "/gate", "/b1/artifact_manifest.json", "/train.json")
        gate.assert_evaluator_match(command)
        gate.assert_method_flags(command, baseline)
        assert command[command.index("--optimizer_name") + 1] == "adam"


def test_manifest_rejects_bad_grids_and_support_seeds() -> None:
    manifest = _manifest()
    bad = copy.deepcopy(manifest)
    bad["baselines"]["lora"]["lr_grid"] = []
    with pytest.raises(ValueError, match="lr_grid"):
        gate.validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["baselines"]["lora"]["support_seed"] = 1001
    with pytest.raises(ValueError, match="support seed"):
        gate.validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["baselines"]["lora"]["method"] = "not-a-method"
    with pytest.raises(ValueError, match="Unknown method"):
        gate.validate_manifest(bad)


def _reference_row(support: int, train: int, score: float, method: str = "ocfda-aligned") -> dict:
    return {
        "run_id": f"ref_{support}_{train}",
        "method": method,
        "support_seed": support,
        "seed": train,
        "trainable_params": 5603328,
        "accuracy": {"Average": score},
        "ocfda_ownership": {
            "host": {"match": True},
            "detach": {"detached": True},
            "optimizer": {"only_ocfda": True},
        },
    }


def test_ocfda_reference_loader_verifies_and_rejects_tampering(tmp_path) -> None:
    manifest = gate.load_manifest(MANIFEST_PATH)
    path = tmp_path / "run_results.jsonl"
    rows = [
        _reference_row(1001, 2001, 46.3389),
        _reference_row(1002, 2002, 54.4044),
        _reference_row(1003, 2003, 55.5676),
        _reference_row(1001, 2001, 37.5316, method="ocfda-independent"),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    ref = gate.load_ocfda_reference(manifest, str(path))
    assert ref["scores"] == [46.3389, 54.4044, 55.5676]
    assert abs(ref["mean"] - 52.10) < 0.005

    tampered = [_reference_row(1001, 2001, 46.34), _reference_row(1002, 2002, 54.4044), _reference_row(1003, 2003, 55.5676)]
    path.write_text("\n".join(json.dumps(row) for row in tampered) + "\n")
    with pytest.raises(RuntimeError, match="differ from the frozen manifest"):
        gate.load_ocfda_reference(manifest, str(path))

    path.write_text(json.dumps(_reference_row(1001, 2001, 46.3389)) + "\n")
    with pytest.raises(RuntimeError, match="incomplete"):
        gate.load_ocfda_reference(manifest, str(path))


def test_gate_dir_must_not_overlap_b1_dir(tmp_path) -> None:
    b1_dir = tmp_path / "b1"
    (b1_dir / "heldout_dataset").mkdir(parents=True)
    (b1_dir / "artifact_manifest.json").write_text(json.dumps({"protocol": "B1-OCFDA"}))
    with pytest.raises(RuntimeError, match="overlaps"):
        gate.check_dirs(str(b1_dir), str(b1_dir), MANIFEST_PATH)
    with pytest.raises(RuntimeError, match="overlaps"):
        gate.check_dirs(str(b1_dir), str(b1_dir / "gate"), MANIFEST_PATH)
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    with pytest.raises(RuntimeError, match="not a gate run"):
        gate.check_dirs(str(b1_dir), str(gate_dir), MANIFEST_PATH)


def test_manifest_change_after_freeze_is_refused(tmp_path) -> None:
    b1_dir = tmp_path / "b1"
    (b1_dir / "heldout_dataset").mkdir(parents=True)
    (b1_dir / "artifact_manifest.json").write_text(json.dumps({"protocol": "B1-OCFDA"}))
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    (gate_dir / "gate_manifest.json").write_text(json.dumps({"manifest_sha256": "deadbeef"}))
    with pytest.raises(RuntimeError, match="changed after freezing"):
        gate.check_dirs(str(b1_dir), str(gate_dir), MANIFEST_PATH)


def test_evaluator_argv_carries_frozen_b1_flags() -> None:
    manifest = gate.load_manifest(MANIFEST_PATH)
    name = next(iter(manifest["baselines"]))
    baseline = manifest["baselines"][name]
    spec = {"phase": "selection", "baseline": name, "method": baseline["method"], "lr": float(baseline["lr_grid"][0]), "train_seed": 9001}
    command = gate.build_argv(manifest, spec, "/heldout", "/gate", "/b1/artifact_manifest.json", "/train.json")
    gate.assert_evaluator_match(command)  # raises on drift
    assert "--skip_accuracy_eval" in command  # selection is tuning-only by construction
    final_spec = dict(spec, phase="final")
    final_command = gate.build_argv(manifest, final_spec, "/heldout", "/gate", "/b1/artifact_manifest.json", "/train.json")
    assert "--skip_accuracy_eval" not in final_command
    with pytest.raises(RuntimeError):
        gate.assert_evaluator_match([flag for flag in command if flag != "--models"])


def test_done_marker_lifecycle_and_atomic_record(tmp_path) -> None:
    run_id = "Llama-3_2-1B_r8_lora_lr0p0005_seed2001"
    run_dir, record_path, done_path = gate.record_paths(str(tmp_path), run_id)
    assert not os.path.exists(done_path)
    gate.atomic_write_json(record_path, {"run_id": run_id})
    assert os.path.exists(record_path) and not os.path.exists(record_path + ".tmp")
    with open(done_path, "x") as handle:
        handle.write("ok\n")
    assert os.path.exists(done_path)


def test_memory_sampler_records_peak_and_degrades_gracefully() -> None:
    series = iter([(100, "stub"), (300, "stub"), (200, "stub"), (None, "stub")])
    sampler = gate.MemorySampler(interval_sec=0.01, read_fn=lambda: next(series, (None, "stub")))
    sampler.start()
    sampler.join(timeout=5)
    report = sampler.stop()
    assert report["peak_bytes"] == 300
    assert report["backend"] == "stub"
    assert report["samples"] >= 3

    failing = gate.MemorySampler(interval_sec=0.01, read_fn=lambda: (_ for _ in ()).throw(OSError("no gpu")))
    failing.start()
    failing.join(timeout=5)
    report = failing.stop()
    assert report["peak_bytes"] is None


def test_execute_child_splits_train_eval_from_markers(tmp_path) -> None:
    child = (
        "import sys, time; "
        "print('step 1', flush=True); time.sleep(0.2); "
        "print('LR tuning validation ppl: 5.0 (nll=0.4, examples=120)', flush=True); time.sleep(0.3)"
    )
    sampler = gate.MemorySampler(interval_sec=0.01, read_fn=lambda: (None, "stub"))
    returncode, wall, train_end = gate.execute_child(
        [sys.executable, "-c", child], str(tmp_path), sampler
    )
    assert returncode == 0
    assert train_end is not None and 0.15 < train_end < wall
    assert wall >= 0.5
    console = tmp_path / "console.log"
    assert console.exists() and "LR tuning validation ppl:" in console.read_text()

    sampler2 = gate.MemorySampler(interval_sec=0.01, read_fn=lambda: (None, "stub"))
    returncode, _, train_end = gate.execute_child(
        [sys.executable, "-c", "print('no markers here')"], str(tmp_path / "nomark"), sampler2
    )
    assert returncode == 0 and train_end is None


def test_gate_verdict_applies_predeclared_rule() -> None:
    verdict = gate.gate_verdict(52.0, {"lora": 53.0, "super": 51.0}, 44_826_624, {"lora": 44_826_624, "super": 44_826_624}, None, {"lora": 10**10, "super": 10**10})
    assert (verdict["outcome"], verdict["route"]) == ("PASS", "methods-paper")
    verdict = gate.gate_verdict(50.0, {"lora": 53.0}, 44_826_624, {"lora": 44_826_624}, None, {"lora": 10**10})
    assert verdict["route"] == "stop-peft-route"  # footprint parity, peaks unevaluable for OCFDA
    verdict = gate.gate_verdict(50.0, {"lora": 53.0}, 30_000_000, {"lora": 44_826_624}, None, {"lora": None})
    assert verdict["route"] == "tradeoff-paper"  # >=25% footprint advantage
    verdict = gate.gate_verdict(50.0, {"lora": 53.0}, 44_826_624, {"lora": 44_826_624}, None, {"lora": None})
    assert verdict["route"] == "stop-peft-route"  # footprint parity is measured evidence of no rescue
    verdict = gate.gate_verdict(50.0, {"lora": 53.0}, 0, {}, None, {})
    assert verdict["route"] == "undecided-missing-data"
