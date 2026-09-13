"""Regression tests for the frozen second-setting (Llama-3.2-3B) driver."""

import copy
import json
import os

import pytest

from fine_tuning import competitive_gate as gate
from fine_tuning import second_setting as ss

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(REPO_DIR, "fine_tuning", "second_setting_manifest.json")


def _manifest() -> dict:
    return ss.load_manifest(MANIFEST_PATH)


def test_manifest_loads_and_budget_arithmetic_matches() -> None:
    manifest = _manifest()
    assert manifest["model"]["model_id"] == "meta-llama/Llama-3.2-3B"
    assert manifest["model"]["model_revision"] == "13afe5124825b4f3751f836b40dafda64c1ed062"
    budget = ss.verify_budget(manifest)
    assert budget["ok"], budget
    assert budget["reference_lora_params"] == 12_156_928
    assert budget["ocfda_params"] == 12_128_256
    assert budget["super_expected_params"] == 12_157_124
    assert budget["closest_k"] == 47
    assert abs(budget["ocfda_error_pct"]) < 0.25


def test_k47_is_the_closest_integer_budget() -> None:
    arch = _manifest()["architecture"]
    reference = ss.reference_lora_params(arch, 8)
    errors = {k: abs(ss.ocfda_params(arch, k) - reference) for k in range(44, 50)}
    assert min(errors, key=errors.get) == 47


def test_matrix_is_nine_runs_with_expected_pairs() -> None:
    manifest = _manifest()
    specs = ss.build_specs(manifest)
    assert len(specs) == 9
    ocfda = [spec for spec in specs if spec["adapter"] == "ocfda"]
    super_specs = [spec for spec in specs if spec["adapter"] == "super"]
    assert len(ocfda) == 6 and len(super_specs) == 3
    assert {(spec["method"], spec["support_seed"], spec["train_seed"]) for spec in ocfda} == {
        ("ocfda-aligned", 1001, 2001),
        ("ocfda-aligned", 1002, 2002),
        ("ocfda-aligned", 1003, 2003),
        ("ocfda-independent", 1001, 2001),
        ("ocfda-independent", 1002, 2002),
        ("ocfda-independent", 1003, 2003),
    }
    for spec in specs:
        assert spec["lr"] == 5e-4
    assert all(spec["support_seed"] is None for spec in super_specs)
    assert {spec["train_seed"] for spec in super_specs} == {2001, 2002, 2003}
    assert all(spec["calibration"] == {"data": "c4", "nsamples": 128, "seed": 228} for spec in super_specs)


def test_run_ids_match_pipeline_convention() -> None:
    manifest = _manifest()
    specs = ss.build_specs(manifest)
    ids = [ss.spec_run_id(manifest, spec) for spec in specs]
    assert "Llama-3_2-3B_r8_ocfda-aligned_lr0p0005_seed2001_support1001" in ids
    assert "Llama-3_2-3B_r8_ocfda-independent_lr0p0005_seed2003_support1003" in ids
    assert "Llama-3_2-3B_r8_super-bottom_lr0p0005_seed2002" in ids
    assert len(set(ids)) == 9


def test_commands_carry_3b_pins_and_transferred_recipe() -> None:
    manifest = _manifest()
    specs = ss.build_specs(manifest)
    for spec in specs:
        command = ss.build_command(manifest, spec, "/run", "/run/artifact_manifest.json", "/run/heldout_dataset")
        ss.assert_frozen_flags(command, manifest)
        ss.assert_method_flags(command, spec)
        assert command[command.index("--models") + 1] == "meta-llama/Llama-3.2-3B"
        assert command[command.index("--model_revision") + 1] == "13afe5124825b4f3751f836b40dafda64c1ed062"
        assert command[command.index("--ocfda_k") + 1] == "47"
        assert command[command.index("--lrs") + 1] == "0.0005"
        assert command[command.index("--num_epochs") + 1] == "3"
        assert command[command.index("--batch_size") + 1] == "16"
        assert command[command.index("--cutoff_len") + 1] == "256"
        assert command[command.index("--generation_num_beams") + 1] == "4"


def test_evaluator_parity_with_validated_1b_flags_except_model_pins() -> None:
    manifest = _manifest()
    spec = ss.build_specs(manifest)[0]
    command = ss.build_command(manifest, spec, "/run", "/run/artifact_manifest.json", "/run/heldout_dataset")
    for flag, expected in gate.FROZEN_ARGV.items():
        if flag in ("--models", "--model_revision", "--tokenizer_revision", "--ocfda_k"):
            continue
        assert command[command.index(flag) + 1] == expected, flag


def test_optimizer_recipes_transfer_from_setting_one() -> None:
    manifest = _manifest()
    specs = ss.build_specs(manifest)
    for spec in specs:
        if spec["adapter"] == "ocfda":
            assert spec["optimizer"] == {"name": "adamw", "weight_decay": 0.0}
            assert spec["calibration"]["data"] == "none"
        else:
            assert spec["optimizer"] == {"name": "adam", "weight_decay": 0.0}
            assert spec["calibration"]["data"] == "c4"


def test_geometry_transfer_verdict_boundaries() -> None:
    strong = ss.geometry_transfer_verdict([55.0, 53.0, 51.0], [50.0, 50.0, 50.0])
    assert strong["passed"] and strong["strong_replication"] and strong["wins"] == 3
    mixed = ss.geometry_transfer_verdict([50.0, 50.0, 53.0], [51.0, 49.0, 50.0])
    assert mixed["passed"] and not mixed["strong_replication"] and mixed["wins"] == 2
    negative = ss.geometry_transfer_verdict([51.0, 52.0, 40.0], [50.0, 50.0, 50.0])
    assert not negative["passed"] and negative["wins"] == 2  # positive wins but negative mean
    few_wins = ss.geometry_transfer_verdict([62.0, 45.0, 45.0], [50.0, 50.0, 50.0])
    assert not few_wins["passed"] and few_wins["wins"] == 1
    inverted = ss.geometry_transfer_verdict([49.0, 48.0, 49.0], [50.0, 51.0, 52.0])
    assert not inverted["passed"] and inverted["wins"] == 0


def test_competitiveness_verdict_boundaries() -> None:
    assert ss.competitiveness_verdict(52.0, 51.0)["passed"]
    assert ss.competitiveness_verdict(49.5, 51.0)["passed"]  # exactly at tolerance
    assert not ss.competitiveness_verdict(48.9, 51.0)["passed"]
    assert ss.competitiveness_verdict(55.0, 51.0)["passed"]


def test_manifest_validation_rejects_drift() -> None:
    manifest = _manifest()
    bad = copy.deepcopy(manifest)
    bad["runs"]["lr"] = 0.001
    with pytest.raises(ValueError, match="5e-4"):
        ss.validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["runs"]["super-bottom"]["calibration"] = {"data": "c4", "nsamples": 64, "seed": 228}
    with pytest.raises(ValueError, match="calibration"):
        ss.validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["frozen_eval"]["benchmarks"] = ["AddSub"]
    with pytest.raises(ValueError, match="benchmarks"):
        ss.validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["fixed_training"]["target_modules"] = ["q_proj"]
    with pytest.raises(ValueError, match="target_modules"):
        ss.validate_manifest(bad)


def test_frozen_run_dir_refuses_changed_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    run_dir = tmp_path / "run"
    ss.freeze_run_dir(str(run_dir), str(manifest_path))
    ss.freeze_run_dir(str(run_dir), str(manifest_path))  # idempotent when unchanged
    manifest_path.write_text(json.dumps({**_manifest(), "runs": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after freezing"):
        ss.freeze_run_dir(str(run_dir), str(manifest_path))
