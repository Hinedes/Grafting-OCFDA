import sys
from types import SimpleNamespace

import pytest

from fine_tuning.launch_math_methods import parse_args as parse_launcher_args
from fine_tuning.math_experiment_tables import (
    RunSpec,
    parse_args,
    parse_method,
    sparse_param_count,
    supra_param_count,
    train_one_run,
)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("super-wanda-bottom", ("super", "super-bottom", 0.0)),
        ("magnitude-bottomk", ("super", "magnitude-bottom", 0.0)),
        ("supra-0.8-bottom", ("supra", "super-bottom", 0.8)),
        ("supra-magnitude-0.3", ("supra", "magnitude-bottom", 0.3)),
        ("sift-rand", ("sift", "random", 0.0)),
    ],
)
def test_parse_method(method, expected) -> None:
    assert parse_method(method) == expected


def test_parameter_count_helpers() -> None:
    shapes = [(100, 100), (80, 120)]
    sparse_rate = 0.2
    assert sparse_param_count(shapes, sparse_rate, add_one=False) == 3920
    assert supra_param_count(shapes, sparse_rate, lora_params_ratio=0.5) == 3920


def test_json_preset_supplies_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["supertuning-math", "--config", "configs/math17k/llama-1b-1epoch.json"],
    )
    args = parse_args()
    assert args.models == "meta-llama/Llama-3.2-1B"
    assert args.num_epochs == 1
    assert args.batch_size == 16
    assert args.calibration_data == "c4"
    assert "supra-magnitude-0.3" in args.methods


def test_launcher_reads_method_grid_from_preset(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["supertuning-launch", "--config", "configs/math17k/llama-8b-1epoch.json"],
    )
    args = parse_launcher_args()
    assert args.datasets == "AddSub,MultiArith,SingleEq,gsm8k,AQuA,SVAMP"
    assert "magnitude-bottomk" in args.methods
    assert "supra-0.3-bottom" in args.methods


@pytest.mark.parametrize("method", ["supra-1.1", "supra-magnitude--0.1"])
def test_parse_method_rejects_invalid_supra_lambda(method) -> None:
    with pytest.raises(ValueError, match="lambda"):
        parse_method(method)


def test_base_dry_run_does_not_load_model(monkeypatch, tmp_path) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry run loaded model weights")

    monkeypatch.setattr(
        "fine_tuning.math_experiment_tables.AutoModelForCausalLM.from_pretrained",
        fail_if_called,
    )
    args = SimpleNamespace(
        checkpoint_dir=str(tmp_path),
        train_data="math17k.json",
        calibration_data="c4",
        rosa_lora_budget_ratio=0.5,
        dry_run=True,
    )
    spec = RunSpec(seed=0, model="example/model", lora_r=8, lr=0.0, method="base")

    model, tokenizer, output_dir = train_one_run(args, spec, budget_plan={}, target_modules=[])

    assert model is None
    assert tokenizer is None
    assert output_dir.startswith(str(tmp_path))


def test_unified_runner_only_passes_supported_arguments_to_rosa(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supertuning-math",
            "--methods",
            "rosa",
            "--lrs",
            "5e-4",
            "--checkpoint_dir",
            str(tmp_path),
        ],
    )
    args = parse_args()
    captured = {}
    expected_model = object()
    expected_tokenizer = object()

    def fake_train_rosa(**kwargs):
        captured.update(kwargs)
        return expected_model, expected_tokenizer

    monkeypatch.setattr(
        "fine_tuning.math_experiment_tables.import_rosa_train",
        lambda: fake_train_rosa,
    )
    spec = RunSpec(seed=0, model="example/model", lora_r=8, lr=5e-4, method="rosa")
    budget_plan = {"train_sparse_rate": 0.01, "train_lora_r": 4}

    model, tokenizer, _ = train_one_run(args, spec, budget_plan, target_modules=["q_proj"])

    assert model is expected_model
    assert tokenizer is expected_tokenizer
    assert "lora_params_ratio" not in captured
    assert "random_indices" not in captured
    assert captured["adapter_name"] == "rosa"
