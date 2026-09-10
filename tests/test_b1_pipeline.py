"""Local B1 pipeline seam rehearsal.

These tests exercise the experiment wiring (run specs, budget planning,
evaluation progress, answer extraction, result rows, table aggregation, and
seeded validation splits) without loading the pinned Llama model. The literal
end-to-end execution remains the MI300X ``smoke_2batch`` phase.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from fine_tuning.evaluate import eval_model
from fine_tuning.math_experiment_tables import (
    FULL_LLAMA_TARGET_MODULES,
    RunSpec,
    build_budget_plan,
    iter_run_specs,
    load_lr_tuning_records,
    make_result_row,
    save_tables,
)

FROZEN_SCALARS = 5_603_328
FROZEN_LORA_REFERENCE = 5_636_096
FROZEN_TOKEN_DENSE = 3 * 16 * 2048 * 8192


class FrozenLlamaConfig:
    num_hidden_layers = 16
    hidden_size = 2048
    intermediate_size = 8192
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 64


def test_budget_plan_matches_frozen_formula_for_both_geometries(monkeypatch) -> None:
    monkeypatch.setattr(
        "fine_tuning.math_experiment_tables.AutoConfig.from_pretrained",
        lambda *args, **kwargs: FrozenLlamaConfig(),
    )
    args = SimpleNamespace(
        model_revision="5d853ed7d16ac794afa8f5c9c7f59f4e9c950954",
        sparse_rate_override=None,
        ocfda_k=57,
        budget_tolerance_pct=3.0,
    )
    plans = {}
    for method, geometry in (("ocfda-aligned", "aligned"), ("ocfda-independent", "independent")):
        spec = RunSpec(
            seed=2001,
            model="meta-llama/Llama-3.2-1B",
            lora_r=8,
            lr=5e-4,
            method=method,
            support_seed=1001,
        )
        plan = build_budget_plan(args, spec, FULL_LLAMA_TARGET_MODULES)
        plans[geometry] = plan
        assert plan["adapter_trainable_params_estimate"] == FROZEN_SCALARS
        assert plan["reference_lora_params"] == FROZEN_LORA_REFERENCE
        assert plan["ocfda_k"] == 57
        assert plan["ocfda_geometry"] == geometry
        assert plan["total_sparse_rate"] == pytest.approx(FROZEN_SCALARS / FROZEN_TOKEN_DENSE)
        assert plan["adapter_budget_error_pct"] == pytest.approx(-0.5814, abs=0.001)
    assert (
        plans["aligned"]["adapter_trainable_params_estimate"]
        == plans["independent"]["adapter_trainable_params_estimate"]
    )
    assert plans["aligned"]["total_sparse_rate"] == plans["independent"]["total_sparse_rate"]


def test_iter_run_specs_attaches_support_seeds_only_to_ocfda() -> None:
    args = SimpleNamespace(
        seeds="2001,2002",
        models="meta-llama/Llama-3.2-1B",
        lora_rs="8",
        methods="ocfda-aligned,ocfda-independent,lora",
        lrs="1e-4",
        support_seeds="1001,1002",
        shard_id=0,
        num_shards=1,
    )
    specs = list(iter_run_specs(args))
    assert len(specs) == 10
    assert len({spec.run_id for spec in specs}) == 10
    ocfda_specs = [spec for spec in specs if spec.method.startswith("ocfda-")]
    assert {spec.support_seed for spec in ocfda_specs} == {1001, 1002}
    assert all("_support" in spec.run_id for spec in ocfda_specs)
    assert all(spec.support_seed is None for spec in specs if not spec.method.startswith("ocfda-"))


class ScriptedTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def __call__(self, prompt, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, sequence, skip_special_tokens=True):
        return self.outputs[int(sequence[0])]


class ScriptedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def generate(self, **kwargs):
        sequence = torch.tensor([[self.calls]])
        self.calls += 1
        return SimpleNamespace(sequences=sequence)


def test_eval_model_progress_seam_extracts_answers_and_resumes(tmp_path) -> None:
    records = [
        {"instruction": "What is 2 + 3?", "input": "", "output": "5", "answer": 5},
        {"instruction": "What is 10 - 3?", "input": "", "output": "7", "answer": 7},
        {"instruction": "What is 2 + 2?", "input": "", "output": "4", "answer": 4},
    ]
    dataset_dir = tmp_path / "AddSub"
    dataset_dir.mkdir()
    (dataset_dir / "test.json").write_text(json.dumps(records), encoding="utf-8")
    progress_path = tmp_path / "progress" / "AddSub.jsonl"

    tokenizer = ScriptedTokenizer(["The answer is 5.", "The answer is 9.", "The answer is 4."])
    model = ScriptedModel()
    score = eval_model(
        "AddSub",
        model,
        tokenizer,
        dataset_dir=str(tmp_path),
        progress_path=str(progress_path),
    )
    assert score == pytest.approx(2 / 3)
    rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["idx"] for row in rows] == [0, 1, 2]
    assert [row["flag"] for row in rows] == [True, False, True]

    resumed_model = ScriptedModel()
    resumed = eval_model(
        "AddSub",
        resumed_model,
        ScriptedTokenizer([]),
        dataset_dir=str(tmp_path),
        progress_path=str(progress_path),
    )
    assert resumed == pytest.approx(2 / 3)
    assert resumed_model.calls == 0


def _result_row(method: str, support_seed: int, seed: int, accuracy: float) -> dict:
    return {
        "model": "meta-llama/Llama-3.2-1B",
        "lora_r": 8,
        "method": method,
        "lr": 5e-4,
        "seed": seed,
        "support_seed": support_seed,
        "accuracy": {"AddSub": accuracy, "MultiArith": accuracy, "Average": accuracy},
        "ppl": {"AddSub": 2.0, "MultiArith": 3.0, "Average": 2.5},
        "nll": {"AddSub": 0.69, "MultiArith": 1.10, "Average": 0.9},
        "lr_tuning": {"nll": 0.8, "ppl": 2.2, "examples": 120},
        "trainable_params": FROZEN_SCALARS,
        "train_lora_r": 0,
        "train_sparse_rate": 0.01,
        "adapter_trainable_params_estimate": FROZEN_SCALARS,
        "adapter_budget_error_pct": -0.58,
        "trainable_budget_error_pct": -0.58,
    }


def test_save_tables_keeps_support_seed_runs(tmp_path) -> None:
    rows = [
        _result_row("ocfda-aligned", 1001, 2001, 40.0),
        _result_row("ocfda-aligned", 1002, 2001, 42.0),
        _result_row("ocfda-independent", 1001, 2001, 38.0),
        _result_row("ocfda-independent", 1002, 2001, 39.0),
    ]
    save_tables(str(tmp_path), rows, ["AddSub", "MultiArith"])

    selected = [
        json.loads(line)
        for line in (tmp_path / "selected_run_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(selected) == 4
    assert {row["support_seed"] for row in selected} == {1001, 1002}
    selected_lr = pd.read_csv(tmp_path / "selected_lr_by_method.csv")
    assert set(selected_lr["method"]) == {"ocfda-aligned", "ocfda-independent"}


def test_lr_tuning_split_is_seed_deterministic(tmp_path) -> None:
    path = tmp_path / "tiny.json"
    records = [{"instruction": f"q{index}", "input": "", "output": f"a{index}"} for index in range(40)]
    path.write_text(json.dumps(records), encoding="utf-8")

    first = load_lr_tuning_records(str(path), val_set_size=8, split_seed=42)
    second = load_lr_tuning_records(str(path), val_set_size=8, split_seed=42)
    other = load_lr_tuning_records(str(path), val_set_size=8, split_seed=43)
    assert len(first) == 8
    assert [row["instruction"] for row in first] == [row["instruction"] for row in second]
    assert [row["instruction"] for row in first] != [row["instruction"] for row in other]


def test_result_rows_carry_protocol_and_support_metadata() -> None:
    args = SimpleNamespace(
        train_data="math_17k.json",
        dataset_dir="heldout",
        model_revision="5d853ed7d16ac794afa8f5c9c7f59f4e9c950954",
        tokenizer_revision="5d853ed7d16ac794afa8f5c9c7f59f4e9c950954",
        optimizer_name="adamw",
        weight_decay=0.0,
        artifact_manifest_path="",
        calibration_data="none",
        calibration_nsamples=128,
        calibration_seed=228,
        full_ft_checkpoint="",
        val_split_seed=42,
        ppl_target="gold_output",
        ppl_eval_data="",
        ppl_eval_name="Math17K",
    )
    budget_plan = {
        "budget_lora_r": 8,
        "total_sparse_rate": 0.01,
        "train_sparse_rate": 0.01,
        "train_lora_r": 0,
        "component_lora_ratio": None,
        "target_dense_params": 100,
        "reference_lora_params": FROZEN_LORA_REFERENCE,
        "adapter_trainable_params_estimate": FROZEN_SCALARS,
        "adapter_budget_error_pct": -0.58,
        "is_baseline": False,
        "is_unbudgeted": False,
        "ocfda_k": 57,
        "ocfda_geometry": "aligned",
    }
    spec = RunSpec(
        seed=2001,
        model="meta-llama/Llama-3.2-1B",
        lora_r=8,
        lr=5e-4,
        method="ocfda-aligned",
        support_seed=1001,
    )
    row = make_result_row(
        args,
        spec,
        budget_plan,
        {
            "trainable_params": FROZEN_SCALARS,
            "requires_grad_params": FROZEN_SCALARS,
            "budget_trainable_params": FROZEN_SCALARS,
            "trainable_budget_error_pct": -0.58,
        },
        ["gate_proj", "up_proj", "down_proj"],
        "checkpoint",
        {"nll": 0.8, "ppl": 2.2, "examples": 120},
        "full_eval_all_lrs",
    )
    assert row["support_seed"] == 1001
    assert row["ocfda_geometry"] == "aligned"
    assert row["ocfda_k"] == 57
    assert row["optimizer_name"] == "adamw"
    assert row["weight_decay"] == 0.0
    assert row["artifact_manifest_sha256"] is None
    assert row["dataset_artifacts"] == {}
    assert row["eval_stage"] == "full_eval_all_lrs"
