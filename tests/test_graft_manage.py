import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from fine_tuning.graft_artifact import save_graft
from fine_tuning.graft_manage import (
    attach_graft,
    detach_graft,
    enable_graft,
    list_grafts,
    swap_graft,
)
from fine_tuning.graft_train import encode_examples, format_prompt_prefix, train_graft
from fine_tuning.ocfda import get_ocfda_model, get_ocfda_model_state_dict

REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 6, bias=False)
        self.up_proj = torch.nn.Linear(4, 6, bias=False)
        self.down_proj = torch.nn.Linear(6, 4, bias=False)


class TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyMlp()


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])
        self.config = SimpleNamespace(intermediate_size=6, hidden_size=4)

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self.state_dict().items()
        }
        save_file(state, os.path.join(path, "model.safetensors"))


def base_model(seed: int = 3) -> TinyModel:
    torch.manual_seed(seed)
    return TinyModel()


def wrap(model: TinyModel, support_seed: int = 7, k: int = 2) -> torch.nn.Module:
    return get_ocfda_model(model, geometry="aligned", support_seed=support_seed, k=k)


def set_deltas(model: torch.nn.Module, seed: int = 11) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for wrapper in model.ocfda_wrappers:
            wrapper.graft_delta.copy_(torch.randn_like(wrapper.graft_delta))


def trained(seed: int = 3, delta_seed: int = 11, support_seed: int = 7, k: int = 2) -> torch.nn.Module:
    model = wrap(base_model(seed), support_seed, k)
    set_deltas(model, delta_seed)
    return model


def swiglu(mlp: torch.nn.Module, probe: torch.Tensor) -> torch.Tensor:
    return mlp.down_proj(F.silu(mlp.gate_proj(probe)) * mlp.up_proj(probe))


def logits(model: torch.nn.Module, probe: torch.Tensor) -> torch.Tensor:
    return swiglu(model.model.layers[0].mlp, probe)


def meta(extra: dict | None = None) -> dict:
    payload = {"base_model": "test/tiny", "model_revision": "rev1", "training_seed": 2001}
    payload.update(extra or {})
    return payload


def llama_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        pad_token_id=0,
        eos_token_id=31,
    )


def tiny_llama(seed: int = 3) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    return LlamaForCausalLM(llama_config())


def run_cli(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "fine_tuning.graft_cli", *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_attach_graft_reproduces_trained_outputs(tmp_path) -> None:
    model = trained()
    save_graft(model, str(tmp_path / "graft"), extra_metadata=meta())
    probe = torch.randn(4, 4)
    expected = logits(model, probe).detach().clone()

    host = base_model()
    attached = attach_graft(host, str(tmp_path / "graft"))

    assert attached is host
    assert torch.equal(logits(attached, probe), expected)


def test_detach_graft_matches_unwrapped_host(tmp_path) -> None:
    model = trained()
    save_graft(model, str(tmp_path / "graft"), extra_metadata=meta())
    attached = attach_graft(base_model(), str(tmp_path / "graft"))
    probe = torch.randn(4, 4)

    detach_graft(attached)
    unwrapped = base_model()

    assert torch.equal(logits(attached, probe), logits(unwrapped, probe))
    assert all(not wrapper.graft_enabled for wrapper in attached.ocfda_wrappers)


def test_enable_graft_restores_grafted_outputs(tmp_path) -> None:
    model = trained()
    save_graft(model, str(tmp_path / "graft"), extra_metadata=meta())
    attached = attach_graft(base_model(), str(tmp_path / "graft"))
    probe = torch.randn(4, 4)
    expected = logits(model, probe).detach().clone()

    detach_graft(attached)
    enable_graft(attached)

    assert torch.equal(logits(attached, probe), expected)
    assert all(wrapper.graft_enabled for wrapper in attached.ocfda_wrappers)


def test_swap_graft_matches_fresh_attach(tmp_path) -> None:
    model_a = trained(delta_seed=11)
    model_b = trained(delta_seed=12)
    save_graft(model_a, str(tmp_path / "a"), extra_metadata=meta())
    save_graft(model_b, str(tmp_path / "b"), extra_metadata=meta())

    probe = torch.randn(4, 4)
    attached = attach_graft(base_model(), str(tmp_path / "a"))
    before = logits(attached, probe).detach().clone()
    freshly_attached_b = attach_graft(base_model(), str(tmp_path / "b"))
    expected_b = logits(freshly_attached_b, probe).detach().clone()

    swap_graft(attached, str(tmp_path / "b"))

    assert torch.equal(logits(attached, probe), expected_b)
    assert not torch.equal(logits(attached, probe), before)


def test_swap_graft_refuses_other_host_and_preserves_state(tmp_path) -> None:
    model_a = trained(seed=3, delta_seed=11)
    other_host = trained(seed=4, delta_seed=13)
    save_graft(model_a, str(tmp_path / "a"), extra_metadata=meta())
    save_graft(other_host, str(tmp_path / "other"), extra_metadata=meta())

    attached = attach_graft(base_model(seed=3), str(tmp_path / "a"))
    probe = torch.randn(4, 4)
    enabled_output = logits(attached, probe).detach().clone()
    enabled_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in get_ocfda_model_state_dict(attached).items()
    }

    with pytest.raises(RuntimeError, match="swap_graft refused"):
        swap_graft(attached, str(tmp_path / "other"))

    assert torch.equal(logits(attached, probe), enabled_output)
    assert all(wrapper.graft_enabled for wrapper in attached.ocfda_wrappers)
    after_enabled_state = get_ocfda_model_state_dict(attached)
    assert set(after_enabled_state) == set(enabled_state)
    assert all(torch.equal(after_enabled_state[name], enabled_state[name]) for name in enabled_state)

    detach_graft(attached)
    detached_output = logits(attached, probe).detach().clone()
    detached_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in get_ocfda_model_state_dict(attached).items()
    }

    with pytest.raises(RuntimeError, match="swap_graft refused"):
        swap_graft(attached, str(tmp_path / "other"))

    assert torch.equal(logits(attached, probe), detached_output)
    assert all(not wrapper.graft_enabled for wrapper in attached.ocfda_wrappers)
    after_detached_state = get_ocfda_model_state_dict(attached)
    assert set(after_detached_state) == set(detached_state)
    assert all(torch.equal(after_detached_state[name], detached_state[name]) for name in detached_state)


def test_list_grafts_finds_v1_and_v2_and_skips_noise(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    model = trained()
    other_host = trained(seed=4)
    save_graft(model, str(artifacts / "graft-v2"), extra_metadata=meta())
    save_graft(model, str(artifacts / "graft-v1"), extra_metadata=meta(), legacy_checkpoint=True)
    save_graft(other_host, str(artifacts / "other-v2"), extra_metadata=meta())
    (artifacts / "notes.txt").write_text("not an artifact", encoding="utf-8")
    (artifacts / "empty").mkdir()
    (artifacts / "not-graft").mkdir()
    (artifacts / "not-graft" / "supertuning_config.json").write_text(
        json.dumps({"format_version": 1, "adapter_name": "lora"}),
        encoding="utf-8",
    )

    entries = list_grafts(str(artifacts))
    by_name = {Path(entry["path"]).name: entry for entry in entries}

    assert set(by_name) == {"graft-v2", "graft-v1", "other-v2"}
    assert by_name["graft-v2"]["format_version"] == 2
    assert by_name["graft-v1"]["format_version"] == 1
    assert all(entry["k"] == 2 and entry["geometry"] == "aligned" and entry["seed"] == 7 for entry in entries)
    assert all("host_match" not in entry for entry in entries)

    matched = {
        Path(entry["path"]).name: entry
        for entry in list_grafts(str(artifacts), host_model=base_model(seed=3))
    }
    assert matched["graft-v2"]["host_match"] is True
    assert matched["graft-v1"]["host_match"] is True
    assert matched["other-v2"]["host_match"] is False
    assert "host mismatch" in matched["other-v2"]["host_mismatch"]


def test_generate_runs_enabled_and_detached(tmp_path) -> None:
    model = tiny_llama()
    wrapped = get_ocfda_model(model, geometry="aligned", support_seed=7, k=2)
    set_deltas(wrapped, seed=11)
    save_graft(wrapped, str(tmp_path / "graft"), extra_metadata=meta())

    host = tiny_llama()
    attached = attach_graft(host, str(tmp_path / "graft"))
    input_ids = torch.tensor([[5, 6, 7]])

    with torch.no_grad():
        enabled = attached.generate(input_ids, max_new_tokens=2, do_sample=False)
        detach_graft(attached)
        detached = attached.generate(input_ids, max_new_tokens=2, do_sample=False)

    assert enabled.shape[0] == 1 and enabled.shape[1] >= input_ids.shape[1]
    assert detached.shape[0] == 1 and detached.shape[1] >= input_ids.shape[1]


def test_attach_graft_uses_explicit_non_seeded_supports(tmp_path) -> None:
    supports = {
        "0": {"gate_proj": [0, 1], "up_proj": [1, 2], "down_proj": [2, 3]},
        "1": {"gate_proj": [3, 4], "up_proj": [4, 5], "down_proj": [0, 2]},
    }
    torch.manual_seed(3)
    trained = get_ocfda_model(
        TinyModel(),
        geometry="independent",
        support_seed=None,
        k=2,
        supports=supports,
        verify_supports_against_seed=False,
    )
    set_deltas(trained, seed=11)
    save_graft(trained, str(tmp_path / "graft"), extra_metadata=meta())

    probe = torch.randn(4, 4)
    expected = logits(trained, probe).detach().clone()

    host = base_model()
    attached = attach_graft(host, str(tmp_path / "graft"))

    assert torch.equal(logits(attached, probe), expected)


def test_cli_list_inspect_and_verify(tmp_path) -> None:
    model = tiny_llama()
    model_dir = tmp_path / "host"
    model.save_pretrained(str(model_dir))

    wrapped = get_ocfda_model(model, geometry="aligned", support_seed=7, k=2)
    set_deltas(wrapped, seed=11)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    save_graft(
        wrapped,
        str(artifacts / "graft-v2"),
        extra_metadata=meta({"base_model": str(model_dir)}),
    )
    save_graft(
        wrapped,
        str(artifacts / "graft-v1"),
        extra_metadata=meta({"base_model": str(model_dir)}),
        legacy_checkpoint=True,
    )

    bad = tiny_llama(seed=4)
    bad_dir = tmp_path / "bad-host"
    bad.save_pretrained(str(bad_dir))

    result = run_cli("list", str(artifacts))
    assert result.returncode == 0, result.stderr
    assert "graft-v1" in result.stdout and "graft-v2" in result.stdout

    result = run_cli("inspect", str(artifacts / "graft-v2"))
    assert result.returncode == 0, result.stderr
    inspected = json.loads(result.stdout)
    assert inspected["format_version"] == 2
    assert inspected["k"] == 2

    result = run_cli("verify", str(artifacts / "graft-v2"), "--model", str(model_dir))
    assert result.returncode == 0, result.stderr + result.stdout
    assert "MATCH" in result.stdout

    result = run_cli("verify", str(artifacts / "graft-v2"), "--model", str(bad_dir))
    assert result.returncode == 1
    assert "MISMATCH" in result.stdout
    assert "first=" in result.stdout


class FixedBatchTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    padding_side = "right"

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=None,
        truncation=None,
        max_length=None,
        **kwargs,
    ):
        count = len(texts)
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8] for _ in range(count)],
            dtype=torch.long,
        )
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def test_train_graft_tiny_model_round_trip(tmp_path) -> None:
    data = [
        {"instruction": f"What is {index}+{index}?", "input": "", "output": str(index + index)}
        for index in range(4)
    ]
    train_data = tmp_path / "train.json"
    train_data.write_text(json.dumps(data), encoding="utf-8")

    torch.manual_seed(42)
    model = LlamaForCausalLM(llama_config())
    result = train_graft(
        model_id="test/tiny",
        train_data=str(train_data),
        output_dir=str(tmp_path / "graft"),
        k=2,
        geometry="aligned",
        support_seed=7,
        learning_rate=1e-2,
        steps=2,
        batch_size=2,
        seed=11,
        max_examples=4,
        max_length=16,
        loss_on="all",
        device="cpu",
        dtype="fp32",
        model=model,
        tokenizer=FixedBatchTokenizer(),
    )

    assert result["manifest"]["format_version"] == 2
    assert result["dtype"] == "float32"
    trained = result["model"]
    trained.eval()
    input_ids = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        expected = trained(input_ids).logits

    torch.manual_seed(42)
    fresh = LlamaForCausalLM(llama_config())
    attached = attach_graft(fresh, result["artifact_path"])
    attached.eval()
    with torch.no_grad():
        actual = attached(input_ids).logits

    assert torch.equal(actual, expected)


def test_grafting_facade_exports_api() -> None:
    import grafting

    for name in (
        "save_graft",
        "load_graft",
        "verify_graft",
        "read_graft_manifest",
        "attach_graft",
        "detach_graft",
        "enable_graft",
        "swap_graft",
        "list_grafts",
        "train_graft",
    ):
        assert callable(getattr(grafting, name))


def test_cross_host_refusal_names_models_and_counts(tmp_path) -> None:
    model = trained(delta_seed=11)
    save_graft(model, str(tmp_path / "graft"), extra_metadata=meta({"base_model": "test/artifact"}))

    host = base_model()
    host.config._name_or_path = "test/host"
    host.model.layers.append(TinyLayer())

    with pytest.raises(RuntimeError) as exc:
        attach_graft(host, str(tmp_path / "graft"))

    message = str(exc.value)
    assert "test/artifact" in message
    assert "test/host" in message
    assert "6 host tensors" in message
    assert "9 host tensors" in message
    assert "different base model" in message
    assert not hasattr(host, "ocfda_wrappers")


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    padding_side = "right"

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=False,
        **kwargs,
    ):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        sequences = []
        for text in texts:
            ids = [ord(character) % 30 + 2 for character in text]
            if truncation and max_length:
                ids = ids[:max_length]
            sequences.append(ids)
        if return_tensors == "pt" or padding:
            width = max((len(sequence) for sequence in sequences), default=1)
            input_ids = torch.zeros((len(sequences), width), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for index, sequence in enumerate(sequences):
                input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
                attention_mask[index, : len(sequence)] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": sequences[0] if single else sequences}


def test_train_graft_response_loss_masks_prompt(tmp_path) -> None:
    tokenizer = CharacterTokenizer()
    example = {"instruction": "Q?", "input": "", "output": "ABC"}
    prefix = format_prompt_prefix(example)
    input_ids, attention_mask, labels = encode_examples(
        tokenizer,
        [example],
        max_length=128,
        loss_on="response",
    )
    boundary = len(prefix)
    assert attention_mask.all()
    assert (labels[0, :boundary] == -100).all()
    assert (labels[0, boundary:] != -100).all()
    assert labels[0, boundary:].tolist() == input_ids[0, boundary:].tolist()

    data = [
        {"instruction": f"What is {index}+{index}?", "input": "", "output": str(index + index)}
        for index in range(4)
    ]
    train_data = tmp_path / "train.json"
    train_data.write_text(json.dumps(data), encoding="utf-8")
    torch.manual_seed(42)
    model = LlamaForCausalLM(llama_config())
    result = train_graft(
        model_id="test/tiny",
        train_data=str(train_data),
        output_dir=str(tmp_path / "graft"),
        k=2,
        geometry="aligned",
        support_seed=7,
        learning_rate=1e-2,
        steps=2,
        batch_size=2,
        seed=11,
        max_examples=4,
        max_length=64,
        loss_on="response",
        device="cpu",
        dtype="fp32",
        model=model,
        tokenizer=CharacterTokenizer(),
    )
    assert result["manifest"]["format_version"] == 2
    assert result["loss_on"] == "response"
