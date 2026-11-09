import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from fine_tuning.graft_artifact import load_graft, save_graft
from fine_tuning.ocfda import (
    OCFDA_PARAMETER_MARKER,
    OCFDA_SUPPORT_MARKER,
    get_ocfda_model,
    get_ocfda_model_state_dict,
    host_tensor_hashes,
)


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


def make_model(seed: int = 3) -> torch.nn.Module:
    torch.manual_seed(seed)
    return get_ocfda_model(TinyModel(), geometry="aligned", support_seed=7, k=2)


def set_deltas(model: torch.nn.Module, seed: int = 11) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for wrapper in model.ocfda_wrappers:
            wrapper.graft_delta.copy_(torch.randn_like(wrapper.graft_delta))


def swiglu(mlp: torch.nn.Module, probe: torch.Tensor) -> torch.Tensor:
    return mlp.down_proj(F.silu(mlp.gate_proj(probe)) * mlp.up_proj(probe))


def metadata(extra: dict | None = None) -> dict:
    payload = {"base_model": "test/tiny", "model_revision": "rev1", "training_seed": 2001}
    payload.update(extra or {})
    return payload


def test_save_load_round_trip_reproduces_grafted_logits(tmp_path) -> None:
    model = make_model()
    set_deltas(model)
    probe = torch.randn(4, 4)
    expected = swiglu(model.model.layers[0].mlp, probe).detach().clone()

    manifest = save_graft(model, str(tmp_path / "graft"), extra_metadata=metadata())
    assert manifest["format_version"] == 2
    assert manifest["k"] == 2
    assert manifest["seed"] == 7

    host = make_model()
    load_graft(host, str(tmp_path / "graft"))
    assert torch.equal(swiglu(host.model.layers[0].mlp, probe), expected)


def test_load_refuses_modified_host_tensor_and_names_it(tmp_path) -> None:
    model = make_model()
    set_deltas(model)
    save_graft(model, str(tmp_path / "graft"), extra_metadata=metadata())

    host = make_model()
    with torch.no_grad():
        host.model.layers[0].mlp.gate_proj.weight.add_(1.0)

    with pytest.raises(RuntimeError, match="gate_proj.weight"):
        load_graft(host, str(tmp_path / "graft"))


def test_load_accepts_format_version_1_checkpoint(tmp_path) -> None:
    model = make_model()
    set_deltas(model)
    probe = torch.randn(4, 4)
    expected = swiglu(model.model.layers[0].mlp, probe).detach().clone()

    state = get_ocfda_model_state_dict(model)
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()},
        str(tmp_path / "model.safetensors"),
    )
    legacy_metadata = {
        "format_version": 1,
        "adapter_name": "ocfda",
        "method": "ocfda-aligned",
        "base_model": "test/tiny",
        "model_revision": "rev1",
        "geometry": "aligned",
        "support_seed": 7,
        "training_seed": 2001,
        "ocfda_k": 2,
        "ocfda_supports": model.ocfda_supports,
        "ocfda_trainable_scalars": model.optimizer_trainable_params,
        "ocfda_ownership": {
            "host": {
                "hashes": host_tensor_hashes(
                    model,
                    exclude=(OCFDA_PARAMETER_MARKER, OCFDA_SUPPORT_MARKER),
                )
            }
        },
    }
    (tmp_path / "supertuning_config.json").write_text(json.dumps(legacy_metadata), encoding="utf-8")

    host = make_model()
    loaded = load_graft(host, str(tmp_path))
    assert loaded["format_version"] == 1
    assert torch.equal(swiglu(host.model.layers[0].mlp, probe), expected)


def test_strict_b1_rejects_non_b1_model_id(tmp_path) -> None:
    model = make_model()
    set_deltas(model)
    save_graft(
        model,
        str(tmp_path / "graft"),
        extra_metadata=metadata({"base_model": "Qwen/Qwen2.5-0.5B", "model_revision": "other"}),
    )

    host = make_model()
    with pytest.raises(RuntimeError, match="strict_b1"):
        load_graft(host, str(tmp_path / "graft"), strict_b1=True)


def test_legacy_save_writes_only_graft_tensors_and_loads(tmp_path) -> None:
    model = make_model()
    set_deltas(model)
    probe = torch.randn(4, 4)
    expected = swiglu(model.model.layers[0].mlp, probe).detach().clone()

    save_graft(model, str(tmp_path), extra_metadata=metadata(), legacy_checkpoint=True)

    assert (tmp_path / "supertuning_config.json").is_file()
    assert not (tmp_path / "graft_manifest.json").exists()
    saved = load_file(str(tmp_path / "model.safetensors"))
    assert saved
    assert all(
        "graft_delta" in name or "graft_support" in name
        for name in saved
    )

    host = make_model()
    loaded = load_graft(host, str(tmp_path))
    assert loaded["format_version"] == 1
    assert torch.equal(swiglu(host.model.layers[0].mlp, probe), expected)
