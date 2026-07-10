import importlib.machinery
import sys
import types

import torch


def test_rosa_wraps_linear_layer_with_official_peft(monkeypatch) -> None:
    fake_bnb = types.ModuleType("bitsandbytes")
    fake_bnb.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes", loader=None)
    fake_bnb.nn = types.SimpleNamespace(Linear4bit=type("Linear4bit", (), {}))
    fake_bnb.functional = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)

    from fine_tuning.rosa.rosa.layer import Linear
    from fine_tuning.rosa.rosa_adapter import get_rosa_model, get_rosa_model_state_dict

    model = torch.nn.Sequential()
    model.add_module("q_proj", torch.nn.Linear(4, 3, bias=False))
    adapted = get_rosa_model(
        model,
        target_modules=["q_proj"],
        r=1,
        d=0.25,
        alpha=2,
        dropout=0.0,
        impl="sp_add",
        schedule="wl1",
        spa_num_grads=1,
        rosa_dtype="fp32",
    )

    assert isinstance(adapted.q_proj, Linear)
    assert sum(parameter.numel() for parameter in adapted.parameters() if parameter.requires_grad) == 10

    state = get_rosa_model_state_dict(adapted)
    assert any("rosa_A" in key for key in state)
    assert any("rosa_B" in key for key in state)
    assert any("rosa_spa" in key and "row_offs" in key for key in state)
    assert all("rosa_" in key for key in state)
