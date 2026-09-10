import json

import pytest
import torch

from fine_tuning.checkpoints import _validate_loaded_adapter, load_metadata


def test_load_metadata(tmp_path) -> None:
    metadata = {"format_version": 1, "adapter_name": "super", "base_model": "example/model"}
    (tmp_path / "supertuning_config.json").write_text(json.dumps(metadata))
    assert load_metadata(str(tmp_path)) == metadata


def test_load_metadata_rejects_unknown_version(tmp_path) -> None:
    (tmp_path / "supertuning_config.json").write_text(json.dumps({"format_version": 999}))
    with pytest.raises(ValueError, match="format version"):
        load_metadata(str(tmp_path))


def test_adapter_state_rejects_host_tensors() -> None:
    class FakeModel:
        def state_dict(self):
            return {"layer.graft_delta": torch.zeros(1)}

        def load_state_dict(self, state, strict):
            assert strict is False

    with pytest.raises(RuntimeError, match="unexpected tensors"):
        _validate_loaded_adapter(
            FakeModel(),
            {"layer.graft_delta": torch.zeros(1), "layer.weight": torch.zeros(1)},
            ("graft_delta",),
        )
