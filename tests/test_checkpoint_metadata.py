import json

import pytest

from fine_tuning.checkpoints import load_metadata


def test_load_metadata(tmp_path) -> None:
    metadata = {"format_version": 1, "adapter_name": "super", "base_model": "example/model"}
    (tmp_path / "supertuning_config.json").write_text(json.dumps(metadata))
    assert load_metadata(str(tmp_path)) == metadata


def test_load_metadata_rejects_unknown_version(tmp_path) -> None:
    (tmp_path / "supertuning_config.json").write_text(json.dumps({"format_version": 999}))
    with pytest.raises(ValueError, match="format version"):
        load_metadata(str(tmp_path))
