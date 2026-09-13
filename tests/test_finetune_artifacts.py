"""Regression tests for tokenizer artifact-path resolution in finetune.save."""

import os

from fine_tuning.finetune import resolve_artifact_path


def test_basename_only_path_resolves_against_output_dir(tmp_path, monkeypatch) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    output_dir = tmp_path / "checkpoints" / "run_1"
    output_dir.mkdir(parents=True)
    (output_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(cwd)

    resolved = resolve_artifact_path("tokenizer_config.json", str(output_dir))

    assert resolved == os.path.join(str(output_dir), "tokenizer_config.json")
    assert os.path.isfile(resolved)


def test_output_dir_relative_path_is_not_joined_twice(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "runs" / "checkpoints" / "run_1"
    output_dir.mkdir(parents=True)
    (output_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    returned = os.path.join("runs", "checkpoints", "run_1", "tokenizer.json")
    resolved = resolve_artifact_path(returned, os.path.join("runs", "checkpoints", "run_1"))

    assert resolved == returned
    assert resolved.count("run_1") == 1
    assert os.path.isfile(resolved)


def test_absolute_path_is_used_directly(tmp_path) -> None:
    output_dir = tmp_path / "checkpoints" / "run_1"
    output_dir.mkdir(parents=True)
    absolute = output_dir / "special_tokens_map.json"
    absolute.write_text("{}", encoding="utf-8")

    assert resolve_artifact_path(str(absolute), str(output_dir)) == str(absolute)


def test_missing_absolute_path_falls_back_without_doubling(tmp_path) -> None:
    output_dir = tmp_path / "checkpoints" / "run_1"
    output_dir.mkdir(parents=True)
    missing = str(output_dir / "tokenizer.json")

    resolved = resolve_artifact_path(missing, str(output_dir))

    assert resolved == missing
