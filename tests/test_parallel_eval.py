import json

import pytest

from fine_tuning.evaluate_pilot_winners import EXPECTED_TRAINABLE_SCALARS, find_winner_checkpoint
from fine_tuning.parallel_eval import _merge_shards, evaluate_checkpoint_parallel


def test_find_winner_checkpoint_run_id(tmp_path) -> None:
    target = (
        tmp_path
        / "pilot"
        / "checkpoints"
        / "Llama-3_2-1B_r8_ocfda-aligned_lr0p0005_seed9001_support9001"
    )
    target.mkdir(parents=True)
    path = find_winner_checkpoint(str(tmp_path), "ocfda-aligned", 5e-4, 9001, 9001)
    assert path == str(target)
    assert EXPECTED_TRAINABLE_SCALARS == 5_603_328


def test_merge_shards_merges_rows_by_dataset_and_index(tmp_path) -> None:
    shard_a = tmp_path / "a.jsonl"
    shard_a.write_text(
        json.dumps({"dataset": "AddSub", "idx": 0, "flag": True})
        + "\n"
        + json.dumps({"dataset": "AQuA", "idx": 0, "flag": False})
        + "\n",
        encoding="utf-8",
    )
    shard_b = tmp_path / "b.jsonl"
    shard_b.write_text(json.dumps({"dataset": "AddSub", "idx": 1, "flag": False}) + "\n", encoding="utf-8")

    merged = _merge_shards([str(shard_a), str(shard_b)], ["AddSub", "AQuA"])

    assert set(merged["AddSub"]) == {0, 1}
    assert merged["AddSub"][0]["flag"] is True
    assert merged["AQuA"][0]["flag"] is False


def test_parallel_eval_rejects_existing_progress(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "AddSub").mkdir(parents=True)
    (dataset_dir / "AddSub" / "test.json").write_text("[]", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "AddSub.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already exists"):
        evaluate_checkpoint_parallel(
            checkpoint_dir="unused",
            datasets=["AddSub"],
            dataset_dir=str(dataset_dir),
            out_dir=str(out_dir),
        )
