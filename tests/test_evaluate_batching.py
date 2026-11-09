import json
from types import SimpleNamespace

import pytest
import torch

from fine_tuning.evaluate import eval_model


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.padding_side = "left"

    def __call__(self, text, return_tensors=None, padding=False):
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [[ord(character) for character in item] for item in texts]
        max_length = max(len(item) for item in encoded)
        input_ids = []
        attention_mask = []
        for item in encoded:
            pad = max_length - len(item)
            input_ids.append([0] * pad + item)
            attention_mask.append([0] * pad + [1] * len(item))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def decode(self, sequence, skip_special_tokens=True):
        tokens = [int(token) for token in sequence]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in (0, 2)]
        return "".join(chr(token) for token in tokens)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.batch_sizes = []

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.batch_sizes.append(int(input_ids.shape[0]))
        continuation = [ord(character) for character in " The answer is 42."]
        rows = []
        for row in input_ids:
            prompt = [int(token) for token in row if int(token) != 0]
            rows.append(prompt + continuation)
        max_length = max(len(row) for row in rows)
        padded = [row + [0] * (max_length - len(row)) for row in rows]
        return SimpleNamespace(sequences=torch.tensor(padded, dtype=torch.long))


def _write_dataset(root, records):
    directory = root / "AddSub"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "test.json").write_text(json.dumps(records), encoding="utf-8")


def _records():
    return [
        {"instruction": "What is 2 + 40?", "input": "", "output": "42", "answer": 42},
        {"instruction": "What is 1 + 0?", "input": "", "output": "1", "answer": 1},
        {"instruction": "What is 40 + 2?", "input": "", "output": "42", "answer": 42},
        {"instruction": "What is 2 + 0?", "input": "", "output": "2", "answer": 2},
    ]


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_batched_generation_matches_serial_rows_and_resumes(tmp_path) -> None:
    _write_dataset(tmp_path, _records())
    serial_model = FakeModel()
    batched_model = FakeModel()
    serial_path = tmp_path / "serial" / "AddSub.jsonl"
    batched_path = tmp_path / "batched" / "AddSub.jsonl"

    serial_score = eval_model(
        "AddSub",
        serial_model,
        FakeTokenizer(),
        dataset_dir=str(tmp_path),
        progress_path=str(serial_path),
        generation_batch_size=1,
    )
    batched_score = eval_model(
        "AddSub",
        batched_model,
        FakeTokenizer(),
        dataset_dir=str(tmp_path),
        progress_path=str(batched_path),
        generation_batch_size=4,
    )

    assert serial_score == pytest.approx(0.5)
    assert batched_score == pytest.approx(0.5)
    assert serial_model.batch_sizes == [1, 1, 1, 1]
    assert batched_model.batch_sizes == [4]

    serial_rows = _read_rows(serial_path)
    batched_rows = _read_rows(batched_path)
    assert [row["idx"] for row in serial_rows] == [0, 1, 2, 3]
    assert [row["idx"] for row in batched_rows] == [0, 1, 2, 3]
    for serial_row, batched_row in zip(serial_rows, batched_rows):
        assert serial_row["output_pred"] == batched_row["output_pred"] == "The answer is 42."
        assert serial_row["pred"] == batched_row["pred"] == 42.0
        assert serial_row["flag"] == batched_row["flag"]

    resumed_model = FakeModel()
    resumed_score = eval_model(
        "AddSub",
        resumed_model,
        FakeTokenizer(),
        dataset_dir=str(tmp_path),
        progress_path=str(batched_path),
        generation_batch_size=4,
    )
    assert resumed_score == pytest.approx(0.5)
    assert resumed_model.batch_sizes == []
