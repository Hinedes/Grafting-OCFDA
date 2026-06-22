import json
import math
import os
from typing import Any, Dict, Optional

import transformers


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


class TrainingCurveCallback(transformers.TrainerCallback):
    """Persist Trainer loss logs as JSONL so optimization curves survive restarts."""

    def __init__(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        self.path = path
        self.metadata = metadata or {}

    def _row(self, event: str, state, logs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        row = {
            **self.metadata,
            "event": event,
            "step": int(state.global_step),
            "epoch": _json_safe(state.epoch),
        }
        if logs:
            for key, value in logs.items():
                row[key] = _json_safe(value)
        return row

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        append_jsonl(self.path, self._row("train_begin", state))

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ARG002
        if logs:
            append_jsonl(self.path, self._row("log", state, logs))

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ARG002
        append_jsonl(self.path, self._row("train_end", state))
