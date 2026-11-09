"""Non-B1 Graft training entry point.

This module trains an OCFDA Graft on any dense SwiGLU causal LM that exposes
``model.layers[*].mlp.{gate_proj,up_proj,down_proj}`` (Llama, Qwen2/2.5, ...).
It deliberately does not use the B1 runner, B1 artifact manifests, or the B1
budget planner.  The trained state is written as a standalone v2 artifact via
``save_graft``.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Iterable, Mapping, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .graft_artifact import save_graft
    from .ocfda import get_ocfda_model
except ImportError:
    from graft_artifact import save_graft
    from ocfda import get_ocfda_model


def format_prompt_prefix(example: Mapping[str, Any]) -> str:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "") or "").strip()
    parts = []
    if instruction:
        parts.append(f"### Instruction:\n{instruction}")
    if input_text:
        parts.append(f"### Input:\n{input_text}")
    parts.append("### Response:\n")
    return "\n\n".join(parts)


def format_prompt(example: Mapping[str, Any]) -> str:
    return format_prompt_prefix(example) + str(example.get("output", "")).strip()


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return resolved


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    choices = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype not in choices:
        raise ValueError(f"Unknown dtype {dtype!r}; choose auto, bf16, fp16, or fp32")
    return choices[dtype]


def _load_examples(path: str, max_examples: Optional[int]) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Training data must be a JSON list, got {type(data).__name__}")
    examples = [row for row in data if isinstance(row, dict)]
    if not examples:
        raise ValueError(f"Training data is empty: {path}")
    if max_examples is not None and max_examples > 0:
        examples = examples[:max_examples]
    return examples


def _token_ids(encoded: Mapping[str, Any], index: int) -> list[int]:
    ids = encoded["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[index]
    return [int(value) for value in ids]


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    return min(len(left), len(right))


def encode_examples(
    tokenizer: Any,
    examples: list[dict[str, Any]],
    max_length: int = 128,
    loss_on: str = "response",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a batch and mask labels according to ``loss_on``.

    ``response`` uses the literal boundary ``### Response:\\n`` from
    ``format_prompt_prefix``: the prefix is instruction/input plus that
    delimiter, and the response is the ``output`` field.  Token-level masking
    starts at the end of the common token prefix between the prefix text and
    the full prompt+response text.
    """

    if loss_on not in {"response", "all"}:
        raise ValueError("loss_on must be 'response' or 'all'")
    full_texts = [format_prompt(example) for example in examples]
    full = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if loss_on == "response":
        prefix_texts = [format_prompt_prefix(example) for example in examples]
        prefix = tokenizer(
            prefix_texts,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        for index in range(len(examples)):
            prefix_ids = _token_ids(prefix, index)
            full_ids = [int(value) for value in input_ids[index].tolist()]
            start = _common_prefix_length(prefix_ids, full_ids)
            labels[index, :start] = -100
        if not labels.ne(-100).any():
            raise ValueError("loss_on='response' batch has no response tokens")
    return input_ids, attention_mask, labels


def train_graft(
    model_id: str,
    train_data: str,
    output_dir: str,
    revision: Optional[str] = None,
    k: int = 57,
    geometry: str = "aligned",
    support_seed: Optional[int] = None,
    supports: Optional[Mapping[str, Mapping[str, Iterable[int]]]] = None,
    learning_rate: float = 5e-4,
    steps: int = 200,
    batch_size: int = 2,
    seed: int = 0,
    max_examples: Optional[int] = 256,
    max_length: int = 128,
    loss_on: str = "response",
    dtype: str = "auto",
    device: Optional[str] = None,
    local_files_only: bool = False,
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[Any] = None,
) -> dict[str, Any]:
    """Train a standalone Graft and write a format-v2 artifact.

    ``supports`` may be explicit; otherwise ``support_seed`` is required and the
    supports are generated deterministically.  The returned dict includes the
    trained ``model`` and ``tokenizer`` for a caller that needs reference
    logits; the CLI strips them from its JSON summary.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if supports is None and support_seed is None:
        raise ValueError("provide support_seed or explicit supports")
    if loss_on not in {"response", "all"}:
        raise ValueError("loss_on must be 'response' or 'all'")

    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    if resolved_device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9)
    torch.manual_seed(seed)
    random.seed(seed)

    examples = _load_examples(train_data, max_examples)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=resolved_dtype,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    model.to(resolved_device)
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = get_ocfda_model(
        model,
        geometry=geometry,
        support_seed=support_seed,
        k=k,
        supports=supports,
        verify_supports_against_seed=supports is None,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("The wrapped model has no trainable Graft parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    model.train()

    rng = random.Random(seed)
    losses: list[float] = []
    started = time.time()
    for _step in range(steps):
        batch = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
        input_ids, attention_mask, labels = encode_examples(tokenizer, batch, max_length, loss_on)
        input_ids = input_ids.to(resolved_device)
        attention_mask = attention_mask.to(resolved_device)
        labels = labels.to(resolved_device)

        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {_step + 1}: {loss.item()}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    duration = time.time() - started
    model.eval()
    host_dtype = next(model.parameters()).dtype
    extra_metadata = {
        "base_model": model_id,
        "model_revision": revision,
        "training_seed": int(seed),
        "training": {
            "steps": int(steps),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "max_examples": max_examples,
            "max_length": int(max_length),
            "loss_on": loss_on,
            "loss_start": losses[0],
            "loss_end": losses[-1],
            "duration_seconds": duration,
            "device": str(resolved_device),
            "dtype": str(host_dtype).replace("torch.", ""),
        },
    }
    manifest = save_graft(model, output_dir, extra_metadata=extra_metadata)
    return {
        "artifact_path": output_dir,
        "manifest": manifest,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "loss_on": loss_on,
        "device": str(resolved_device),
        "dtype": str(host_dtype).replace("torch.", ""),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "duration_seconds": duration,
        "model": model,
        "tokenizer": tokenizer,
    }
