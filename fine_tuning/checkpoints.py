"""Save-format helpers and reusable adapter checkpoint loading."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from dense_plus_sparse_linear import get_dense_plus_sparse_model
from dense_plus_sparse_linear_plus_lora import get_dense_plus_sparse_plus_lora_model

METADATA_NAME = "supertuning_config.json"


def load_metadata(checkpoint_dir: str) -> dict[str, Any]:
    path = os.path.join(checkpoint_dir, METADATA_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {METADATA_NAME} in {checkpoint_dir}. "
            "This loader supports checkpoints produced by the public Super-Tuning runner."
        )
    with open(path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format version: {metadata.get('format_version')!r}")
    return metadata


def _torch_dtype(name: str, metadata: dict[str, Any]) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if metadata.get("bf16") else torch.float16
    choices = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in choices:
        raise ValueError(f"Unknown dtype {name!r}; choose auto, bf16, fp16, or fp32.")
    return choices[name]


def _load_adapter_state(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        return load_file(safetensors_path)
    torch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(torch_path):
        return torch.load(torch_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}")


def _validate_loaded_adapter(model, state: dict[str, torch.Tensor], markers: tuple[str, ...]) -> None:
    expected = {
        name
        for name in model.state_dict()
        if any(marker in name for marker in markers)
    }
    missing = sorted(expected - set(state))
    if missing:
        raise RuntimeError(f"Checkpoint is missing {len(missing)} adapter tensors; first: {missing[0]}")
    model.load_state_dict(state, strict=False)


def load_checkpoint(
    checkpoint_dir: str,
    base_model: Optional[str] = None,
    dtype: str = "auto",
    device_map: Optional[str] = "auto",
    merge_rosa: bool = True,
):
    """Load a checkpoint produced with ``--save_adapters``.

    Returns ``(model, tokenizer, metadata)``. Sparse-only and Supra checkpoints
    reconstruct their lightweight adapter layers before loading the saved value
    vectors and index buffers.
    """

    metadata = load_metadata(checkpoint_dir)
    base_model = base_model or metadata.get("base_model")
    if not base_model:
        raise ValueError("base_model is required when it is absent from checkpoint metadata")

    torch_dtype = _torch_dtype(dtype, metadata)
    if not torch.cuda.is_available() and torch_dtype != torch.float32:
        torch_dtype = torch.float32
        device_map = None

    tokenizer_source = checkpoint_dir if os.path.exists(os.path.join(checkpoint_dir, "tokenizer_config.json")) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    adapter_name = metadata.get("adapter_name")
    method = metadata.get("method", adapter_name)
    if method == "full" or adapter_name in {"no", "sift"}:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        if adapter_name == "lora":
            model = PeftModel.from_pretrained(model, checkpoint_dir)
        elif adapter_name == "super":
            model = get_dense_plus_sparse_model(
                model,
                target_modules_list=metadata["target_modules"],
                sparse_rate=float(metadata["sparse_rate"]),
                indices_choice="random",
            )
            state = _load_adapter_state(checkpoint_dir)
            _validate_loaded_adapter(model, state, ("values", "indices"))
        elif adapter_name == "supra":
            model = get_dense_plus_sparse_plus_lora_model(
                model,
                lora_params_ratio=float(metadata["lora_params_ratio"]),
                sparse_rate=float(metadata["sparse_rate"]),
                lora_alpha=int(metadata["lora_alpha"]),
                lora_dropout=float(metadata["lora_dropout"]),
                target_modules_list=metadata["target_modules"],
                indices_choice="random",
            )
            state = _load_adapter_state(checkpoint_dir)
            _validate_loaded_adapter(model, state, ("values", "indices", "lora_A", "lora_B"))
        elif adapter_name == "rosa":
            from .rosa.rosa.scheduler import RosaScheduler  # noqa: F401
            from .rosa.rosa_adapter import get_rosa_model

            model = get_rosa_model(
                model,
                target_modules=metadata["target_modules"],
                r=int(metadata["lora_r"]),
                d=float(metadata["sparse_rate"]),
                alpha=int(metadata["lora_alpha"]),
                dropout=float(metadata["lora_dropout"]),
                impl="sp_add",
                schedule=metadata["rosa_schedule"],
                spa_num_grads=int(metadata["rosa_spa_num_grads"]),
                rosa_dtype=metadata["rosa_dtype"],
            )
            state = _load_adapter_state(checkpoint_dir)
            _validate_loaded_adapter(model, state, ("rosa_", "spa_mask", "lora_A", "lora_B"))
            if merge_rosa:
                model = model.merge_and_unload(progressbar=False)
        else:
            raise ValueError(f"Unsupported adapter in checkpoint metadata: {adapter_name!r}")

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, tokenizer, metadata
