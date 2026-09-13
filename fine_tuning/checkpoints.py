"""Save-format helpers and reusable adapter checkpoint loading."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from typing import Any, Optional

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from dense_plus_sparse_linear import get_dense_plus_sparse_model
from dense_plus_sparse_linear_plus_lora import get_dense_plus_sparse_plus_lora_model

try:
    from .artifacts import sha256_file, sha256_tree
    from .ocfda import get_ocfda_model, verify_host_tensor_hashes
except ImportError:
    from artifacts import sha256_file, sha256_tree
    from ocfda import get_ocfda_model, verify_host_tensor_hashes

METADATA_NAME = "supertuning_config.json"
B1_PROTOCOL = "B1-OCFDA"
B1_MODEL_ID = "meta-llama/Llama-3.2-1B"
B1_MODEL_REVISION = "5d853ed7d16ac794afa8f5c9c7f59f4e9c950954"
B1_SUPERTUNING_COMMIT = "3e961f0bb7ca49417f3804d7a61b24af58fab21d"
B1_TRAIN_BYTES = 12_098_055
B1_TRAIN_BLOB_SHA = "e72c024ec9957e8f7e67d2478450ac8851b666a7"
B1_OCFDA_K = 57
B1_OCFDA_SUPPORT_SEEDS = {9001, 1001, 1002, 1003}
B1_OCFDA_TRAINING_SEEDS = {9001, 2001, 2002, 2003}
SECOND_SETTING_PROTOCOL = "B2-OCFDA"
SECOND_SETTING_MODEL_ID = "meta-llama/Llama-3.2-3B"
SECOND_SETTING_MODEL_REVISION = "13afe5124825b4f3751f836b40dafda64c1ed062"
SECOND_SETTING_OCFDA_K = 47


def _validate_b1_provenance(
    metadata: dict[str, Any], checkpoint_dir: str, requested_base_model: Optional[str]
) -> None:
    if metadata.get("protocol") == SECOND_SETTING_PROTOCOL:
        _validate_second_setting_provenance(metadata, checkpoint_dir, requested_base_model)
        return
    if metadata.get("protocol") != B1_PROTOCOL:
        return
    expected = {
        "base_model": B1_MODEL_ID,
        "model_revision": B1_MODEL_REVISION,
        "tokenizer_revision": B1_MODEL_REVISION,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"B1 checkpoint has invalid {key}: {metadata.get(key)!r}")
    if requested_base_model and requested_base_model != B1_MODEL_ID:
        raise RuntimeError("B1 checkpoint cannot be loaded with a different base model")

    manifest_candidates = [os.path.join(checkpoint_dir, "artifact_manifest.json")]
    if metadata.get("artifact_manifest_path"):
        manifest_candidates.append(metadata["artifact_manifest_path"])
    manifest_path = next((path for path in manifest_candidates if os.path.exists(path)), None)
    if manifest_path is None:
        raise RuntimeError("B1 checkpoint is missing its artifact manifest")
    with open(manifest_path, "rb") as manifest_file:
        manifest_bytes = manifest_file.read()
    expected_manifest_sha256 = metadata.get("artifact_manifest_sha256")
    if not expected_manifest_sha256 or sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise RuntimeError("B1 artifact manifest digest does not match checkpoint metadata")
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("protocol") != B1_PROTOCOL
        or manifest.get("model") != B1_MODEL_ID
        or manifest.get("model_revision") != B1_MODEL_REVISION
        or manifest.get("tokenizer_revision") != B1_MODEL_REVISION
    ):
        raise RuntimeError("B1 artifact manifest does not match the frozen model protocol")
    expected_dataset_artifacts = {
        key: manifest[key] for key in ("train_data", "heldout_dataset", "benchmark_source")
    }
    if metadata.get("dataset_artifacts") != expected_dataset_artifacts:
        raise RuntimeError("B1 dataset artifact metadata does not match its manifest")
    code_artifacts = manifest.get("code_artifacts", {})
    model_snapshot = manifest.get("model_snapshot", {})
    if (
        code_artifacts.get("scaffold_commit") != B1_SUPERTUNING_COMMIT
        or not code_artifacts.get("files")
        or not model_snapshot.get("path")
        or not model_snapshot.get("files")
    ):
        raise RuntimeError("B1 artifact manifest is missing code or model snapshot provenance")
    code_files = list(code_artifacts["files"])
    if not all(os.path.isfile(path) for path in code_files) or sha256_tree(code_files) != code_artifacts["files"]:
        raise RuntimeError("B1 source code does not match its artifact manifest")
    if sha256_tree([model_snapshot["path"]]) != model_snapshot["files"]:
        raise RuntimeError("B1 model snapshot does not match its artifact manifest")

    tokenizer_files = metadata.get("tokenizer_files")
    if not isinstance(tokenizer_files, dict) or not tokenizer_files:
        raise RuntimeError("B1 checkpoint is missing tokenizer file hashes")
    for relative_path, expected_digest in tokenizer_files.items():
        path = os.path.join(checkpoint_dir, *str(relative_path).split("/"))
        if not os.path.isfile(path) or sha256_file(path) != expected_digest:
            raise RuntimeError(f"B1 tokenizer artifact does not match checkpoint metadata: {relative_path}")

    train_data = manifest["train_data"]
    if train_data.get("bytes") != B1_TRAIN_BYTES or train_data.get("git_blob_sha1") != B1_TRAIN_BLOB_SHA:
        raise RuntimeError("B1 training data provenance does not match the frozen Super-Tuning artifact")
    if os.path.getsize(train_data["path"]) != train_data["bytes"] or sha256_file(train_data["path"]) != train_data["sha256"]:
        raise RuntimeError("B1 training data does not match its artifact manifest")
    for key in ("heldout_dataset", "benchmark_source"):
        artifact = manifest[key]
        if sha256_tree([artifact["path"]]) != artifact["files"]:
            raise RuntimeError(f"B1 {key} does not match its artifact manifest")


def _validate_second_setting_provenance(
    metadata: dict[str, Any], checkpoint_dir: str, requested_base_model: Optional[str]
) -> None:
    expected = {
        "base_model": SECOND_SETTING_MODEL_ID,
        "model_revision": SECOND_SETTING_MODEL_REVISION,
        "tokenizer_revision": SECOND_SETTING_MODEL_REVISION,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Second-setting checkpoint has invalid {key}: {metadata.get(key)!r}")
    if requested_base_model and requested_base_model != SECOND_SETTING_MODEL_ID:
        raise RuntimeError("Second-setting checkpoint cannot be loaded with a different base model")
    if int(metadata.get("ocfda_k", -1)) != SECOND_SETTING_OCFDA_K:
        raise RuntimeError(f"Second-setting checkpoint does not carry the frozen k={SECOND_SETTING_OCFDA_K}")
    manifest_path = os.path.join(checkpoint_dir, "artifact_manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError("Second-setting checkpoint is missing its artifact manifest copy")
    with open(manifest_path, "rb") as manifest_file:
        manifest = json.loads(manifest_file.read())
    if (
        manifest.get("protocol") != SECOND_SETTING_PROTOCOL
        or manifest.get("model") != SECOND_SETTING_MODEL_ID
        or manifest.get("model_revision") != SECOND_SETTING_MODEL_REVISION
    ):
        raise RuntimeError("Second-setting artifact manifest does not match the frozen protocol or model")
    tokenizer_files = metadata.get("tokenizer_files")
    if not isinstance(tokenizer_files, dict) or not tokenizer_files:
        raise RuntimeError("Second-setting checkpoint is missing tokenizer file hashes")
    for relative_path, expected_digest in tokenizer_files.items():
        path = os.path.join(checkpoint_dir, *str(relative_path).split("/"))
        if not os.path.isfile(path) or sha256_file(path) != expected_digest:
            raise RuntimeError(f"Second-setting tokenizer artifact does not match checkpoint metadata: {relative_path}")


def _validate_b1_host_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    ownership = metadata.get("ocfda_ownership", {})
    expected_host_hashes = ownership.get("host", {}).get("hashes")
    if not isinstance(expected_host_hashes, dict) or not expected_host_hashes:
        raise RuntimeError("B1 checkpoint is missing pretrained host tensor hashes")
    if ownership.get("host", {}).get("after_hashes") != expected_host_hashes:
        raise RuntimeError("B1 checkpoint has inconsistent pretrained host hash reports")
    if ownership.get("detach", {}).get("detached") is not True:
        raise RuntimeError("B1 checkpoint is missing the OCFDA detach integrity result")
    if ownership.get("optimizer", {}).get("only_ocfda") is not True:
        raise RuntimeError("B1 checkpoint is missing the OCFDA optimizer integrity result")
    return expected_host_hashes


def load_metadata(checkpoint_dir: str) -> dict[str, Any]:
    path = os.path.join(checkpoint_dir, METADATA_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {METADATA_NAME} in {checkpoint_dir}. "
            "This loader supports checkpoints produced by the public Super-Tuning runner."
        )
    with open(path, "r", encoding="utf-8") as metadata_file:
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
    unexpected = sorted(set(state) - expected)
    if missing:
        raise RuntimeError(f"Checkpoint is missing {len(missing)} adapter tensors; first: {missing[0]}")
    if unexpected:
        raise RuntimeError(f"Checkpoint contains {len(unexpected)} unexpected tensors; first: {unexpected[0]}")
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
    adapter_name = metadata.get("adapter_name")
    method = metadata.get("method", adapter_name)
    if adapter_name == "ocfda" and metadata.get("protocol") not in {B1_PROTOCOL, SECOND_SETTING_PROTOCOL}:
        raise RuntimeError("OCFDA checkpoints require a frozen protocol metadata")
    _validate_b1_provenance(metadata, checkpoint_dir, base_model)
    base_model = base_model or metadata.get("base_model")
    if not base_model:
        raise ValueError("base_model is required when it is absent from checkpoint metadata")

    torch_dtype = _torch_dtype(dtype, metadata)
    if not torch.cuda.is_available() and torch_dtype != torch.float32:
        torch_dtype = torch.float32
        device_map = None

    tokenizer_source = checkpoint_dir if os.path.exists(os.path.join(checkpoint_dir, "tokenizer_config.json")) else base_model
    tokenizer_revision = metadata.get("tokenizer_revision")
    tokenizer_kwargs = {"revision": tokenizer_revision} if tokenizer_revision and tokenizer_source == base_model else {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, **tokenizer_kwargs)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if method == "full" or adapter_name in {"no", "sift"}:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if metadata.get("model_revision"):
            model_kwargs["revision"] = metadata["model_revision"]
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs,
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
        elif adapter_name == "ocfda":
            if (
                metadata.get("method") not in {"ocfda-aligned", "ocfda-independent"}
                or metadata.get("geometry")
                != ("aligned" if metadata.get("method") == "ocfda-aligned" else "independent")
                or metadata.get("target_modules") != ["gate_proj", "up_proj", "down_proj"]
                or (metadata.get("protocol") == B1_PROTOCOL and metadata.get("ocfda_k") != B1_OCFDA_K)
                or int(metadata.get("support_seed", -1)) not in B1_OCFDA_SUPPORT_SEEDS
                or int(metadata.get("training_seed", -1)) not in B1_OCFDA_TRAINING_SEEDS
                or metadata.get("optimizer_name") != "adamw"
                or metadata.get("weight_decay") != 0.0
                or metadata.get("bf16") is not True
            ):
                raise RuntimeError("B1 checkpoint OCFDA metadata does not match the frozen protocol")
            model = get_ocfda_model(
                model,
                geometry=metadata["geometry"],
                support_seed=int(metadata["support_seed"]),
                k=int(metadata["ocfda_k"]),
                supports=metadata.get("ocfda_supports"),
            )
            if metadata.get("ocfda_trainable_scalars") != model.optimizer_trainable_params:
                raise RuntimeError("Checkpoint OCFDA budget does not match the reconstructed adapter")
            state = _load_adapter_state(checkpoint_dir)
            ownership = metadata.get("ocfda_ownership", {})
            expected_host_hashes = (
                _validate_b1_host_metadata(metadata)
                if metadata.get("protocol") == B1_PROTOCOL
                else ownership.get("host", {}).get("hashes")
            )
            if not isinstance(expected_host_hashes, dict) or not expected_host_hashes:
                raise RuntimeError("OCFDA checkpoint is missing pretrained host tensor hashes")
            host_state_names = {
                name
                for name, _ in list(model.named_parameters()) + list(model.named_buffers())
                if "graft_delta" not in name and "graft_support" not in name
            }
            if set(expected_host_hashes) != host_state_names:
                raise RuntimeError("OCFDA checkpoint does not contain a complete pretrained host hash set")
            if expected_host_hashes:
                verify_host_tensor_hashes(model, expected_host_hashes)
            expected_supports = {
                name: tensor
                for name, tensor in model.state_dict().items()
                if "graft_support" in name
            }
            for name, tensor in expected_supports.items():
                if name not in state or not torch.equal(tensor.cpu(), state[name].cpu()):
                    raise RuntimeError(f"Checkpoint support does not match metadata for {name}")
            _validate_loaded_adapter(model, state, ("graft_delta", "graft_support"))
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
