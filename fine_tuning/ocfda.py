"""OCFDA sparse FFN adapters used by the B1 experiment.

OCFDA owns one FP32 delta vector per selected FFN row/column.  The pretrained
weights stay as ordinary frozen parameters; the adapter only computes the
selected contribution, so the trainable budget is exactly the number of
stored delta scalars.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager, nullcontext
from typing import Iterable, Iterator, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

OCFDA_GEOMETRIES = ("aligned", "independent")
OCFDA_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
OCFDA_DEFAULT_K = 57
OCFDA_PARAMETER_MARKER = "graft_delta"
OCFDA_SUPPORT_MARKER = "graft_support"


def _support_seed(seed: int, layer: int, projection: int) -> int:
    """Derive stable, independent CPU RNG seeds without using Python hashing."""

    return (int(seed) + 1_000_003 * (layer + 1) + 97_003 * projection) % (2**63 - 1)


def _draw_support(intermediate_size: int, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(intermediate_size, generator=generator, dtype=torch.int64)[:k].sort().values


def generate_supports(
    intermediate_size: int,
    num_layers: int,
    k: int = OCFDA_DEFAULT_K,
    geometry: str = "aligned",
    support_seed: int = 0,
) -> dict[str, dict[str, list[int]]]:
    """Generate reproducible intermediate-coordinate supports.

    The first draw in each layer is always the Gate support.  Consequently an
    aligned and an independent run with the same support seed share Gate while
    differing only in the cross-projection reuse of that support.
    """

    if geometry not in OCFDA_GEOMETRIES:
        raise ValueError(f"geometry must be one of {OCFDA_GEOMETRIES}, got {geometry!r}")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not 0 < k <= intermediate_size:
        raise ValueError(f"k must satisfy 0 < k <= intermediate_size, got k={k}, intermediate_size={intermediate_size}")

    supports: dict[str, dict[str, list[int]]] = {}
    for layer in range(num_layers):
        gate = _draw_support(intermediate_size, k, _support_seed(support_seed, layer, 0))
        if geometry == "aligned":
            layer_supports = {projection: gate.tolist() for projection in OCFDA_PROJECTIONS}
        else:
            layer_supports = {
                "gate_proj": gate.tolist(),
                "up_proj": _draw_support(intermediate_size, k, _support_seed(support_seed, layer, 1)).tolist(),
                "down_proj": _draw_support(intermediate_size, k, _support_seed(support_seed, layer, 2)).tolist(),
            }
        supports[str(layer)] = layer_supports
    return supports


def _as_support_tensor(support: Iterable[int], device: torch.device) -> torch.Tensor:
    if isinstance(support, torch.Tensor):
        tensor = support.to(dtype=torch.long, device=device)
    else:
        tensor = torch.as_tensor(list(support), dtype=torch.long, device=device)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError("OCFDA supports must be non-empty one-dimensional sequences")
    if tensor.min().item() < 0:
        raise ValueError("OCFDA supports must not contain negative coordinates")
    if tensor.unique().numel() != tensor.numel():
        raise ValueError("OCFDA supports must not contain duplicates")
    return tensor.sort().values


class OCFDALinear(nn.Module):
    """A frozen linear layer plus a sparse row/column delta."""

    def __init__(self, base_layer: nn.Linear, projection: str, support: Iterable[int]) -> None:
        super().__init__()
        if projection not in OCFDA_PROJECTIONS:
            raise ValueError(f"Unsupported OCFDA projection: {projection}")
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"OCFDA expects nn.Linear, got {type(base_layer).__name__}")

        self.projection = projection
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        support_tensor = _as_support_tensor(support, self.weight.device)
        if int(support_tensor.max()) >= max(self.in_features, self.out_features):
            raise ValueError(
                f"Support index {int(support_tensor.max())} is out of bounds for {projection} "
                f"shape={tuple(self.weight.shape)}"
            )
        if projection in {"gate_proj", "up_proj"} and int(support_tensor.max()) >= self.out_features:
            raise ValueError(f"{projection} support is out of bounds for output size {self.out_features}")
        if projection == "down_proj" and int(support_tensor.max()) >= self.in_features:
            raise ValueError(f"down_proj support is out of bounds for input size {self.in_features}")

        self.register_buffer("graft_support", support_tensor)
        delta_shape = (
            (support_tensor.numel(), self.in_features)
            if projection in {"gate_proj", "up_proj"}
            else (self.out_features, support_tensor.numel())
        )
        self.graft_delta = nn.Parameter(
            torch.zeros(delta_shape, dtype=torch.float32, device=self.weight.device)
        )
        self.graft_delta._ocfda_owned = True
        self.graft_enabled = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = F.linear(input, self.weight, self.bias)
        if not self.graft_enabled:
            return result

        # ROCm tensors use PyTorch's ``cuda`` device type for the HIP-compatible API.
        autocast_off = torch.autocast(device_type="cuda", enabled=False) if input.device.type == "cuda" else nullcontext()
        with autocast_off:
            delta_input = input.to(dtype=self.graft_delta.dtype)
            if self.projection in {"gate_proj", "up_proj"}:
                selected_delta = torch.matmul(delta_input, self.graft_delta.transpose(0, 1))
                selected_delta = selected_delta.to(dtype=result.dtype)
                shape = (*selected_delta.shape[:-1], self.out_features)
                scattered = torch.zeros(shape, dtype=result.dtype, device=result.device)
                indices = self.graft_support.to(device=result.device)
                scattered.scatter_(-1, indices.expand(selected_delta.shape[:-1] + (indices.numel(),)), selected_delta)
                return result + scattered

            selected_input = delta_input.index_select(-1, self.graft_support.to(device=delta_input.device))
            selected_delta = F.linear(selected_input, self.graft_delta).to(dtype=result.dtype)
            return result + selected_delta


def _llama_layers(model) -> nn.ModuleList:
    model_body = getattr(model, "model", None)
    layers = getattr(model_body, "layers", None)
    if layers is None:
        layers = getattr(getattr(model_body, "decoder", None), "layers", None)
    if layers is None:
        raise ValueError("Could not find transformer layers at model.layers or model.decoder.layers")
    return layers


def _normalise_supports(
    supports: Mapping[str, Mapping[str, Iterable[int]]],
    num_layers: int,
    k: int,
) -> dict[str, dict[str, list[int]]]:
    normalised = {}
    for layer in range(num_layers):
        layer_key = str(layer)
        if layer_key not in supports:
            raise ValueError(f"Missing OCFDA support for layer {layer}")
        layer_supports = supports[layer_key]
        if set(layer_supports) != set(OCFDA_PROJECTIONS):
            raise ValueError(f"Layer {layer} must define supports for {OCFDA_PROJECTIONS}")
        layer_values = {
            projection: sorted(list(layer_supports[projection])) for projection in OCFDA_PROJECTIONS
        }
        if any(len(layer_values[projection]) != k for projection in OCFDA_PROJECTIONS):
            raise ValueError(f"Every OCFDA support must contain exactly k={k} coordinates")
        normalised[layer_key] = layer_values
    if len(supports) != num_layers:
        raise ValueError(f"Expected supports for {num_layers} layers, got {len(supports)}")
    return normalised


def get_ocfda_model(
    model: nn.Module,
    geometry: str = "aligned",
    support_seed: int = 0,
    k: int = OCFDA_DEFAULT_K,
    supports: Optional[Mapping[str, Mapping[str, Iterable[int]]]] = None,
) -> nn.Module:
    """Attach OCFDA to every Llama MLP Gate/Up/Down projection."""

    if geometry not in OCFDA_GEOMETRIES:
        raise ValueError(f"geometry must be one of {OCFDA_GEOMETRIES}, got {geometry!r}")
    if k <= 0:
        raise ValueError("k must be positive")
    layers = _llama_layers(model)
    generated_supports = generate_supports(
        intermediate_size=int(model.config.intermediate_size),
        num_layers=len(layers),
        k=k,
        geometry=geometry,
        support_seed=support_seed,
    )
    if supports is None:
        supports = generated_supports
    else:
        supports = _normalise_supports(supports, len(layers), k)
        if supports != generated_supports:
            raise ValueError("Provided OCFDA supports do not match the deterministic support seed")
    if geometry == "aligned":
        for layer, layer_supports in supports.items():
            if len({tuple(layer_supports[projection]) for projection in OCFDA_PROJECTIONS}) != 1:
                raise ValueError(f"Aligned OCFDA supports differ across projections in layer {layer}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    wrappers = []
    for layer_index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError(f"Layer {layer_index} has no MLP module")
        for projection in OCFDA_PROJECTIONS:
            base_layer = getattr(mlp, projection, None)
            if base_layer is None:
                raise ValueError(f"Layer {layer_index} has no {projection} projection")
            wrapper = OCFDALinear(base_layer, projection, supports[str(layer_index)][projection])
            setattr(mlp, projection, wrapper)
            wrappers.append(wrapper)

    model.ocfda_geometry = geometry
    model.ocfda_support_seed = int(support_seed)
    model.ocfda_k = int(k)
    model.ocfda_supports = supports
    model.ocfda_wrappers = tuple(wrappers)
    model.ocfda_parameter_names = tuple(
        name for name, parameter in model.named_parameters() if parameter is not None and OCFDA_PARAMETER_MARKER in name
    )
    model.optimizer_trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model.requires_grad_trainable_params = model.optimizer_trainable_params
    expected = 3 * len(layers) * k * int(model.config.hidden_size)
    if model.optimizer_trainable_params != expected:
        raise RuntimeError(
            f"OCFDA trainable scalar count is {model.optimizer_trainable_params}, expected {expected}"
        )
    return model


def get_ocfda_model_state_dict(model: nn.Module, state_dict: Optional[Mapping[str, torch.Tensor]] = None) -> dict[str, torch.Tensor]:
    if state_dict is None:
        state_dict = model.state_dict()
    return {
        name: tensor
        for name, tensor in state_dict.items()
        if OCFDA_PARAMETER_MARKER in name or OCFDA_SUPPORT_MARKER in name
    }


def is_ocfda_parameter(name: str, parameter: Optional[torch.Tensor] = None) -> bool:
    return OCFDA_PARAMETER_MARKER in name or bool(getattr(parameter, "_ocfda_owned", False))


def host_tensor_hashes(model: nn.Module) -> dict[str, str]:
    """Hash parameters and buffers, including names and tensor metadata."""

    hashes: dict[str, str] = {}
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    for name, tensor in tensors:
        value = tensor.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        hashes[name] = digest.hexdigest()
    return hashes


def verify_host_tensor_hashes(model: nn.Module, expected: Mapping[str, str]) -> dict[str, object]:
    actual = host_tensor_hashes(model)
    missing = sorted(set(expected) - set(actual))
    changed = sorted(name for name in expected if name in actual and expected[name] != actual[name])
    if missing or changed:
        details = []
        if missing:
            details.append(f"missing={missing[:3]}")
        if changed:
            details.append(f"changed={changed[:3]}")
        raise RuntimeError("Pretrained host tensors changed: " + ", ".join(details))
    return {
        "match": True,
        "tensor_count": len(expected),
        "hashes": dict(expected),
        "after_hashes": {name: actual[name] for name in expected},
    }


@contextmanager
def graft_detached(model: nn.Module) -> Iterator[nn.Module]:
    wrappers = tuple(getattr(model, "ocfda_wrappers", ()))
    previous = [wrapper.graft_enabled for wrapper in wrappers]
    for wrapper in wrappers:
        wrapper.graft_enabled = False
    try:
        yield model
    finally:
        for wrapper, enabled in zip(wrappers, previous):
            wrapper.graft_enabled = enabled


def verify_optimizer_ownership(optimizer: torch.optim.Optimizer, model: nn.Module) -> dict[str, object]:
    allowed = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if is_ocfda_parameter(name, parameter) and parameter.requires_grad
    }
    if not allowed:
        raise RuntimeError("No trainable OCFDA parameters found")
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if trainable != allowed:
        raise RuntimeError(f"Found {len(trainable - allowed)} trainable non-OCFDA parameters")
    observed = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    observed_ids = {id(parameter) for parameter in observed}
    unexpected = len(observed_ids - allowed)
    missing = len(allowed - observed_ids)
    if unexpected or missing or len(observed) != len(allowed):
        raise RuntimeError(
            f"Optimizer ownership violation: expected {len(allowed)} OCFDA parameters, "
            f"observed {len(observed)}, unexpected={unexpected}, missing={missing}"
        )
    invalid_state = [id(parameter) for parameter in optimizer.state if id(parameter) not in allowed]
    if invalid_state:
        raise RuntimeError(f"Optimizer contains state for {len(invalid_state)} non-OCFDA parameters")
    return {
        "optimizer_parameter_count": len(observed),
        "trainable_parameter_count": len(trainable),
        "optimizer_state_parameter_count": len(optimizer.state),
        "only_ocfda": True,
    }


def verify_ocfda_detach(
    model: nn.Module,
    expected_host_hashes: Mapping[str, str],
    host_report: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    wrappers = tuple(getattr(model, "ocfda_wrappers", ()))
    if not wrappers:
        raise RuntimeError("No OCFDA wrappers found for the detach check")
    with graft_detached(model):
        report = dict(host_report) if host_report is not None else verify_host_tensor_hashes(model, expected_host_hashes)
        for wrapper in wrappers:
            probe = torch.randn(
                (1, 2, wrapper.in_features),
                dtype=wrapper.weight.dtype,
                device=wrapper.weight.device,
            )
            with torch.no_grad():
                expected = F.linear(probe, wrapper.weight, wrapper.bias)
                actual = wrapper(probe)
            if not torch.equal(expected, actual):
                raise RuntimeError(f"Detached OCFDA layer does not match its host for {wrapper.projection}")
    report["detached"] = True
    return report


def verify_zero_graft_noop(model: nn.Module) -> None:
    """Check exact equality with the frozen host for zero-valued deltas."""

    wrappers = tuple(getattr(model, "ocfda_wrappers", ()))
    if not wrappers:
        raise RuntimeError("No OCFDA wrappers found for the zero-delta check")
    for wrapper in wrappers:
        if torch.count_nonzero(wrapper.graft_delta.detach()).item() != 0:
            raise RuntimeError(f"Zero-delta check received a trained OCFDA parameter for {wrapper.projection}")
        probe = torch.randn(
            (1, 2, wrapper.in_features),
            dtype=wrapper.weight.dtype,
            device=wrapper.weight.device,
        )
        with torch.no_grad():
            previous_enabled = wrapper.graft_enabled
            wrapper.graft_enabled = False
            host_output = wrapper(probe)
            wrapper.graft_enabled = previous_enabled
            graft_output = wrapper(probe)
        if not torch.equal(host_output, graft_output):
            raise RuntimeError(f"Zero OCFDA delta is not an exact no-op for {wrapper.projection}")
