"""SIFT-style sparse incremental fine-tuning.

This module implements the update rule described by Song et al., "Sparse is
Enough in Fine-tuning Pre-trained Large Language Models" (ICML 2024). The
first backward pass fixes either a TopK-gradient or random support for every
selected matrix. Adam then operates only on sparse value vectors; their
updates are folded into the frozen dense weights before the next optimizer
step.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn


def _sample_without_replacement(size: int, count: int, device: torch.device) -> torch.Tensor:
    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count >= size:
        return torch.arange(size, dtype=torch.long, device=device)

    selected = torch.empty(0, dtype=torch.long, device=device)
    while selected.numel() < count:
        remaining = count - selected.numel()
        draw = torch.randint(size, (remaining,), device=device)
        selected = torch.unique(torch.cat((selected, draw)))
    return selected


class SIFT:
    """Attach SIFT sparse-gradient hooks to selected model parameters."""

    def __init__(
        self,
        model: nn.Module,
        sparse_module: Sequence[str],
        sparse_rate: float,
        exception: Optional[Sequence[str]] = None,
        grad_acc: int = 1,
        gradient_checkpointing: bool = False,
        random_indices: bool = False,
    ) -> None:
        if not 0.0 <= sparse_rate <= 1.0:
            raise ValueError("sparse_rate must be between 0 and 1")
        if grad_acc < 1:
            raise ValueError("grad_acc must be at least 1")

        self.model = model
        self.sparse_module = tuple(sparse_module)
        self.exception = tuple(exception or ())
        self.sparse_rate = sparse_rate
        self.grad_acc = grad_acc
        self.random_indices = random_indices
        self.total_num = sum(parameter.numel() for parameter in model.parameters())
        self._dense_parameters: list[tuple[str, nn.Parameter]] = []
        self._optimizer_parameters: list[tuple[str, nn.Parameter]] = []
        self._sparse_values: dict[str, nn.Parameter] = {}
        self._indices: dict[str, Optional[torch.Tensor]] = {}
        self._backward_counts: dict[str, int] = {}

        first_name = next(iter(model.named_parameters()), (None, None))[0]
        for name, parameter in model.named_parameters():
            if any(module in name for module in self.sparse_module):
                self._register_sparse_parameter(name, parameter)
            elif any(item in name for item in self.exception):
                parameter.requires_grad_(True)
                self._dense_parameters.append((name, parameter))
                self._optimizer_parameters.append((name, parameter))
            elif gradient_checkpointing and name == first_name:
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)

    def _register_sparse_parameter(self, name: str, parameter: nn.Parameter) -> None:
        if parameter.ndim != 2:
            raise ValueError(f"SIFT expects matrix parameters, got {name} with shape {tuple(parameter.shape)}")

        count = min(int(self.sparse_rate * parameter.numel()) + 1, parameter.numel())
        values = nn.Parameter(parameter.new_zeros(count), requires_grad=True)
        model_name = f"_sift_{name.replace('.', '_')}"
        self.model.register_parameter(model_name, values)

        parameter.requires_grad_(True)
        self._dense_parameters.append((name, parameter))
        self._optimizer_parameters.append((f"{name}.sift_values", values))
        self._sparse_values[name] = values
        self._indices[name] = None
        self._backward_counts[name] = 0
        parameter.register_hook(self._gradient_hook(name, parameter))

    def _gradient_hook(self, name: str, dense_parameter: nn.Parameter):
        def hook(gradient: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                values = self._sparse_values[name]
                flat_gradient = gradient.reshape(-1).to(device=values.device, dtype=values.dtype)
                indices = self._indices[name]

                if indices is None:
                    if self.random_indices:
                        indices = _sample_without_replacement(
                            dense_parameter.numel(), values.numel(), gradient.device
                        )
                    else:
                        indices = flat_gradient.float().abs().topk(values.numel()).indices
                    self._indices[name] = indices.to(device=dense_parameter.device, dtype=torch.long)
                    return torch.zeros_like(gradient)

                values_gradient = flat_gradient.gather(0, indices.to(flat_gradient.device))
                if values.grad is None:
                    values.grad = values_gradient.clone()
                else:
                    values.grad.add_(values_gradient)

                self._backward_counts[name] += 1
                if self._backward_counts[name] == self.grad_acc:
                    dense_parameter.view(-1).index_add_(
                        0,
                        indices.to(dense_parameter.device),
                        values.detach().to(dtype=dense_parameter.dtype, device=dense_parameter.device),
                    )
                    values.zero_()
                    self._backward_counts[name] = 0

            return torch.zeros_like(gradient)

        return hook

    def named_trainable_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        return iter(self._dense_parameters)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for _, parameter in self._dense_parameters)

    def named_parameters_in_optimizer(self) -> Iterable[tuple[str, nn.Parameter]]:
        return iter(self._optimizer_parameters)

    def parameters_in_optimizer(self) -> Iterable[nn.Parameter]:
        return (parameter for _, parameter in self._optimizer_parameters)

    def get_trainable_num(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters_in_optimizer())

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        state = {}
        for name, values in self._sparse_values.items():
            safe_name = name.replace(".", "_")
            indices = self._indices[name]
            if indices is None:
                indices = torch.empty(0, dtype=torch.long, device=values.device)
            state[f"{safe_name}.sparse_values"] = values.detach()
            state[f"{safe_name}.sparse_indices"] = indices.detach()
        return state

    def print_trainable_parameters(self) -> None:
        trainable = self.get_trainable_num()
        percentage = 100.0 * trainable / max(self.total_num, 1)
        print(f"trainable params: {trainable:,d} || all params: {self.total_num:,d} || trainable%: {percentage:.6f}")

    def set_trainer(self, trainer) -> None:
        self.grad_acc = trainer.args.gradient_accumulation_steps
