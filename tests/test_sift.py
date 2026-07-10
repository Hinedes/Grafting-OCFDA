import torch
import torch.nn as nn

from fine_tuning.baselines import SIFT


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.proj(inputs)


def test_sift_uses_sparse_optimizer_vectors_and_fixes_support() -> None:
    torch.manual_seed(7)
    model = TinyModel()
    sift = SIFT(model, sparse_module=["proj.weight"], sparse_rate=0.25)

    optimizer_parameters = list(sift.parameters_in_optimizer())
    assert len(optimizer_parameters) == 1
    assert optimizer_parameters[0].numel() == 4
    assert sift.get_trainable_num() == 4

    inputs = torch.randn(2, 4)
    model(inputs).sum().backward()
    assert model.proj.weight.grad is not None
    assert torch.count_nonzero(model.proj.weight.grad) == 0
    assert sift._indices["proj.weight"] is not None
    assert sift._indices["proj.weight"].numel() == 4

    model.zero_grad(set_to_none=True)
    model(inputs).sum().backward()
    assert optimizer_parameters[0].grad is not None
    assert optimizer_parameters[0].grad.numel() == 4


def test_sift_random_support_has_unique_indices() -> None:
    torch.manual_seed(3)
    model = TinyModel()
    sift = SIFT(model, sparse_module=["proj.weight"], sparse_rate=0.5, random_indices=True)
    model(torch.randn(1, 4)).sum().backward()

    indices = sift._indices["proj.weight"]
    assert indices is not None
    assert indices.unique().numel() == indices.numel()
