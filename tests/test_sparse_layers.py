import torch
import torch.nn.functional as F

from dense_plus_sparse_linear import (
    DensePlusSparseLinear,
    SparseDenseLinear,
    random_sparse_indices,
    select_magnitude_indices,
)
from fine_tuning.baselines.sift import _sample_without_replacement
from src.mask import select_rowwise_bottom_then_global_fill


def test_magnitude_selection_uses_expected_extremes() -> None:
    weight = torch.tensor([[1.0, -6.0, 2.0], [-5.0, 3.0, -4.0]])
    top = select_magnitude_indices(weight, sparse_rate=0.5, largest=True)
    bottom = select_magnitude_indices(weight, sparse_rate=0.5, largest=False)

    assert set(top.tolist()) == {1, 3, 4, 5}
    assert set(bottom.tolist()) == {0, 2, 4, 5}


def test_rowwise_bottom_selection_fills_global_remainder() -> None:
    metric = torch.tensor(
        [
            [1.0, 10.0, 11.0, 12.0, 13.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    )
    indices = select_rowwise_bottom_then_global_fill(metric, sparse_rate=0.3)

    assert indices.numel() == int(metric.numel() * 0.3)
    assert set(indices.tolist()) == {0, 5, 6}


def test_sparse_autograd_matches_dense_reference() -> None:
    torch.manual_seed(11)
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4)
    values = torch.randn(3, requires_grad=True)
    indices = torch.tensor([0, 7, 19], dtype=torch.int32)

    reference_weight = weight.flatten().scatter_add(0, indices.long(), values).view_as(weight)
    reference = F.linear(inputs, reference_weight)
    reference.sum().backward()
    reference_input_grad = inputs.grad.clone()
    reference_values_grad = values.grad.clone()

    inputs.grad = None
    values.grad = None
    actual = DensePlusSparseLinear.apply(inputs, weight, indices, values, None)
    actual.sum().backward()

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(inputs.grad, reference_input_grad)
    torch.testing.assert_close(values.grad, reference_values_grad)


def test_sparse_layer_is_identical_to_base_at_initialization() -> None:
    torch.manual_seed(5)
    base = torch.nn.Linear(4, 3, bias=True)
    layer = SparseDenseLinear(base, sparse_rate=0.25, indices=torch.tensor([0, 2, 5, 8]))
    inputs = torch.randn(2, 4)

    torch.testing.assert_close(layer(inputs), base(inputs))


def test_random_sparse_sampler_is_not_biased_toward_low_indices() -> None:
    means = []
    for seed in range(100):
        torch.manual_seed(seed)
        indices = random_sparse_indices(100, 10, torch.device("cpu"))
        assert indices.unique().numel() == 10
        means.append(indices.float().mean())

    assert 40.0 < torch.stack(means).mean().item() < 60.0


def test_sift_random_sampler_is_not_biased_toward_low_indices() -> None:
    means = []
    for seed in range(100):
        torch.manual_seed(seed)
        indices = _sample_without_replacement(100, 10, torch.device("cpu"))
        assert indices.unique().numel() == 10
        means.append(indices.float().mean())

    assert 40.0 < torch.stack(means).mean().item() < 60.0
