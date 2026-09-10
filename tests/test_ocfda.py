from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from fine_tuning.ocfda import (
    OCFDA_PROJECTIONS,
    OCFDALinear,
    generate_supports,
    get_ocfda_model,
    get_ocfda_model_state_dict,
    graft_detached,
    host_tensor_hashes,
    verify_host_tensor_hashes,
    verify_optimizer_ownership,
    verify_zero_graft_noop,
)

FROZEN_LAYERS = 16
FROZEN_HIDDEN = 2048
FROZEN_K = 57
FROZEN_OCFDA_SCALARS = 5_603_328


class TinyMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 6, bias=False)
        self.up_proj = torch.nn.Linear(4, 6, bias=False)
        self.down_proj = torch.nn.Linear(6, 4, bias=False)


class TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyMlp()


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])
        self.config = SimpleNamespace(intermediate_size=6, hidden_size=4)


def swiglu(mlp: torch.nn.Module, probe: torch.Tensor) -> torch.Tensor:
    return mlp.down_proj(F.silu(mlp.gate_proj(probe)) * mlp.up_proj(probe))


def empirical_dense_delta(wrapped: OCFDALinear, host: torch.nn.Linear) -> torch.Tensor:
    probe = torch.eye(host.in_features)
    with torch.no_grad():
        delta = wrapped(probe) - host(probe)
    return delta.transpose(0, 1)


def graft_deltas(model: torch.nn.Module):
    return [parameter for name, parameter in model.named_parameters() if "graft_delta" in name]


def test_aligned_supports_are_shared_across_gate_up_down() -> None:
    supports = generate_supports(6, 2, k=2, geometry="aligned", support_seed=11)
    for layer in ("0", "1"):
        assert len({tuple(support) for support in supports[layer].values()}) == 1


def test_independent_supports_are_separately_drawn_but_share_the_aligned_gate() -> None:
    aligned = generate_supports(8192, FROZEN_LAYERS, k=FROZEN_K, geometry="aligned", support_seed=1001)
    independent = generate_supports(8192, FROZEN_LAYERS, k=FROZEN_K, geometry="independent", support_seed=1001)
    for layer in (0, 7, FROZEN_LAYERS - 1):
        key = str(layer)
        assert independent[key]["gate_proj"] == aligned[key]["gate_proj"]
        assert len({tuple(independent[key][projection]) for projection in OCFDA_PROJECTIONS}) == 3


def test_frozen_ocfda_scalar_count_formula() -> None:
    assert 3 * FROZEN_LAYERS * FROZEN_K * FROZEN_HIDDEN == FROZEN_OCFDA_SCALARS


def test_aligned_and_independent_own_the_same_trainable_budget() -> None:
    counts = []
    for geometry in ("aligned", "independent"):
        model = get_ocfda_model(TinyModel(), geometry=geometry, support_seed=7, k=2)
        counts.append(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    assert counts == [48, 48]


def test_zero_graft_is_bit_exact_noop_and_host_stays_frozen() -> None:
    torch.manual_seed(3)
    model = TinyModel()
    expected_hashes = host_tensor_hashes(model)
    probe = torch.randn(1, 2, 4)
    host_output = swiglu(model.model.layers[0].mlp, probe)

    wrapped = get_ocfda_model(model, geometry="aligned", support_seed=7, k=2)
    verify_zero_graft_noop(wrapped)
    torch.testing.assert_close(swiglu(wrapped.model.layers[0].mlp, probe), host_output, rtol=0, atol=0)
    verify_host_tensor_hashes(wrapped, expected_hashes)
    assert all(
        parameter.requires_grad == ("graft_delta" in name)
        for name, parameter in wrapped.named_parameters()
    )
    state = get_ocfda_model_state_dict(wrapped)
    assert len([name for name in state if "graft_delta" in name]) == 6
    assert len([name for name in state if "graft_support" in name]) == 6


def test_pretrained_tensors_are_bit_identical_after_an_adamw_step() -> None:
    torch.manual_seed(31)
    base = TinyModel()
    expected_hashes = host_tensor_hashes(base)

    torch.manual_seed(31)
    model = get_ocfda_model(TinyModel(), geometry="aligned", support_seed=2, k=2)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
        weight_decay=0.0,
    )
    verify_optimizer_ownership(optimizer, model)

    probe = torch.randn(4, 4)
    loss = swiglu(model.model.layers[0].mlp, probe).square().mean()
    loss.backward()
    optimizer.step()

    verify_host_tensor_hashes(model, expected_hashes)
    report = verify_optimizer_ownership(optimizer, model)
    assert report["optimizer_parameter_count"] == 6
    assert report["only_ocfda"] is True
    for name, parameter in model.named_parameters():
        if "graft_delta" not in name:
            assert parameter.grad is None, name


def test_gradients_reach_every_graft_projection_and_no_host_tensor() -> None:
    torch.manual_seed(32)
    model = get_ocfda_model(TinyModel(), geometry="independent", support_seed=3, k=2)
    probe = torch.randn(8, 4)
    hidden = swiglu(model.model.layers[0].mlp, probe)
    swiglu(model.model.layers[1].mlp, hidden).square().mean().backward()

    graft_seen = 0
    for name, parameter in model.named_parameters():
        if "graft_delta" in name:
            graft_seen += 1
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert torch.count_nonzero(parameter.grad).item() > 0, name
        else:
            assert parameter.grad is None, name
    assert graft_seen == 6


def test_swiglu_forward_matches_manual_dense_delta_reconstruction() -> None:
    torch.manual_seed(41)
    base = TinyModel()
    host_gate = base.model.layers[0].mlp.gate_proj.weight.detach().clone()
    host_up = base.model.layers[0].mlp.up_proj.weight.detach().clone()
    host_down = base.model.layers[0].mlp.down_proj.weight.detach().clone()

    torch.manual_seed(41)
    model = get_ocfda_model(TinyModel(), geometry="aligned", support_seed=4, k=2)
    mlp = model.model.layers[0].mlp
    support = mlp.gate_proj.graft_support.tolist()
    gate_delta = torch.randn(2, 4)
    up_delta = torch.randn(2, 4)
    down_delta = torch.randn(4, 2)
    with torch.no_grad():
        mlp.gate_proj.graft_delta.copy_(gate_delta)
        mlp.up_proj.graft_delta.copy_(up_delta)
        mlp.down_proj.graft_delta.copy_(down_delta)

    dense_gate = host_gate.clone()
    dense_up = host_up.clone()
    dense_down = host_down.clone()
    dense_gate[support] += gate_delta
    dense_up[support] += up_delta
    dense_down[:, support] += down_delta

    probe = torch.randn(5, 4)
    expected = F.linear(
        F.silu(F.linear(probe, dense_gate)) * F.linear(probe, dense_up),
        dense_down,
    )
    torch.testing.assert_close(swiglu(mlp, probe), expected, rtol=1e-5, atol=1e-6)


def test_aligned_geometry_touches_exactly_supported_rows_and_columns() -> None:
    support = [1, 5, 9]
    gate_host = torch.nn.Linear(4, 12, bias=False)
    up_host = torch.nn.Linear(4, 12, bias=False)
    down_host = torch.nn.Linear(12, 4, bias=False)
    gate = OCFDALinear(gate_host, "gate_proj", support)
    up = OCFDALinear(up_host, "up_proj", support)
    down = OCFDALinear(down_host, "down_proj", support)

    gate_delta = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    up_delta = torch.arange(101, 113, dtype=torch.float32).reshape(3, 4)
    down_delta = torch.arange(201, 213, dtype=torch.float32).reshape(4, 3)
    with torch.no_grad():
        gate.graft_delta.copy_(gate_delta)
        up.graft_delta.copy_(up_delta)
        down.graft_delta.copy_(down_delta)
        gate_host.weight.zero_()
        up_host.weight.zero_()
        down_host.weight.zero_()

    expected_gate = torch.zeros(12, 4)
    expected_gate[support] = gate_delta
    expected_up = torch.zeros(12, 4)
    expected_up[support] = up_delta
    expected_down = torch.zeros(4, 12)
    expected_down[:, support] = down_delta

    assert torch.equal(empirical_dense_delta(gate, gate_host), expected_gate)
    assert torch.equal(empirical_dense_delta(up, up_host), expected_up)
    assert torch.equal(empirical_dense_delta(down, down_host), expected_down)


def test_independent_geometry_touches_three_unrelated_coordinates() -> None:
    gate_host = torch.nn.Linear(4, 6, bias=False)
    up_host = torch.nn.Linear(4, 6, bias=False)
    down_host = torch.nn.Linear(6, 4, bias=False)
    gate = OCFDALinear(gate_host, "gate_proj", [1])
    up = OCFDALinear(up_host, "up_proj", [2])
    down = OCFDALinear(down_host, "down_proj", [3])

    with torch.no_grad():
        gate.graft_delta.fill_(3.0)
        up.graft_delta.fill_(5.0)
        down.graft_delta.fill_(7.0)
        gate_host.weight.zero_()
        up_host.weight.zero_()
        down_host.weight.zero_()

    expected_gate = torch.zeros(6, 4)
    expected_gate[1] = 3.0
    expected_up = torch.zeros(6, 4)
    expected_up[2] = 5.0
    expected_down = torch.zeros(4, 6)
    expected_down[:, 3] = 7.0

    gate_dense = empirical_dense_delta(gate, gate_host)
    up_dense = empirical_dense_delta(up, up_host)
    down_dense = empirical_dense_delta(down, down_host)
    assert torch.equal(gate_dense, expected_gate)
    assert torch.equal(up_dense, expected_up)
    assert torch.equal(down_dense, expected_down)
    assert torch.count_nonzero(gate_dense[2]).item() == 0
    assert torch.count_nonzero(up_dense[1]).item() == 0
    assert torch.count_nonzero(down_dense[:, 1]).item() == 0


def test_detach_and_checkpoint_round_trip_restore_grafted_logits() -> None:
    torch.manual_seed(51)
    base = TinyModel()
    probe = torch.randn(4, 4)
    base_logits = swiglu(base.model.layers[0].mlp, probe)

    torch.manual_seed(51)
    model = get_ocfda_model(TinyModel(), geometry="aligned", support_seed=6, k=2)
    with torch.no_grad():
        for wrapper in model.ocfda_wrappers:
            wrapper.graft_delta.copy_(torch.randn_like(wrapper.graft_delta))
    grafted_logits = swiglu(model.model.layers[0].mlp, probe)
    assert not torch.equal(grafted_logits, base_logits)

    with graft_detached(model):
        assert torch.equal(swiglu(model.model.layers[0].mlp, probe), base_logits)

    state = get_ocfda_model_state_dict(model)
    torch.manual_seed(51)
    reloaded = get_ocfda_model(TinyModel(), geometry="aligned", support_seed=6, k=2)
    reloaded.load_state_dict(state, strict=False)
    assert torch.equal(swiglu(reloaded.model.layers[0].mlp, probe), grafted_logits)


def test_graft_parameters_learn_and_detach_restores_the_host() -> None:
    torch.manual_seed(61)
    base = TinyModel()
    probe = torch.randn(16, 4)
    host_output = swiglu(base.model.layers[0].mlp, probe).detach()
    host_hashes = host_tensor_hashes(base)

    torch.manual_seed(61)
    model = get_ocfda_model(TinyModel(), geometry="aligned", support_seed=8, k=2)
    mlp = model.model.layers[0].mlp
    with torch.no_grad():
        for wrapper in model.ocfda_wrappers:
            wrapper.graft_delta.copy_(0.5 * torch.randn_like(wrapper.graft_delta))
        teacher = swiglu(mlp, probe).detach().clone()
        for wrapper in model.ocfda_wrappers:
            wrapper.graft_delta.zero_()

    initial_loss = F.mse_loss(swiglu(mlp, probe), teacher).item()
    assert initial_loss > 0
    optimizer = torch.optim.AdamW(graft_deltas(model), lr=1e-2)
    for _ in range(300):
        optimizer.zero_grad()
        loss = F.mse_loss(swiglu(mlp, probe), teacher)
        loss.backward()
        optimizer.step()
    final_loss = F.mse_loss(swiglu(mlp, probe), teacher).item()
    assert final_loss < initial_loss
    assert final_loss < 0.5 * initial_loss
    verify_host_tensor_hashes(model, host_hashes)
    with graft_detached(model):
        assert torch.equal(swiglu(mlp, probe), host_output)


def test_same_seed_reproduces_identical_miniature_run() -> None:
    def run_once():
        torch.manual_seed(71)
        model = get_ocfda_model(TinyModel(), geometry="aligned", support_seed=9, k=2)
        probe = torch.randn(8, 4)
        hidden = swiglu(model.model.layers[0].mlp, probe)
        swiglu(model.model.layers[1].mlp, hidden).square().mean().backward()
        return [parameter.grad.detach().clone() for parameter in graft_deltas(model)]

    first = run_once()
    second = run_once()
    assert len(first) == len(second) == 6
    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_supports_are_seed_deterministic_and_seed_sensitive() -> None:
    first = generate_supports(8192, FROZEN_LAYERS, k=FROZEN_K, geometry="aligned", support_seed=1001)
    repeat = generate_supports(8192, FROZEN_LAYERS, k=FROZEN_K, geometry="aligned", support_seed=1001)
    other = generate_supports(8192, FROZEN_LAYERS, k=FROZEN_K, geometry="aligned", support_seed=1002)
    assert first == repeat
    assert first["0"]["gate_proj"] != other["0"]["gate_proj"]
    assert first[str(FROZEN_LAYERS - 1)]["down_proj"] != other[str(FROZEN_LAYERS - 1)]["down_proj"]


def test_ocfda_rejects_supports_not_derived_from_seed() -> None:
    supports = generate_supports(6, 2, k=2, geometry="aligned", support_seed=11)
    original = set(supports["0"]["gate_proj"])
    replacement = [index for index in range(6) if index not in original][:2]
    supports["0"]["gate_proj"] = replacement
    with pytest.raises(ValueError, match="deterministic support seed"):
        get_ocfda_model(TinyModel(), geometry="aligned", support_seed=11, k=2, supports=supports)
