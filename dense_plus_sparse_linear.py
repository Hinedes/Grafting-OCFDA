import torch
import torch.nn as nn
import math

from src.mask import prepare_super_mask


def random_sparse_indices(num_elements: int, train_num: int, device) -> torch.Tensor:
    if train_num <= 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    if train_num >= num_elements:
        return torch.arange(num_elements, dtype=torch.int32, device=device)

    selected = torch.empty(0, dtype=torch.int64, device=device)
    while selected.numel() < train_num:
        remaining = train_num - selected.numel()
        sample_count = min(num_elements, max(remaining + remaining // 10 + 16, remaining))
        sample = torch.randint(0, num_elements, (sample_count,), dtype=torch.int64, device=device)
        selected = torch.unique(torch.cat([selected, sample]))

    return selected[:train_num].to(dtype=torch.int32)


class DensePlusSparseLinear(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, input, weight, indices, values, bias=None):
        ctx.save_for_backward(input, weight, indices, values, bias)
        
        dense_plus_sparse = weight.view(-1).scatter_add(0, indices.to(torch.int64), values.to(weight.dtype))
        dense_plus_sparse = dense_plus_sparse.view_as(weight)

        return torch.nn.functional.linear(input, dense_plus_sparse, bias)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        input, weight, indices, values, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_indices = grad_values = grad_bias = None

        dense_plus_sparse = weight.view(-1).scatter_add(0, indices.to(torch.int64), values)
        dense_plus_sparse = dense_plus_sparse.view_as(weight)

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, dense_plus_sparse)

        if any(ctx.needs_input_grad[1:]):
            if input.dim() != 2:
                grad_output = grad_output.reshape(-1, grad_output.shape[-1])
                input = input.reshape(-1, input.shape[-1])
            grad_matrix = grad_output.t().mm(input)
                
                # grad_matrix = torch.bmm(grad_output.transpose(1, 2), input.to(grad_output.dtype)).to(weight.dtype)
                # grad_matrix = grad_matrix.sum(dim=0)

            if ctx.needs_input_grad[1]:
                grad_weight = grad_matrix
            
            if ctx.needs_input_grad[3]:
                grad_values = grad_matrix.view(-1).gather(0, indices.to(torch.int64))

        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(dim=0)# if input.dim() == 2 else grad_output.sum(dim=(0, 1))

        return grad_input, grad_weight, grad_indices, grad_values, grad_bias


class SparseDenseLinear(nn.Module):
    def __init__(self, base_layer, sparse_rate: float, indices=None):
        super().__init__()
        assert 0.0 <= sparse_rate <= 1.0, "sparse_rate should be a ratio between 0 and 1"
        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.num_elements = self.weight.numel()

        in_features, out_features = self.weight.shape

        #super_params = (out_features + in_features) * r
        super_params = min(int(sparse_rate * self.weight.numel()) + 1, self.weight.numel())

        if getattr(base_layer, "state", None) is not None:
            self.state = base_layer.state

        if indices is None:
            indices = random_sparse_indices(self.num_elements, super_params, self.weight.device)
        indices = indices.to(dtype=torch.int32, device=self.weight.device)[:super_params]
        
        self.values = nn.Parameter(
            torch.zeros(super_params, dtype=torch.float32, device=self.weight.device)
        )
        self.register_buffer('indices', indices)
        
    def forward(self, input):
        return DensePlusSparseLinear.apply(input, self.weight, self.indices, self.values, self.bias)


def get_dense_plus_sparse_model(
        model,
        target_modules_list,
        sparse_rate: float,
        indices_choice="random",
        tokenizer=None,
        exception=[],
        calibration_data="c4",
        calibration_nsamples=128,
        calibration_seed=228,
):
    if indices_choice == "super":
        assert tokenizer is not None, "`Super` option requires tokenizer to determine outliers indices."
        prepare_super_mask(
            model,
            tokenizer,
            dev=model.device,
            sparse_rate=sparse_rate,
            nsamples=calibration_nsamples,
            seed=calibration_seed,
            calibration_data=calibration_data,
        )

    def _get_submodules(key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    if indices_choice not in {"random", "super"}:
        raise ValueError("indices_choice must be either 'random' or 'super'.")

    replaced_modules = 0
    total_indices = 0
    total_unique_indices = 0

    def _replace_module(parent_module, child_name, old_module):
        nonlocal replaced_modules, total_indices, total_unique_indices
        if indices_choice == "super":
            indices = getattr(old_module.weight, "wanda_topk_indices", None)
            if indices is None:
                raise RuntimeError("Wanda indices were not prepared for a Super sparse layer.")
        else:
            indices = None
        new_module = SparseDenseLinear(old_module, sparse_rate=sparse_rate, indices=indices)
        setattr(parent_module, child_name, new_module)
        replaced_modules += 1
        total_indices += int(new_module.indices.numel())
        total_unique_indices += int(torch.unique(new_module.indices).numel())

    for module_name, _ in model.named_modules():
        if not any(module_name.endswith(target_key) for target_key in target_modules_list):
            continue
        
        parent, target, target_name = _get_submodules(module_name)
        _replace_module(parent, target_name, target)

    print(
        "Sparse mask source:",
        "wanda" if indices_choice == "super" else "random",
        "replaced modules:",
        replaced_modules,
        "sparse entries:",
        total_indices,
        "unique sparse entries:",
        total_unique_indices,
    )
    
    for name, p in model.named_parameters():
        if not ("values" in name or any([item in name for item in exception])):
            p.requires_grad_(False)
    
    return model


def get_sparse_dense_model_state_dict(model, state_dict=None):
    if state_dict is None:
        state_dict = model.state_dict()
    return {k: state_dict[k] for k in state_dict if "values" in k or "indices" in k}
