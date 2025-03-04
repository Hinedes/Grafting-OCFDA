import torch
import torch.nn as nn
import math

from src.mask import prepare_super_mask


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
    def __init__(self, base_layer, r: int = 8, indices=None):
        super().__init__()
        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.num_elements = self.weight.numel()

        in_features, out_features = self.weight.shape

        super_params = (out_features + in_features) * r

        if getattr(base_layer, "state", None) is not None:
            self.state = base_layer.state

        if indices is None:
            # indices = torch.randperm(self.num_elements-1)
            indices = torch.randint(0, self.num_elements, (super_params,))
        indices = indices.to(dtype=torch.int32, device=self.weight.device)[:super_params]
        
        self.values = nn.Parameter(
            torch.zeros(super_params, dtype=torch.float32, device=self.weight.device)
        )
        self.register_buffer('indices', indices)
        
    def forward(self, input):
        return DensePlusSparseLinear.apply(input, self.weight, self.indices, self.values, self.bias)


def get_dense_plus_sparse_model(model, target_modules_list, r: int = 8, indices_choice="random", tokenizer=None, exception=[]):
    if indices_choice == "super":
        assert tokenizer is not None, "`Super` option requires tokenizer to determine outliers indices."
        prepare_super_mask(model, tokenizer, dev=model.device, r=r)

    def _get_submodules(key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(parent_module, child_name, old_module):
        indices = getattr(old_module.weight, "wanda_topk_indices", None)
        new_module = SparseDenseLinear(old_module, r=r, indices=indices)
        setattr(parent_module, child_name, new_module)

    for module_name, _ in model.named_modules():
        if not any(module_name.endswith(target_key) for target_key in target_modules_list):
            continue
        
        parent, target, target_name = _get_submodules(module_name)
        _replace_module(parent, target_name, target)
    
    for name, p in model.named_parameters():
        if not ("values" in name or any([item in name for item in exception])):
            p.requires_grad_(False)
    
    return model


def get_sparse_dense_model_state_dict(model, state_dict=None):
    if state_dict is None:
        state_dict = model.state_dict()
    return {k: state_dict[k] for k in state_dict if "values" in k or "indices" in k}
