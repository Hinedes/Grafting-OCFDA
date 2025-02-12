import torch
import torch.nn as nn
import math

from src.mask import prepare_super_mask

class dense_plus_sparse_linear(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, input, weight, indices, values, bias=None):
        ctx.save_for_backward(input, weight, indices, values, bias)
        
        dense_plus_sparse = weight.view(-1).scatter_add(0, indices.to(torch.int64), values)
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
        assert 0.0 <= sparse_rate <= 1.0, "sparse_rate shoud be a ratio between 0 and 1"
        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.num_elements = self.weight.numel()
        self.num_nonzero = int(self.weight.numel() * (sparse_rate))
        self.sparse_rate = sparse_rate

        if getattr(base_layer, "state", None) is not None:
            self.state = base_layer.state

        if indices is None:
            indices = torch.randperm(self.num_elements-1)[:self.num_nonzero]
        indices = indices.to(dtype=torch.int32, device=self.weight.device)
        
        self.values = nn.Parameter(
            torch.zeros(self.num_nonzero, dtype=torch.float32, device=self.weight.device)
        )
        self.register_buffer('indices', indices)
        
    def forward(self, input):
        return dense_plus_sparse_linear.apply(input, self.weight, self.indices, self.values, self.bias)
    

def get_dense_plus_sparse_model(model, target_modules_list, sparse_rate=0.01, indices_choice="random", tokenizer=None, exception=[]):
    if indices_choice == "super":
        assert tokenizer is not None, "`Super` option requires tokenizer to determine outliers indices."
        prepare_super_mask(model, tokenizer, dev=model.device, outliers_ratio=sparse_rate)
    
    for name, p in model.named_parameters():
        if not any([item in name for item in exception]):
            p.requires_grad_(False)

    def _get_submodules(key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(parent_module, child_name, old_module):
        if hasattr(old_module, "wanda_topk_indices"):
            indices = old_module.wanda_topk_indices
        else:
            indices = None
        new_module = SparseDenseLinear(old_module, sparse_rate=sparse_rate, indices=indices)
        new_module.weight.requires_grad_(False)
        setattr(parent_module, child_name, new_module)

        # new_module.weight = old_module.weight
        # if old_module.bias is not None:
        #     new_module.bias = old_module.bias
        # if getattr(old_module, "state", None) is not None:
        #     new_module.state = old_module.state

        # # dispatch to correct device
        # for name, module in new_module.named_modules():
        #     if "lora_" in name:
        #         module.to(old_module.weight.device)

    for module_name, _ in model.named_modules():
        if not any(module_name.endswith(target_key) for target_key in target_modules_list):
            continue

        parent, target, target_name = _get_submodules(module_name)
        _replace_module(parent, target_name, target)