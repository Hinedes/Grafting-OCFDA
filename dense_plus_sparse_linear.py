import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class DensePlusSparseLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, indices, values, bias=None):
        ctx.save_for_backward(input, weight, indices, values, bias)
        
        sparse_update = weight.view(-1).scatter_add(0, indices.to(torch.int64), values)
        sparse_update = sparse_update.view_as(weight)
        
        output = torch.nn.functional.linear(input.to(weight.dtype), sparse_update, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, indices, values, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_indices = grad_values = grad_bias = None

        # Recompute sparse_update
        sparse_update = weight.view(-1).scatter_add(0, indices.to(torch.int64), values)
        sparse_update = sparse_update.view_as(weight)

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, sparse_update)

        if any(ctx.needs_input_grad[1:]):
            if input.dim() != 2:
                grad_output = grad_output.reshape(-1, grad_output.shape[-1])
                input = input.reshape(-1, input.shape[-1])
            grad_matrix = grad_output.t().mm(input.to(grad_output.dtype))
                
                # grad_matrix = torch.bmm(grad_output.transpose(1, 2), input.to(grad_output.dtype)).to(weight.dtype)
                # grad_matrix = grad_matrix.sum(dim=0)

            if ctx.needs_input_grad[1]:
                grad_weight = grad_matrix
            
            if ctx.needs_input_grad[3]:
                grad_values = grad_matrix.view(-1)[indices.to(torch.int64)]

        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(dim=0)# if input.dim() == 2 else grad_output.sum(dim=(0, 1))

        return grad_input, grad_weight, grad_indices, grad_values, grad_bias
