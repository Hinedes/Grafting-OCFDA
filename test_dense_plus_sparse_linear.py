import torch
import pytest

from dense_plus_sparse_linear import DensePlusSparseLinear

@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("input_shape", [(32, 10), (8, 16, 10)])
def test_custom_linear(input_shape, input_dtype, weight_dtype, index_dtype):
    torch.manual_seed(42)
    
    # Prepare dimensions
    in_features = input_shape[-1]
    out_features = 20
    batch_dims = input_shape[:-1]
    
    # Create inputs
    input = torch.randn(*input_shape, dtype=input_dtype, requires_grad=True)
    weight = torch.randn(out_features, in_features, dtype=weight_dtype, requires_grad=True)
    
    # Create sparse updates
    num_updates = 50
    indices = torch.randint(0, weight.numel(), (num_updates,), dtype=index_dtype)
    values = torch.randn(num_updates, dtype=weight_dtype, requires_grad=True)
    bias = torch.randn(out_features, dtype=weight_dtype, requires_grad=True)

    def run_naive():
        indices_unraveled = torch.stack(torch.unravel_index(indices, weight.size())).to(torch.int64)
        sparse = torch.sparse_coo_tensor(indices=indices_unraveled, values=values, size=weight.size())
        return torch.nn.functional.linear(input.to(weight.dtype), weight + sparse, bias)

    def run_custom():
        return DensePlusSparseLinear.apply(input, weight, indices, values, bias)

    # Forward pass comparison
    output_naive = run_naive()
    output_custom = run_custom()
    
    assert torch.allclose(output_naive, output_custom, rtol=1e-3, atol=1e-3), \
        "Forward passes don't match"

    # # Backward pass comparison
    # grad_output = torch.randn_like(output_naive)

    
    # # Naive backward
    # output_naive.backward(grad_output)
    output_naive.norm().backward()
    grad_input_naive = input.grad.clone()
    grad_weight_naive = weight.grad.clone()
    grad_values_naive = values.grad.clone()
    grad_bias_naive = bias.grad.clone()
    
    # Reset grads
    input.grad = None
    weight.grad = None
    values.grad = None
    bias.grad = None
    
    # # Custom backward
    # output_custom.backward(grad_output)
    output_custom.norm().backward()
    grad_input_custom = input.grad
    grad_weight_custom = weight.grad
    grad_values_custom = values.grad
    grad_bias_custom = bias.grad

    # Compare gradients
    assert torch.allclose(grad_input_naive, grad_input_custom, rtol=1e-3, atol=1e-3), \
        "Input gradients don't match"
    assert torch.allclose(grad_weight_naive, grad_weight_custom, rtol=1e-3, atol=1e-3), \
        "Weight gradients don't match"
    assert torch.allclose(grad_values_naive, grad_values_custom, rtol=1e-3, atol=1e-3), \
        "Values gradients don't match"
    assert torch.allclose(grad_bias_naive, grad_bias_custom, rtol=1e-3, atol=1e-3), \
        "Bias gradients don't match"

if __name__ == "main":
    pytest.main([__file__])
