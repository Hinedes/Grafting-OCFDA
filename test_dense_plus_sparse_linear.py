import torch
import pytest
import gc
import os

from dense_plus_sparse_linear import dense_plus_sparse_linear, SparseDenseLinear, get_dense_plus_sparse_model


def prepare_test_data(input_dtype, weight_dtype, values_dtype, index_dtype, input_shape, 
                       is_bias, device):
    # Prepare dimensions
    in_features = input_shape[-1]
    out_features = 20
    batch_dims = input_shape[:-1]
    
    # Create inputs
    input = torch.randn(*input_shape, dtype=input_dtype, requires_grad=True, device=device)
    weight = torch.randn(out_features, in_features, dtype=weight_dtype, requires_grad=True, device=device)
    
    # Create sparse updates
    num_updates = 50
    indices = torch.randint(0, weight.numel(), (num_updates,), dtype=index_dtype, device=device)
    values = torch.randn(num_updates, dtype=values_dtype, requires_grad=True, device=device)
    if is_bias:
        bias = torch.randn(out_features, dtype=weight_dtype, requires_grad=True, device=device)
    else:
        bias = None
    return input, weight, indices, values, bias


def get_tensor_info(include_grad=True, impl="custom"):
    total_size = 0
    tensors = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensors.append(obj)
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensors.append(obj.data)
        except:
            pass

    for tensor in tensors:
        if tensor.device != torch.device("cpu"):
            print('Type:', type(tensor), 'Size:', tensor.size(), 'Device:', tensor.device, "Dtype:", tensor.dtype, 'Resuires_grad:', tensor.requires_grad, 'Grad fn:', tensor.grad_fn, 'Memory (MB):', tensor.element_size() * tensor.nelement() / 1024**2)

            total_size += tensor.element_size() * tensor.nelement()
    
    total_number_of_tensors = len([x for x in tensors if x.device != torch.device("cpu")])
    print(f'Implementation: {impl}, Total number of tensors: {total_number_of_tensors}, Total Memory Used: {total_size/1024**2:.2f} MB')
    return total_number_of_tensors


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("values_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("input_shape", [(32, 10), (8, 16, 10)])
@pytest.mark.parametrize("is_bias", [False])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("CUBLAS_WORKSPACE_CONFIG", [":4096:8", ":16:8"])
def test_custom_linear(input_dtype, weight_dtype, values_dtype, index_dtype, input_shape, 
                       is_bias, device, CUBLAS_WORKSPACE_CONFIG):
    torch.manual_seed(42)
    autocast_enabled = False
    if not (input_dtype == weight_dtype == values_dtype):
        autocast_enabled = True
    if device != torch.device("cpu"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=CUBLAS_WORKSPACE_CONFIG
        torch.use_deterministic_algorithms(True)
    
    input, weight, indices, values, bias = prepare_test_data(
        input_dtype, weight_dtype, values_dtype, index_dtype, input_shape, is_bias, device
    )

    def run_naive():
        indices_unraveled = torch.stack(torch.unravel_index(indices, weight.size())).to(torch.int64)
        with torch.autocast(device_type="cpu" if device == "cpu" else "cuda:0", dtype=torch.bfloat16, enabled=autocast_enabled):
            sparse = weight + torch.sparse_coo_tensor(indices=indices_unraveled, values=values, size=weight.size())
            return torch.nn.functional.linear(input, sparse, bias)

    def run_custom():
        with torch.autocast(device_type="cpu" if device == "cpu" else "cuda:0", dtype=torch.bfloat16, enabled=autocast_enabled):
            return dense_plus_sparse_linear.apply(input, weight, indices, values, bias)

    # Forward pass comparison
    output_naive = run_naive()

    # Backward pass comparison
    grad_output = torch.randn_like(output_naive)
    
    # Naive backward
    output_naive.backward(grad_output)
    grad_input_naive = input.grad.clone()
    grad_weight_naive = weight.grad.clone()
    grad_values_naive = values.grad.clone()
    if bias is not None:
        grad_bias_naive = bias.grad.clone()

    # Reset grads
    input.grad = None
    weight.grad = None
    values.grad = None
    if bias is not None:
        bias.grad = None

    output_custom = run_custom()
    assert output_naive.dtype == output_custom.dtype, \
        "Forward dtypes don't match"
    assert torch.allclose(output_naive, output_custom, rtol=1e-3, atol=1e-3), \
        "Forward passes don't match"
    
    # Custom backward
    output_custom.backward(grad_output)
    grad_input_custom = input.grad
    grad_weight_custom = weight.grad
    grad_values_custom = values.grad
    if bias is not None:
        grad_bias_custom = bias.grad

    # Compare gradients
    assert torch.allclose(grad_input_naive, grad_input_custom, rtol=1e-3, atol=1e-3), \
        "Input gradients don't match"
    assert torch.allclose(grad_weight_naive, grad_weight_custom, rtol=1e-3, atol=1e-3), \
        "Weight gradients don't match"
    assert torch.allclose(grad_values_naive, grad_values_custom, rtol=1e-3, atol=1e-3), \
        "Values gradients don't match"
    if bias is not None:
        assert torch.allclose(grad_bias_naive, grad_bias_custom, rtol=1e-3, atol=1e-3), \
            "Bias gradients don't match"


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("values_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("input_shape", [(32, 10), (8, 16, 10)])
@pytest.mark.parametrize("is_bias", [False, True])
@pytest.mark.parametrize("device", ["cuda:0"])
def test_memory(input_dtype, weight_dtype, values_dtype, index_dtype, input_shape, 
                       is_bias, device):
    torch.manual_seed(42)
    autocast_enabled = False
    if not (input_dtype == weight_dtype == values_dtype):
        autocast_enabled = True

    input, weight, indices, values, bias = prepare_test_data(
        input_dtype, weight_dtype, values_dtype, index_dtype, input_shape, is_bias, device
    )

    def run_custom():
        with torch.autocast(device_type="cpu" if device == "cpu" else "cuda:0", dtype=torch.bfloat16, enabled=autocast_enabled):
            return dense_plus_sparse_linear.apply(input, weight, indices, values, bias)

    output_custom = run_custom()
    
    # Custom backward
    total_number_of_tensors = get_tensor_info()
    output_custom.norm().backward()
    assert total_number_of_tensors == 5 if not is_bias else 6


if __name__ == "main":
    pytest.main([__file__])
