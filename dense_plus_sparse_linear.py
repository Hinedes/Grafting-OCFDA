import json
import os

import torch
import torch.nn as nn

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

        dense_plus_sparse = weight.view(-1).scatter_add(0, indices.to(torch.int64), values.to(weight.dtype))
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
                grad_values = grad_matrix.view(-1).gather(0, indices.to(torch.int64)).to(values.dtype)

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


def resolve_safetensors_weight_files(checkpoint_path: str) -> dict:
    if not checkpoint_path:
        raise ValueError("full_ft_checkpoint is required for full-delta sparse masks.")
    if os.path.isfile(checkpoint_path):
        return {"__single_file__": checkpoint_path}
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Full fine-tuned checkpoint not found: {checkpoint_path}")

    single_file = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(single_file):
        return {"__single_file__": single_file}

    index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index = json.load(f)
        return {
            weight_name: os.path.join(checkpoint_path, shard_name)
            for weight_name, shard_name in index.get("weight_map", {}).items()
        }

    raise FileNotFoundError(
        f"Could not find model.safetensors or model.safetensors.index.json in {checkpoint_path}"
    )


def load_safetensors_weight(weight_files: dict, weight_name: str) -> torch.Tensor:
    from safetensors import safe_open

    if "__single_file__" in weight_files:
        shard_path = weight_files["__single_file__"]
    else:
        shard_path = weight_files.get(weight_name)
        if shard_path is None:
            raise KeyError(f"Weight {weight_name} is not present in the full fine-tuned checkpoint index.")
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        if weight_name not in f.keys():
            raise KeyError(f"Weight {weight_name} is not present in {shard_path}.")
        return f.get_tensor(weight_name)


@torch.no_grad()
def prepare_full_delta_mask(model, target_modules_list, sparse_rate: float, full_ft_checkpoint: str) -> None:
    weight_files = resolve_safetensors_weight_files(full_ft_checkpoint)
    prepared_layers = 0
    total_indices = 0

    for module_name, module in model.named_modules():
        if not any(module_name.endswith(target_key) for target_key in target_modules_list):
            continue
        if not hasattr(module, "weight"):
            continue

        full_weight_name = f"{module_name}.weight"
        full_weight = load_safetensors_weight(weight_files, full_weight_name)
        base_weight = module.weight.detach().cpu()
        if full_weight.shape != base_weight.shape:
            raise ValueError(
                f"Shape mismatch for {full_weight_name}: base={tuple(base_weight.shape)}, "
                f"full_ft={tuple(full_weight.shape)}"
            )

        train_num = min(int(sparse_rate * module.weight.numel()) + 1, module.weight.numel())
        delta_metric = (full_weight.float() - base_weight.float()).abs().view(-1)
        indices = torch.topk(delta_metric, k=train_num, largest=True).indices.to(dtype=torch.int32)
        module.weight.full_delta_topk_indices = indices
        prepared_layers += 1
        total_indices += int(indices.numel())

        selected = delta_metric[indices.to(torch.int64)]
        print(
            "full-delta mask",
            full_weight_name,
            "train_num",
            train_num,
            "delta_min",
            float(selected.min().item()) if selected.numel() else 0.0,
            "delta_mean",
            float(selected.mean().item()) if selected.numel() else 0.0,
            "delta_max",
            float(selected.max().item()) if selected.numel() else 0.0,
        )

    if prepared_layers == 0:
        raise RuntimeError("No target modules were found when preparing full-delta masks.")
    print(
        "Prepared full-delta sparse masks from",
        full_ft_checkpoint,
        "layers:",
        prepared_layers,
        "sparse entries:",
        total_indices,
    )


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
        full_ft_checkpoint=None,
):
    if indices_choice in {"super", "super-bottom"}:
        assert tokenizer is not None, "`Super` option requires tokenizer to determine outliers indices."
        prepare_super_mask(
            model,
            tokenizer,
            dev=model.device,
            sparse_rate=sparse_rate,
            nsamples=calibration_nsamples,
            seed=calibration_seed,
            calibration_data=calibration_data,
            metric_order="bottom" if indices_choice == "super-bottom" else "top",
        )
    elif indices_choice == "full-delta":
        prepare_full_delta_mask(
            model=model,
            target_modules_list=target_modules_list,
            sparse_rate=sparse_rate,
            full_ft_checkpoint=full_ft_checkpoint,
        )

    def _get_submodules(key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    if indices_choice not in {"random", "super", "super-bottom", "full-delta"}:
        raise ValueError("indices_choice must be 'random', 'super', 'super-bottom', or 'full-delta'.")

    replaced_modules = 0
    total_indices = 0
    total_unique_indices = 0

    def _replace_module(parent_module, child_name, old_module):
        nonlocal replaced_modules, total_indices, total_unique_indices
        if indices_choice in {"super", "super-bottom"}:
            attr_name = "wanda_bottomk_indices" if indices_choice == "super-bottom" else "wanda_topk_indices"
            indices = getattr(old_module.weight, attr_name, None)
            if indices is None:
                raise RuntimeError(f"Wanda indices were not prepared for a Super sparse layer ({attr_name}).")
        elif indices_choice == "full-delta":
            indices = getattr(old_module.weight, "full_delta_topk_indices", None)
            if indices is None:
                raise RuntimeError("Full-delta indices were not prepared for a Super sparse layer.")
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
        {
            "super": "wanda-top",
            "super-bottom": "wanda-bottom",
            "random": "random",
            "full-delta": "full-ft-delta-top",
        }[indices_choice],
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
