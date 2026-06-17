import torch
import torch.nn as nn
import math

from dense_plus_sparse_linear import DensePlusSparseLinear, random_sparse_indices
from src.mask import prepare_super_mask


class SparseDenseLoraLinear(nn.Module):
    def __init__(self,
                 base_layer,
                 sparse_rate: float,
                 lora_params_ratio: float = 0.5,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 indices=None):
        super().__init__()
        assert 0.0 <= sparse_rate <= 1.0, "sparse_rate should be a ratio between 0 and 1"
        assert 0.0 <= lora_params_ratio <= 1.0, "lora_params_ratio should be a ratio between 0 and 1"

        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.num_elements = self.weight.numel()

        in_features, out_features = self.weight.shape

        #params_selected_for_lora = math.ceil(lora_params_ratio * sparse_rate * self.weight.numel())
        #r_lora = params_selected_for_lora // (out_features + in_features)
        r_lora = math.ceil(lora_params_ratio * sparse_rate * self.weight.numel() / (out_features + in_features))
        lora_params = (out_features + in_features) * r_lora

        #super_params = (out_features + in_features) * r_super
        super_params = int(sparse_rate * self.weight.numel()) - lora_params

        print("lora_params = ", lora_params)
        print("super_params = ", super_params)

        if getattr(base_layer, "state", None) is not None:
            self.state = base_layer.state

        if indices is None:
            indices = random_sparse_indices(self.num_elements, super_params, self.weight.device)
        indices = indices.to(dtype=torch.int32, device=self.weight.device)[:super_params]

        self.values = nn.Parameter(torch.zeros(super_params, dtype=torch.float32, device=self.weight.device))
        self.register_buffer('indices', indices)

        if r_lora > 0:
            self.lora_A = nn.Linear(out_features, r_lora, bias=False, device=self.weight.device, dtype=torch.float32)
            self.lora_B = nn.Linear(r_lora, in_features, bias=False, device=self.weight.device, dtype=torch.float32)
            self.scaling = lora_alpha / r_lora
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, input):
        previous_dtype = input.dtype

        # result = torch.nn.functional.linear(input=input, weight=self.weight, bias=self.bias)
        result = DensePlusSparseLinear.apply(input, self.weight, self.indices, self.values, self.bias)

        if hasattr(self, "lora_A"):
            after_A = self.lora_A(self.lora_dropout(input.to(self.lora_A.weight.dtype)))
            after_B = self.lora_B(after_A)
            result += after_B * self.scaling

        return result.to(previous_dtype)


def get_dense_plus_sparse_plus_lora_model(model,
                                          target_modules_list,
                                          sparse_rate: float,
                                          lora_params_ratio: float = 0.5,
                                          lora_alpha: int = 16,
                                          lora_dropout: float = 0.05,
                                          indices_choice="random",
                                          tokenizer=None,
                                          exception=None,
                                          calibration_data="c4",
                                          calibration_nsamples=128,
                                          calibration_seed=228):
    if exception is None:
        exception = []
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

    def _get_submodules(key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    if indices_choice not in {"random", "super", "super-bottom"}:
        raise ValueError("indices_choice must be 'random', 'super', or 'super-bottom'.")

    replaced_modules = 0
    total_indices = 0
    total_unique_indices = 0

    def _replace_module(parent_module, child_name, old_module):
        nonlocal replaced_modules, total_indices, total_unique_indices
        if indices_choice in {"super", "super-bottom"}:
            attr_name = "wanda_bottomk_indices" if indices_choice == "super-bottom" else "wanda_topk_indices"
            indices = getattr(old_module.weight, attr_name, None)
            if indices is None:
                raise RuntimeError(f"Wanda indices were not prepared for a Supra sparse layer ({attr_name}).")
        else:
            indices = None
        new_module = SparseDenseLoraLinear(old_module,
                                           sparse_rate=sparse_rate,
                                           lora_params_ratio=lora_params_ratio,
                                           lora_alpha=lora_alpha,
                                           lora_dropout=lora_dropout,
                                           indices=indices)
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
        {"super": "wanda-top", "super-bottom": "wanda-bottom", "random": "random"}[indices_choice],
        "replaced modules:",
        replaced_modules,
        "sparse entries:",
        total_indices,
        "unique sparse entries:",
        total_unique_indices,
    )

    for name, p in model.named_parameters():
        if not ("values" in name or "lora" in name or any([item in name for item in exception])):
            p.requires_grad_(False)

    return model


def get_sparse_dense_lora_model_state_dict(model, state_dict=None):
    if state_dict is None:
        state_dict = model.state_dict()
    return {k: state_dict[k] for k in state_dict if "values" in k or "indices" in k or "lora" in k}
