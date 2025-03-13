import torch
import torch.nn as nn
import math

from dense_plus_sparse_linear import DensePlusSparseLinear
from src.mask import prepare_super_mask


class SparseDenseLoraLinear(nn.Module):
    def __init__(self,
                 base_layer,
                 sparse_rate: float,
                 r_lora: int = 4,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 indices=None):
        super().__init__()
        assert 0.0 <= sparse_rate <= 1.0, "sparse_rate should be a ratio between 0 and 1"
        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.num_elements = self.weight.numel()

        in_features, out_features = self.weight.shape

        #super_params = (out_features + in_features) * r_super

        lora_params = (out_features + in_features) * r_lora
        super_params = max(0, min(int(sparse_rate * self.weight.numel()) + 1, self.weight.numel()) - lora_params)

        print("lora_params = ", lora_params)
        print("super_params = ", super_params)

        if getattr(base_layer, "state", None) is not None:
            self.state = base_layer.state

        if indices is None:
            indices = torch.randint(0, self.num_elements, (super_params,))
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
                                          r_lora: int = 4,
                                          lora_alpha: int = 16,
                                          lora_dropout: float = 0.05,
                                          indices_choice="random",
                                          tokenizer=None,
                                          exception=None):
    if exception is None:
        exception = []
    if indices_choice == "super":
        assert tokenizer is not None, "`Super` option requires tokenizer to determine outliers indices."
        prepare_super_mask(model, tokenizer, dev=model.device, sparse_rate=sparse_rate)

    def _get_submodules(key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(parent_module, child_name, old_module):
        indices = getattr(old_module.weight, "wanda_topk_indices", None)
        new_module = SparseDenseLoraLinear(old_module,
                                           sparse_rate=sparse_rate,
                                           r_lora=r_lora,
                                           lora_alpha=lora_alpha,
                                           lora_dropout=lora_dropout,
                                           indices=indices)
        setattr(parent_module, child_name, new_module)

    for module_name, _ in model.named_modules():
        if not any(module_name.endswith(target_key) for target_key in target_modules_list):
            continue

        parent, target, target_name = _get_submodules(module_name)
        _replace_module(parent, target_name, target)

    for name, p in model.named_parameters():
        if not ("values" in name or "lora" in name or any([item in name for item in exception])):
            p.requires_grad_(False)

    return model


def get_sparse_dense_lora_model_state_dict(model, state_dict=None):
    if state_dict is None:
        state_dict = model.state_dict()
    return {k: state_dict[k] for k in state_dict if "values" in k or "indices" in k or "lora" in k}
