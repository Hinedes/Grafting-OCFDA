import torch
import torch.nn as nn
import math


class LoraLinear(nn.Module):
    def __init__(self,
                 base_layer,
                 sparse_rate: float,
                 r: int = 8,
                 dynamic_r: bool = False,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05):
        super().__init__()

        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.num_elements = self.weight.numel()

        in_features, out_features = self.weight.shape

        r_lora = r
        lora_params = (out_features + in_features) * r_lora

        if dynamic_r:
            assert 0.0 <= sparse_rate <= 1.0, "sparse_rate should be a ratio between 0 and 1"

            params_selected_for_lora = math.ceil(sparse_rate * self.weight.numel())

            r_lora = params_selected_for_lora // (out_features + in_features)
            lora_params = (out_features + in_features) * r_lora

        print("lora_params = ", lora_params)

        if getattr(base_layer, "state", None) is not None:
            self.state = base_layer.state

        if r > 0:
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

        after_A = self.lora_A(self.lora_dropout(input.to(self.lora_A.weight.dtype)))
        after_B = self.lora_B(after_A)

        result = after_B * self.scaling

        return result.to(previous_dtype)


def get_custom_lora_model(model,
                          target_modules_list,
                          sparse_rate: float,
                          r: int = 0.5,
                          dynamic_r: bool = False,
                          lora_alpha: int = 16,
                          lora_dropout: float = 0.05,
                          tokenizer=None,
                          exception=None):
    if exception is None:
        exception = []

    def _get_submodules(key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(parent_module, child_name, old_module):
        new_module = LoraLinear(old_module,
                                sparse_rate=sparse_rate,
                                r=r,
                                dynamic_r=dynamic_r,
                                lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout)
        setattr(parent_module, child_name, new_module)

    for module_name, _ in model.named_modules():
        if not any(module_name.endswith(target_key) for target_key in target_modules_list):
            continue

        parent, target, target_name = _get_submodules(module_name)
        _replace_module(parent, target_name, target)

    for name, p in model.named_parameters():
        if not ("lora" in name or any([item in name for item in exception])):
            p.requires_grad_(False)

    return model


def get_custom_lora_model_state_dict(model, state_dict=None):
    if state_dict is None:
        state_dict = model.state_dict()
    return {k: state_dict[k] for k in state_dict if "lora" in k}
