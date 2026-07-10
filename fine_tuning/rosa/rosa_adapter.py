# rosa_adapter.py  (put it anywhere on your PYTHONPATH)

from types import MethodType
from typing import Dict, List, Optional

import torch
import torch.nn as nn

try:
    from .rosa.config import RosaConfig
    from .rosa.layer import Linear as RosaLinear
    from .rosa.layer import RosaLayer
except ImportError:
    from rosa.config import RosaConfig
    from rosa.layer import Linear as RosaLinear
    from rosa.layer import RosaLayer


def _find_mask(masks: Dict[str, torch.Tensor], target_key: str) -> Optional[torch.Tensor]:
    for key, mask in masks.items():
        if target_key in key or key in target_key:
            return mask
    return None


def _set_spa_masks(self, masks: Dict[str, torch.Tensor]):
    missing = []
    for name, module in self.named_modules():
        if not isinstance(module, RosaLayer):
            continue
        mask = _find_mask(masks, name)
        if mask is None:
            missing.append(name)
            continue
        module.set_spa_mask(mask)
    if missing:
        raise KeyError(f"Missing RoSA sparse masks for {len(missing)} layers, first missing layer: {missing[0]}")
    print("spa masks set.")
    self.spa_activated = True


def _merge_and_unload(self, progressbar: bool = False, safe_merge: bool = False, adapter_names=None):
    del progressbar
    rosa_module_names = [
        name
        for name, module in self.named_modules()
        if name and isinstance(module, RosaLayer)
    ]
    for name in rosa_module_names:
        module = self.get_submodule(name)
        module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
        base_layer = module.get_base_layer()
        pieces = name.split(".")
        parent_name = ".".join(pieces[:-1])
        parent = self.get_submodule(parent_name) if parent_name else self
        setattr(parent, pieces[-1], base_layer)
    return self


def get_rosa_model(
    model: nn.Module,
    target_modules: List[str],
    r: int = 8,
    d: float = 0.003,
    alpha: int = 16,
    dropout: float = 0.05,
    impl: str = "sp_add",
    schedule: str = "wl64",
    spa_num_grads: int = 1,
    rosa_dtype: str = "bf16",
):
    """
    Recursively replaces every linear layer whose name ends with one of
    `target_modules` by a RosaLinear wrapper and freezes the dense weight.
    """
    cfg = RosaConfig(
        r=r,
        d=d,
        lora_alpha=alpha,
        lora_dropout=dropout,
        impl=impl,
        target_modules=target_modules,
        rosa_dtype=rosa_dtype,
        schedule=schedule,
        spa_num_grads=spa_num_grads,
    )

    def _replace(parent, child_name, old):
        new = RosaLinear(
            old,
            adapter_name="default",
            r=cfg.r, d=cfg.d,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            impl=cfg.impl,
            spa_store_transpose=cfg.spa_store_transpose,
            rosa_dtype=cfg.rosa_dtype,
            init_lora_weights=True,
            use_rslora=False,
        )
        setattr(parent, child_name, new)

    for name, module in list(model.named_modules()):
        if not any(name.endswith(t) for t in target_modules):
            continue
        parent = model.get_submodule(".".join(name.split(".")[:-1]))
        child_name = name.split(".")[-1]
        _replace(parent, child_name, module)

    # freeze everything except the new adapter params
    for n, p in model.named_parameters():
        p.requires_grad = ("rosa_" in n)

    # Minimal compatibility surface used by the vendored RosaScheduler.
    model.peft_config = {"default": cfg}
    model.spa_activated = False
    model.set_spa_masks = MethodType(_set_spa_masks, model)
    model.merge_and_unload = MethodType(_merge_and_unload, model)

    return model


def get_rosa_model_state_dict(model, state_dict=None):
    if state_dict is None:
        state_dict = model.state_dict()
    return {key: value for key, value in state_dict.items() if "rosa_" in key}
