# rosa_adapter.py  (put it anywhere on your PYTHONPATH)

from types import MethodType
from typing import Dict, List

import torch
import torch.nn as nn
try:
    from .rosa.layer import Linear as RosaLinear
    from .rosa.layer import RosaLayer
    from .rosa.config import RosaConfig
except ImportError:
    from rosa.layer import Linear as RosaLinear
    from rosa.layer import RosaLayer
    from rosa.config import RosaConfig


def _set_spa_masks(self, masks: Dict[str, torch.Tensor]):
    for name, module in self.named_modules():
        if isinstance(module, RosaLayer) and name in masks:
            module.set_spa_mask(masks[name])
    self.spa_activated = True


def get_rosa_model(
    model: nn.Module,
    target_modules: List[str],
    r: int = 8,
    d: float = 0.003,
    alpha: int = 16,
    dropout: float = 0.05,
    impl: str = "sp_add",
    schedule: str = "wl0",
    spa_num_grads: int = 1,
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
        rosa_dtype='fp32',
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

    return model


def get_rosa_model_state_dict(model, state_dict=None):
    if state_dict is None:
        state_dict = model.state_dict()
    return {
        k: v
        for k, v in state_dict.items()
        if any(
            substr in k
            for substr in [
                "spa_mask", "values", "indices",  # sparse part
                "lora_A", "lora_B",               # low-rank part
                "scaling"                         # optional: alpha scaling
            ]
        )
    }
