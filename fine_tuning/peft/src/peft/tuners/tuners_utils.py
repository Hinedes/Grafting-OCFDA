class BaseTuner:
    pass


class BaseTunerLayer:
    @property
    def active_adapters(self):
        active_adapter = getattr(self, "_active_adapter", None)
        if active_adapter is None:
            return []
        if isinstance(active_adapter, str):
            return [active_adapter]
        return list(active_adapter)

    @property
    def disable_adapters(self):
        return getattr(self, "_disable_adapters", False)

    @disable_adapters.setter
    def disable_adapters(self, value):
        self._disable_adapters = value

    @property
    def merged(self):
        return len(getattr(self, "merged_adapters", [])) > 0

    def get_base_layer(self):
        base_layer = self
        while hasattr(base_layer, "base_layer"):
            base_layer = base_layer.base_layer
        return base_layer

    def set_adapter(self, adapter_names):
        if isinstance(adapter_names, str):
            self._active_adapter = adapter_names
        elif adapter_names:
            self._active_adapter = list(adapter_names)[0]
        else:
            self._active_adapter = None


def check_target_module_exists(*args, **kwargs):
    raise NotImplementedError("check_target_module_exists is not implemented in the local PEFT compatibility shim.")


def onload_layer(*args, **kwargs):
    raise NotImplementedError("onload_layer is not implemented in the local PEFT compatibility shim.")
