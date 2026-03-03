from __future__ import annotations

import sys
import types


def _install_torchvision_functional_tensor_alias() -> None:
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    try:
        import torchvision.transforms._functional_tensor as functional_tensor  # type: ignore[attr-defined]
    except Exception:
        return

    shim = types.ModuleType(module_name)
    for attribute in dir(functional_tensor):
        if attribute.startswith("__"):
            continue
        setattr(shim, attribute, getattr(functional_tensor, attribute))
    sys.modules[module_name] = shim


_install_torchvision_functional_tensor_alias()
