from __future__ import annotations

import numpy as np
import torch


_PATCHED = False


def _patch_numpy_legacy_aliases() -> None:
    alias_map = {
        "bool": "bool_",
        "int": "int_",
        "float": "float64",
        "complex": "complex128",
        "object": "object_",
        "str": "str_",
        "long": "int_",
        "unicode": "str_",
    }
    for alias, replacement in alias_map.items():
        if alias in np.__dict__:
            continue
        replacement_value = getattr(np, replacement, None)
        if replacement_value is not None:
            setattr(np, alias, replacement_value)


def _patch_torch_load_weights_only() -> None:
    if getattr(torch.load, "_bert4rec_v4b_patched", False):
        return

    original_torch_load = torch.load

    def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
        kwargs.setdefault("weights_only", False)
        if pickle_module is not None:
            return original_torch_load(
                f,
                map_location=map_location,
                pickle_module=pickle_module,
                **kwargs,
            )
        return original_torch_load(f, map_location=map_location, **kwargs)

    patched_torch_load._bert4rec_v4b_patched = True  # type: ignore[attr-defined]
    torch.load = patched_torch_load


def apply_runtime_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _patch_numpy_legacy_aliases()
    _patch_torch_load_weights_only()
    _PATCHED = True
