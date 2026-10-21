# -*- coding: utf-8 -*-
"""Registry of ovulation-phase classification models."""
from __future__ import annotations

from . import sota_model_dl as dl
from . import sota_model_ml as ml

__all__ = [
    "ml",
    "dl",
    "list_all_models",
    "build",
]


def list_all_models() -> dict[str, list[str]]:
    return {"ml": ml.list_models(), "dl": dl.list_models()}


def build(group: str, name: str, **kwargs):
    """
    group: 'ml' | 'dl'
    name:  one of list_all_models()[group]
    """
    g = group.lower().strip()
    if g == "ml":
        return ml.build_model(name, **kwargs)
    if g == "dl":
        return dl.build_model(name, **kwargs)
    raise KeyError(f"group must be 'ml' or 'dl', got {group!r}")
