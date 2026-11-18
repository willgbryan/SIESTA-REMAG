"""Resolve historical workbench imports only when a caller requests them."""
from importlib import import_module
import sys


def load_module(name, *, alias=None):
    try:
        module = import_module(f"twins.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != "twins":
            raise
        raise ModuleNotFoundError(
            "This workbench operation moved to TWINS. Install the twins-workbench "
            "distribution from https://github.com/willgbryan/TWINS."
        ) from exc
    if alias is not None:
        sys.modules[alias] = module
    return module
