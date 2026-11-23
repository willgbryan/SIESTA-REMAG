"""Compatibility import. The research workbench is maintained in TWINS."""
from ._workbench import load_module

if __name__ == '__main__':
    raise SystemExit(load_module('model_policy').main())

load_module('model_policy', alias=__name__)
