"""Compatibility import. The research workbench is maintained in TWINS."""
from ._workbench import load_module

load_module('evaluation', alias=__name__)
