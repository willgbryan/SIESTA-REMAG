"""Compatibility import. The research workbench is maintained in TWINS."""
from ._workbench import load_module

load_module('discovery_scores', alias=__name__)
