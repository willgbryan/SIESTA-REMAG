"""Compatibility import. The research workbench is maintained in TWINS."""
from ._workbench import load_module

load_module('candidate_metrics', alias=__name__)
