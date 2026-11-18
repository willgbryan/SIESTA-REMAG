"""Compatibility import. The research workbench is maintained in TWINS."""
from ._workbench import load_module

load_module('workspace_actions', alias=__name__)
