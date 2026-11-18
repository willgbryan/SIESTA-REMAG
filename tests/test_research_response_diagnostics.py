"""Analytic diagnostic responses on synthetic files; no native solver calls."""
import pytest
from siesta_remag.research.response_diagnostics import validate_diagnostic

def definitions():
    base=dict(stage='soc-z',spin=0,k_indices=[0,5,13])
    return dict(coupling=dict(kind='coupling',metric='frobenius',groups=[{'site':0,'l':0},{'site':1,'l':0}],**base),
        singular=dict(kind='coupling',metric='largest_singular_value',groups=[{'site':0,'l':0},{'site':1,'l':0}],**base),
        bands=dict(kind='band_sum',reference='absolute',**base),
        relative_bands=dict(kind='band_sum',reference='own_fermi',**base),
        orbital=dict(kind='orbital_moment',stage='soc-z',selection={'sites':[0],'l':[1]},component=2))

@pytest.mark.parametrize('change',[{'k_indices':[True]}, {'k_indices':[0]*2}, {'spin':1}, {'metric':'made_up'},
    {'groups':[{'site':0,'l':0}]*2}, {'groups':[{'site':0,'l':5},{'site':1,'l':0}]}, {'extra':0}])
def test_diagnostic_targets_require_explicit_valid_selection(change):
    with pytest.raises(ValueError):validate_diagnostic(definitions()['coupling']|change)
