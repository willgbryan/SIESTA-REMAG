import numpy as np
import pytest

from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save
from siesta_remag.research.inspection import inspect_electronic, projected_dos, compare_electronic


def spectrum_fixture(root, shift=0):
    root.mkdir()
    np.savez(root/'spectrum.npz', eigenvalues_ev=np.array([[-.1,.2],[.05,.3]])+shift,
        k_fractional=[[0,0,0],[.5,0,0]], k_weights=[.25,.75], occupations=[[1,0],[.5,0]],
        orbital_mulliken=np.array([[[1.1,.2],[-.1,.8]],[[.8,-.2],[.2,1.2]]]),
        spin_pauli=np.array([[[0,0,1],[0,0,-1]],[[1,0,0],[-1,0,0]]]))
    p=root/'electronic.json'
    save(p,dict(format='REMAG-ELECTRONIC/1', hsx_sha256='a'*64, artifact='spectrum.npz',
        artifact_sha256=sha256(root/'spectrum.npz'), native_comparison_passed=True, native_max_abs_error_ev=.0001,
        fermi_ev=0, temperature_ev=.025, spin_components=8, collinear_spin=None,
        orbitals=[dict(index=0,atom_index=0,l=2),dict(index=1,atom_index=1,l=2)],
        projection='Mulliken', spin_frame='Cartesian', occupations_model='Fermi-Dirac reconstruction'))
    return p


def test_bound_selection_and_signed_population(tmp_path):
    p=spectrum_fixture(tmp_path/'a')
    r=inspect_electronic(p,dict(atom_indices=[1],k_indices=[0],energy_window_ev=[-.2,.4],limit=1))
    assert r['matching_states']==2 and r['omitted_states']==1
    assert r['states'][0]['selected_mulliken_weight']==pytest.approx(-.1)
    assert r['states'][0]['spin_pauli']==[0,0,1]
    assert r['selected_k_weight']==.25
    assert r['metadata_sha256']==sha256(p)


def test_dos_preserves_signed_weights_and_sampling_normalization(tmp_path):
    p=spectrum_fixture(tmp_path/'a')
    r=projected_dos(p,dict(atom_indices=[1],k_indices=[0],energy_window_ev=[-.1,0]),sigma_ev=.01,points=2)
    assert r['density_states_per_ev'][0]==pytest.approx(-.1*.25/(.01*np.sqrt(2*np.pi)))
    assert r['selected_k_weight']==.25


def test_comparison_has_common_grid_and_no_state_tracking_claim(tmp_path):
    a=spectrum_fixture(tmp_path/'a'); b=spectrum_fixture(tmp_path/'b',shift=.01)
    r=compare_electronic(a,b,points=11)
    assert len(r['delta_dos'])==11
    assert max(abs(x) for x in r['delta_dos'])>0
    assert not r['mechanism_verified']
    m=load(b);m['orbitals'][1]['l']=1;save(b,m)
    with pytest.raises(ValueError,match='metadata differs'):compare_electronic(a,b)


@pytest.mark.parametrize('selection', [dict(atom_indices=[4]),dict(l=[True]),dict(k_indices=[0,0]),
    dict(orbital_indices=[-1]),dict(limit=0),dict(energy_window_ev=[0,float('nan')]),dict(unknown=1)])
def test_invalid_selection_rejected(tmp_path,selection):
    p=spectrum_fixture(tmp_path/'a')
    with pytest.raises(ValueError):inspect_electronic(p,selection)


def test_unchecked_and_tampered_spectra_are_not_served(tmp_path):
    p=spectrum_fixture(tmp_path/'a');m=load(p);m['native_comparison_passed']=False;save(p,m)
    with pytest.raises(ValueError,match='comparison'):inspect_electronic(p)
    m['native_comparison_passed']=True;save(p,m)
    (p.parent/'spectrum.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='binding'):inspect_electronic(p)
