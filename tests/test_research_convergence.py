from copy import deepcopy
import pytest
from siesta_remag.discovery.config import Settings
from siesta_remag.research.convergence import assess


def report(k, e, f):
    return dict(format='REMAG-EXPERIMENT-RESULT/1',status='measured',source_sha256='a'*64,
        executable_sha256='b'*64,environment={},
        request=dict(source_sha256='a'*64,fixed_geometry=True,kmesh=[k]*3,basis_size='DZP',basis_energy_shift_ry=.02,
                     pseudopotentials={'Fe.psf':'c'*64},settings=Settings().to_dict()),
        observations={'soc-z':dict(total_energy_ev=0.,free_energy_ev=0.),
                      'soc-x':dict(total_energy_ev=e,free_energy_ev=f),'soc-y':dict(total_energy_ev=e,free_energy_ev=f)},
        stages=[dict(stage=s,accepted=True) for s in ['scalar','soc-z','soc-x','soc-y']],anisotropy=dict(mae_ev_cell=e))


def test_stability_requires_both_energies_and_two_refinements():
    rows=[report(16,.003,.003),report(20,.00301,.00302),report(24,.00302,.00303)]
    assert assess(rows,varied='kmesh',tolerance_ev=5e-5)['observed_stability_passed']
    assert not assess(rows[:2],varied='kmesh',tolerance_ev=5e-5)['observed_stability_passed']
    rows[-1]['observations']['soc-x']['free_energy_ev']+=.0002
    assert not assess(rows,varied='kmesh',tolerance_ev=5e-5)['observed_stability_passed']


def test_unrelated_changes_and_failed_runs_cannot_count_as_convergence():
    rows=[report(16,.003,.003),report(20,.00301,.00302)]
    rows[-1]['request']['settings']['smearing_k']=500
    with pytest.raises(ValueError,match='Unrelated'):assess(rows,varied='kmesh',tolerance_ev=5e-5)
    rows[-1]['request']['settings']['smearing_k']=300
    rows[-1]['status']='held_soc-x'
    with pytest.raises(ValueError):assess(rows,varied='kmesh',tolerance_ev=5e-5)


def test_changed_time_limits_are_not_a_physical_change_and_order_is_checked():
    rows=[report(16,.003,.003),report(20,.00301,.00302)]
    rows[-1]['request']['settings']['total_dft_seconds']=7200
    assert assess(rows,varied='kmesh',tolerance_ev=5e-5,successive_changes=1)['observed_stability_passed']
    with pytest.raises(ValueError):assess(rows[::-1],varied='kmesh',tolerance_ev=5e-5)
