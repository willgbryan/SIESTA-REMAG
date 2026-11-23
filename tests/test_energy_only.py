import copy

import importlib.util
from pathlib import Path
import shutil

import pytest

from siesta_remag.energy_only import record_errors


def test_energy_only_patch_preserves_shared_export_initialization(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / 'upstream/siesta/Src'
    if not source.is_dir():
        pytest.skip('Fetch pinned upstream to exercise integration patch')
    (tmp_path / 'Src').mkdir()
    for name in ('final_H_f_stress.F', 'state_analysis.F', 'siesta_analysis.F90'):
        shutil.copyfile(source / name, tmp_path / 'Src' / name)
    spec = importlib.util.spec_from_file_location('patch_siesta', root / 'scripts/patch_siesta.py')
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    patch.patch_energy_only(tmp_path)
    text = (tmp_path / 'Src/final_H_f_stress.F').read_text()
    assert text.index('not_using_auxcell = (no_s == no_u)') < text.index('if(remag_energy_only_enabled()==1)then')
    assert text.count('not_using_auxcell =') == 1
    assert text.index('call ts_write_tshs') > text.index('! Determine whether anything should be saved')


def publication():
    record = dict(schema=1, mode='energy_only', epoch=4, status=0, iterations=10,
                  capacity=50, spins=8, forces_calculated=False, stress_calculated=False,
                  hamiltonian_required=False, hamiltonian_cycle_export_bytes=0)
    cycles = [dict(epoch=4, status=0, converged=True, iterations=10,
                   terminal=dict(capacity=50, spins=8))]
    output = dict(total_energy_ev=-12., free_energy_ev=-12.1, fermi_energy_ev=-2.,
                  forces_ev_ang=None, stress_ev_ang3=None)
    text = 'REMAG: energy-only finalization; forces and stress not calculated'
    return record, cycles, output, text


def test_energy_only_requires_accepted_state_and_absent_forces():
    args = publication()
    assert record_errors(*args) == []
    for field, value in [('status', 1), ('epoch', 3), ('iterations', 9),
                         ('forces_calculated', True), ('schema', 0)]:
        bad = copy.deepcopy(args)
        bad[0][field] = value
        assert record_errors(*bad)
    for key, value in [('forces_ev_ang', [[0., 0., 0.]]),
                       ('stress_ev_ang3', [[0., 0., 0.]]*3),
                       ('free_energy_ev', None), ('total_energy_ev', float('nan'))]:
        bad = copy.deepcopy(args)
        bad[2][key] = value
        assert record_errors(*bad)
    bad = copy.deepcopy(args)
    bad[1][0]['converged'] = False
    assert record_errors(*bad)
    assert record_errors(args[0], args[1]*2, args[2], args[3])
    assert record_errors(*args[:3], '')


def test_energy_only_keeps_prepared_orbitals_but_rejects_derivative_allocation():
    from siesta_remag.orbital_basis import record_errors as orbital_errors
    # Observed FePt preparation: compact orbital data exists, force derivatives do not.
    record = dict(epoch=1, shape=[2, 48, 8000, 945599, 8, 19440, 810, 65568, 7776],
                  compact_setup_bytes=9446368, device_allocation_bytes=9446368,
                  derivative_bytes=423628352, shadow_setup_bytes=0, shadow_export_bytes=0,
                  status=-1, consumed=False, passed=False, generated=False,
                  comparison_performed=False, host_derivatives_authoritative=False)
    assert not orbital_errors(record, dict(epoch=1), False, energy_only=True)
    for key, value in [('device_allocation_bytes', 433074720), ('generated', True),
                       ('consumed', True), ('epoch', 2)]:
        changed = dict(record, **{key: value})
        assert orbital_errors(changed, dict(epoch=1), False, energy_only=True)
    assert orbital_errors(record, dict(epoch=1), False)
