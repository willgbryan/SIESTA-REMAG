import copy

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from siesta_remag.energy_only import publication_records, record_errors, sweep_record_errors


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
    reused = dict(record, epoch=2, compact_setup_bytes=0, device_allocation_bytes=0, reused_geometry=True)
    assert not orbital_errors(reused, dict(epoch=2), False, energy_only=True, reused_geometry=True)
    assert orbital_errors(reused, dict(epoch=2), False, energy_only=True)
    assert orbital_errors(record, dict(epoch=1), False, energy_only=True, reused_geometry=True)
    assert orbital_errors(dict(reused, generated=True), dict(epoch=2), False, energy_only=True, reused_geometry=True)


def sweep():
    record, cycles, output, text = publication()
    rows = [dict(record, epoch=i) for i in (4, 5, 6)]
    cycles = [dict(cycles[0], epoch=i) for i in (4, 5, 6)]
    return dict(schema=2, mode='energy_only', cycles=rows), cycles, [output]*3, [text]*3


def test_sweep_publications_cover_each_accepted_direction():
    report, cycles, outputs, texts = sweep()
    assert publication_records(report) == report['cycles']
    assert sweep_record_errors(report, cycles, outputs, texts, expected_directions=3) == []
    # The single-case runner must still reject a sweep, even if its last point passes.
    assert record_errors(report, cycles, outputs[-1], texts[-1])
    assert record_errors(report, [cycles[-1]], outputs[-1], texts[-1])
    partial = dict(report, cycles=report['cycles'][:2])
    assert sweep_record_errors(partial, cycles[:2], outputs[:2], texts[:2], expected_directions=3)
    for slot in range(4):
        args = copy.deepcopy((report, cycles, outputs, texts))
        if slot == 0:
            args[0]['cycles'].pop(0)
        else:
            args[slot].pop(0)
        assert sweep_record_errors(*args, expected_directions=3)


@pytest.mark.parametrize('defect', ['failed_earlier', 'unconverged_earlier', 'wrong_epoch', 'duplicate_epoch',
                                  'reordered', 'missing_energy', 'forces', 'missing_marker', 'nested'])
def test_last_sweep_success_cannot_erase_earlier_failure(defect):
    report, cycles, outputs, texts = copy.deepcopy(sweep())
    if defect == 'failed_earlier':report['cycles'][0]['status'] = 1
    elif defect == 'unconverged_earlier':cycles[0]['converged'] = False
    elif defect == 'wrong_epoch':report['cycles'][0]['epoch'] = 3
    elif defect == 'duplicate_epoch':cycles[0]['epoch'] = report['cycles'][0]['epoch'] = 5
    elif defect == 'reordered':cycles.reverse();report['cycles'].reverse()
    elif defect == 'missing_energy':outputs[0] = dict(outputs[0], free_energy_ev=None)
    elif defect == 'forces':report['cycles'][0]['forces_calculated'] = True
    elif defect == 'missing_marker':texts[0] = ''
    elif defect == 'nested':report['cycles'][0] = dict(schema=2, mode='energy_only', cycles=[report['cycles'][0]])
    assert sweep_record_errors(report, cycles, outputs, texts, expected_directions=3)


@pytest.mark.parametrize('report', [None, [], {}, dict(schema=True, mode='energy_only'),
                                 dict(schema=3, mode='energy_only'),
                                 dict(schema=2, mode='energy_only', cycles=[]),
                                 dict(schema=2, mode='energy_only', cycles=[None])])
def test_malformed_publication_is_not_an_empty_passing_sweep(report):
    assert publication_records(report) == []
    assert sweep_record_errors(report, [], [], [], expected_directions=3)
    assert record_errors(report, [], {}, '')


@pytest.fixture(scope='module')
def publication_writer(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler:pytest.skip('C++ compiler needed for publication writer')
    folder = tmp_path_factory.mktemp('energy_publication')
    source = folder/'check.cpp'
    source.write_text(r'''
#include "energy_only_publication.hpp"
#include <cassert>
#include <string>
int main(int argc,char** argv) {
    assert(argc==2);
    std::vector<remag::EnergyOnlyPublication> rows{{4,50,8,10}};
    if(std::string(argv[1])=="single"){remag::publish_energy_only(rows);return 0;}
    rows.push_back({5,50,8,10});rows.push_back({6,50,8,10});
    if(std::string(argv[1])=="sweep"){remag::publish_energy_only(rows);return 0;}
    // Failed publications must preserve the last complete receipt on disk.
    if(std::string(argv[1])=="invalid")rows[1].epoch=4;
    bool rejected=false;
    try{remag::publish_energy_only(rows);}catch(const std::exception&){rejected=true;}
    assert(rejected);
}
''')
    target = folder/'check'
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                             '-I'+str(root/'native'), str(source), '-o', str(target)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return target


def test_native_publication_preserves_every_epoch_and_last_good_file(publication_writer, tmp_path):
    def run(mode):
        subprocess.run([str(publication_writer), mode], cwd=tmp_path, check=True, capture_output=True, text=True)
    receipt = tmp_path/'remag-energy-only.json'
    run('single')
    assert json.loads(receipt.read_text()) == publication()[0]
    run('sweep')
    assert json.loads(receipt.read_text()) == sweep()[0]
    previous = receipt.read_bytes()
    run('invalid')
    assert receipt.read_bytes() == previous
    # A temporary-file open failure cannot truncate the published receipt.
    (tmp_path/'remag-energy-only.json.tmp').mkdir()
    run('io_failure')
    assert receipt.read_bytes() == previous
