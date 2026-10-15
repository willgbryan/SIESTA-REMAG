"""Retained hcp Co constrained-anisotropy benchmark: REMAG-SIESTA against pw.x and experiment.

These tests read the committed records under benchmarks/spin-constraint-hcp-co/reference;
they run no electronic-structure code. See that directory's README for how the
records were produced and how to rerun the REMAG-SIESTA side.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]/'benchmarks/spin-constraint-hcp-co'
RY_UEV = 13.605693122994e6


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/f'{name}.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.fixture(scope='module')
def records():
    load = lambda name: json.loads((ROOT/'reference'/name).read_text())['runs']
    return dict(siesta=load('siesta.json'), qe=load('qe.json'), checks=load('siesta-checks.json'))


def test_inputs_regenerate_byte_for_byte(records, tmp_path):
    """Every retained run's input is what the committed generators write today."""
    siesta, qe = module('make_siesta'), module('make_qe')
    for r in records['siesta']:
        deck = siesta.deck(r['theta_deg'], r['phi_deg'], r['kmesh'], r['basis'], r['mesh_cutoff_ry'], r['temperature_k'],
                           r['energy_shift_ry'], r['constrained'], 0.01, r['penalty'], r.get('soc', 'onsite'),
                           r.get('occupation', 'FD'), r.get('xc', 'PBE'))
        assert hashlib.sha256(deck.encode()).hexdigest() == r['input_sha256'], r['directory']
    for r in records['qe']:
        deck = qe.deck(r['theta_deg'], r['kmesh'], r['ecutwfc_ry'], r['temperature_k'], r['lambda_ry'], '/root/pseudo',
                       occupation=r.get('occupation', 'smearing'), dft=r.get('dft'))
        assert hashlib.sha256(deck.encode()).hexdigest() == r['input_sha256'], r['directory']


def test_siesta_constraint_is_met(records):
    """Every REMAG-SIESTA run reached its target direction within the requested 0.01 degrees."""
    for r in records['siesta']+records['checks']:
        assert r['converged'] and r['deviation_deg'] <= 0.01, r['directory']
        assert r['max_torque_muB'] <= 1.e-3, r['directory']


def test_acceptance_checks_hold_for_retained_runs(records):
    """The README's acceptance checks, applied to the retained REMAG-SIESTA and pw.x records."""
    compare = module('compare')
    result = compare.summary(records['siesta'], records['qe'])
    outcome = compare.checks(result, records['siesta'], records['siesta'])
    assert len(outcome) == 6 and all(passed for _, passed, _ in outcome), outcome


def test_qe_penalty_limit(records):
    """pw.x energies at 45 degrees no longer depend on lambda, and the angle error falls as 1/lambda."""
    for mesh in ([6, 6, 3], [20, 20, 11]):
        runs = {r['lambda_ry']: r for r in records['qe'] if r['kmesh'] == mesh and r['theta_deg'] == 45.
                and r['temperature_k'] == 500. and r['ecutwfc_ry'] == 100.}
        assert {5., 25., 100.} <= set(runs), mesh
        assert abs(runs[25.]['free_energy_ry']-runs[100.]['free_energy_ry'])*RY_UEV < 0.5, mesh
        errors = [runs[l]['max_angle_error_deg'] for l in (5., 25., 100.)]
        assert errors[0] > errors[1] > errors[2] and errors[2] < 0.01, (mesh, errors)


def test_qe_cutoff_converged(records):
    """E(90)-E(0) at 6x6x3 changes by under 1 micro-eV per cell from 80 to 120 Ry."""
    def mae(ecut):
        e = {r['theta_deg']: r['free_energy_ry'] for r in records['qe'] if r['kmesh'] == [6, 6, 3]
             and r['ecutwfc_ry'] == ecut and r['lambda_ry'] == 25. and r['temperature_k'] == 500.}
        return (e[90.]-e[0.])*RY_UEV
    values = [mae(e) for e in (80., 100., 120.)]
    assert max(values)-min(values) < 1., values


def test_device_matches_host(records):
    """The GPU device SCF and the host path give the same constrained energies and moments."""
    runs = [r for r in records['checks'] if r['kmesh'] == [6, 6, 3] and r.get('xc', 'GGA-PBE') == 'GGA-PBE']
    pairs = {}
    for r in runs: pairs.setdefault(r['input_sha256'], {})[r['path']] = r
    complete = [p for p in pairs.values() if {'device', 'host'} <= set(p)]
    assert len(complete) >= 5
    for p in complete:
        assert abs(p['device']['free_energy_ev']-p['host']['free_energy_ev']) <= 2.e-8
        assert max(abs(a-b) for a, b in zip(p['device']['moment_muB'], p['host']['moment_muB'])) <= 1.e-4


def test_energy_torque_identity_holds_for_lda(records):
    """With LDA the holding field equals the energy slope; with SIESTA's noncollinear GGA it does not.

    dF/dtheta = -(b . e_theta)|M| for the constrained free energy. The GGA
    potential of the pinned libgridxc (ggaxc.F90, old_scheme) is not the exact
    derivative of the GGA energy, so the field carries an extra torque there.
    """
    def ratio(xc):
        r = {x['theta_deg']: x for x in records['checks'] if x.get('xc') == xc and x['path'] == 'host'
             and x.get('torque_tolerance_muB') == 1.e-6}
        slope = (r[50.]['free_energy_ev']-r[40.]['free_energy_ev'])/math.radians(10.)
        t = math.radians(45.); e_theta = (math.cos(t), 0., -math.sin(t))
        field = -sum(b*e for b, e in zip(r[45.]['field_ev_per_muB'], e_theta))*r[45.]['moment_magnitude_muB']
        return field/slope
    assert abs(ratio('LDA-PZ')-1.) < 0.02
    assert ratio('GGA-PBE') > 1.5


@pytest.fixture(scope='module')
def records_r2():
    load = lambda name: json.loads((ROOT/'reference'/name).read_text())['runs']
    return dict(siesta=load('siesta-r2.json'), qe=load('qe-r2.json'))


def test_r2_inputs_regenerate(records_r2):
    """Second-round inputs are what the generators write; torque runs tighten one line by hand."""
    siesta, qe = module('make_siesta'), module('make_qe')
    tol = {1.e-6: '1.e-6', 1.e-5: '1.e-5', None: '1.e-6'}
    for r in records_r2['siesta']:
        deck = siesta.deck(r['theta_deg'], r['phi_deg'], r['kmesh'], r['basis'], r['mesh_cutoff_ry'], r['temperature_k'],
                           r['energy_shift_ry'], r['constrained'], 0.01, r['penalty'], r.get('soc', 'onsite'),
                           r.get('occupation', 'FD'), r.get('xc', 'PBE'), r.get('nc_gga'),
                           tol[r.get('h_tolerance_ev')], r.get('pao_block'), r.get('mc_polar'))
        if r['torque_tolerance_muB'] == 1.e-6:
            deck = deck.replace('TorqueTolerance 1.e-3', 'TorqueTolerance 1.e-6')
        assert hashlib.sha256(deck.encode()).hexdigest() == r['input_sha256'], r['directory']
    for r in records_r2['qe']:
        deck = qe.deck(r['theta_deg'], r['kmesh'], r['ecutwfc_ry'], r['temperature_k'], r['lambda_ry'], '/root/pseudo',
                       occupation=r['occupation'], dft=r['dft'])
        assert hashlib.sha256(deck.encode()).hexdigest() == r['input_sha256'], r['directory']


def test_r2_acceptance_checks():
    """Off-site SOC with the local-frame GGA: SIESTA and pw.x agree on E(theta) and torques in PBE and LDA."""
    compare = module('compare_r2')
    outcome = compare.checks(compare.summary())
    assert len(outcome) == 7 and all(passed for _, passed, _ in outcome), outcome
