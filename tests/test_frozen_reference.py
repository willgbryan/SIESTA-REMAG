"""SOC-free experiments must not silently become ordinary SOC regressions."""
from copy import deepcopy
import pytest
from siesta_remag.frozen_reference import requested, evidence_errors, scf_soc_mode


def env():
    return dict(REMAG_FROZEN_SOC_FREE='1', REMAG_FROZEN_ROTATIONS='rotations.txt',
                REMAG_DEVICE_SCF='1', REMAG_GPU_INITIALIZATION='1', REMAG_ENERGY_ONLY='1')


def telemetry():
    return dict(gpu_device_cycle=dict(cycles=[dict(soc_free_reference=True, zero_soc_workspace_bytes=1024)]),
                frozen_potential_probe=dict(reference_soc_scale=0, target_soc_scale=1, kpoints=64,
                    generator_spectrum_max_abs_error_ry=1e-12, reference_control=dict(completed=True, status=0,
                        soc_scale=0, eigensystem_jobs=64, raw_spectrum='remag-frozen-reference.csv')))


def test_historical_default_and_explicit_experimental_mode():
    assert not requested({}) and requested(env())
    assert scf_soc_mode({}) == 'input'
    assert scf_soc_mode(dict(runtime_settings=env())) == 'soc_free_reference'
    assert evidence_errors({}, False) == []
    assert evidence_errors(telemetry(), True) == []


@pytest.mark.parametrize('key', ['REMAG_FROZEN_ROTATIONS', 'REMAG_DEVICE_SCF', 'REMAG_GPU_INITIALIZATION', 'REMAG_ENERGY_ONLY'])
def test_partial_configuration_fails_before_execution(key):
    e = env();e.pop(key)
    with pytest.raises(ValueError):requested(e)


@pytest.mark.parametrize('option', ['', 'true', '2', '-1'])
def test_invalid_switch(option):
    with pytest.raises(ValueError):requested(dict(REMAG_FROZEN_SOC_FREE=option))


@pytest.mark.parametrize('defect', ['binding', 'missing_control', 'partial_control', 'wrong_jobs', 'wrong_scale', 'spectrum', 'nonfinite'])
def test_missing_or_wrong_reference_cannot_pass(defect):
    t = deepcopy(telemetry());p = t['frozen_potential_probe']
    if defect == 'binding':t['gpu_device_cycle']['cycles'][0]['soc_free_reference'] = False
    elif defect == 'missing_control':p.pop('reference_control')
    elif defect == 'partial_control':p['reference_control']['completed'] = False
    elif defect == 'wrong_jobs':p['reference_control']['eigensystem_jobs'] = 63
    elif defect == 'wrong_scale':p['reference_soc_scale'] = 1
    else:p['generator_spectrum_max_abs_error_ry'] = float('nan') if defect == 'nonfinite' else 1e-6
    assert evidence_errors(t, True)
    assert evidence_errors(telemetry(), False)
