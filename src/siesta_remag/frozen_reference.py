"""Guard the experimental SOC-free SCF reference used by frozen SOC probes."""
import math


def requested(env):
    option = env.get('REMAG_FROZEN_SOC_FREE', '0')
    if option not in ('0', '1'):
        raise ValueError('REMAG_FROZEN_SOC_FREE must be 0 or 1')
    enabled = option == '1'
    if enabled and (not env.get('REMAG_FROZEN_ROTATIONS') or any(
            env.get(key) != '1' for key in ('REMAG_DEVICE_SCF', 'REMAG_GPU_INITIALIZATION', 'REMAG_ENERGY_ONLY'))):
        raise ValueError('SOC-free reference requires frozen rotations, GPU initialization and energy-only device SCF')
    return enabled


def scf_soc_mode(result):
    # Historical results have no switch and retain the SOC treatment of the FDF.
    return ('soc_free_reference' if result.get('runtime_settings', {}).get('REMAG_FROZEN_SOC_FREE') == '1'
            else 'input')


def evidence_errors(telemetry, enabled):
    rows = telemetry.get('gpu_device_cycle', {}).get('cycles', [])
    report = telemetry.get('frozen_potential_probe', {})
    errors = []
    if any(r.get('soc_free_reference', False) is not enabled for r in rows):
        errors.append('SCF reference SOC treatment differs from the request')
    if report and (report.get('reference_soc_scale', 1) != int(not enabled)
                   or report.get('target_soc_scale', 1) != 1):
        errors.append('Frozen reference/target SOC treatment differs from the request')
    if not enabled:
        return errors
    if not rows or any(not r.get('zero_soc_workspace_bytes', 0) > 0 for r in rows):
        errors.append('SOC-free SCF binding evidence is missing')
    control = report.get('reference_control') or {}
    if (not report or control.get('completed') is not True or control.get('status') != 0
            or control.get('soc_scale') != 0 or not report.get('kpoints', 0) > 0
            or control.get('eigensystem_jobs') != report.get('kpoints')
            or control.get('raw_spectrum') != 'remag-frozen-reference.csv'):
        errors.append('SOC-free reference identity solve is missing or failed')
    error = report.get('generator_spectrum_max_abs_error_ry')
    if not isinstance(error, (int, float)) or not math.isfinite(error) or not 0 <= error*13.605693122994 <= 1e-7:
        errors.append('SOC-free generator spectrum identity failed')
    return errors
