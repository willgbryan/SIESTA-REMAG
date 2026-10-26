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


def direction_checks(report, directions, *, moment_tolerance_degrees=1.):
    """Match frozen points to requested axes, independently of reported rotations.

    `directions` comes from the ordered target input manifests, starting with
    the SCF reference direction. A self-consistent quaternion/moment pair alone
    cannot establish that a result belongs to its requested crystallographic
    axis. Eigenvalues-only probes check rotation labels but have no moment proof.
    """
    import numpy as np
    from .angular_energy import unit_directions

    axes = unit_directions(directions)
    if not math.isfinite(moment_tolerance_degrees) or not 0 < moment_tolerance_degrees < 90:
        raise ValueError('A finite positive moment angle tolerance below 90 degrees is required')
    points = report.get('points', [])
    moments = report.get('spin_expectations')
    if (not isinstance(points, list) or len(points) != len(axes) or type(moments) is not bool):
        return dict(passed=False, reason='Frozen point coverage or spin mode differs', points=[])
    rows = []
    for index, (point, axis) in enumerate(zip(points, axes, strict=True)):
        row = dict(index=index, expected_direction=axis.tolist(), passed=False,
                   rotation_matches_target=False, moment_direction_checked=moments,
                   moment_error_degrees=None)
        try:
            if type(point.get('index')) is not int or point['index'] != index:
                raise ValueError('Frozen point order differs from target order')
            quaternion = point.get('quaternion')
            if np.iscomplexobj(quaternion):
                raise ValueError('Complex quaternion')
            q = np.asarray(quaternion, dtype=float)
            if q.shape != (4,) or not np.isfinite(q).all() or not np.any(q):
                raise ValueError('Missing or invalid quaternion')
            q = q/max(abs(q));q = q/np.linalg.norm(q)
            rotated = axes[0]+2*np.cross(q[1:], np.cross(q[1:], axes[0])+q[0]*axes[0])
            row['rotation_target_vector_error'] = float(np.linalg.norm(rotated-axis))
            row['rotation_matches_target'] = bool(row['rotation_target_vector_error'] <= 1e-10
                and (index != 0 or np.linalg.norm(q[1:]) <= 1e-12))
            row['passed'] = row['rotation_matches_target']
            if moments:
                moment = point.get('spin_muB')
                if np.iscomplexobj(moment):
                    raise ValueError('Complex magnetic moment')
                m = np.asarray(moment, dtype=float)
                if (m.shape != (3,) or not np.isfinite(m).all() or math.hypot(*m) < 1e-8):
                    raise ValueError('Missing, nonfinite or zero magnetic moment')
                m = unit_directions([m])[0]
                angle = math.degrees(math.atan2(float(np.linalg.norm(np.cross(m, axis))), float(m@axis)))
                row['moment_error_degrees'] = angle
                row['passed'] &= angle <= moment_tolerance_degrees
        except (ValueError, TypeError, AttributeError) as error:
            row.update(passed=False, reason=str(error))
        rows.append(row)
    return dict(passed=all(r['passed'] for r in rows), points=rows,
                moment_tolerance_degrees=moment_tolerance_degrees,
                physical_accuracy_qualified=False)
