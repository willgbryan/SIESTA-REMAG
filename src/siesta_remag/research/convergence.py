"""One-variable numerical stability checks, distinct from physical qualification."""
import math
from pathlib import Path

from ..corpus import sha256
from ..discovery.common import load


VARIABLES = {'kmesh': ('kmesh',), 'grid': ('mesh_ry',),
             'scf': ('dm_tolerance', 'h_tolerance_ev'), 'basis': ('basis_size',),
             'radial_extent': ('basis_energy_shift_ry',), 'smearing': ('smearing_k',)}


def numerical_configuration(report):
    request = report['request']
    result = {key: request[key] for key in ('kmesh', 'basis_size', 'basis_energy_shift_ry', 'pseudopotentials')}
    result.update({key: request['settings'][key] for key in
                   ('mesh_ry', 'smearing_k', 'dm_tolerance', 'h_tolerance_ev', 'mixing_weight', 'mixing_history')})
    result.update(source_sha256=report['source_sha256'], executable_sha256=report['executable_sha256'],
                  environment=report['environment'])
    return result


def assess(reports, *, varied, tolerance_ev, successive_changes=2, transverse_tolerance_ev=None):
    """Check E and F anisotropies without combining unlike calculations.

    Paths bind file hashes; dictionaries support in-memory use and have no file
    provenance claim. Execution timeouts and iteration ceilings are excluded
    from the physical configuration after native convergence has passed.
    """
    if varied not in VARIABLES:
        raise ValueError('Unknown convergence variable')
    if type(tolerance_ev) not in (int, float) or not math.isfinite(tolerance_ev) or tolerance_ev <= 0:
        raise ValueError('Positive finite tolerance in eV/cell required')
    if type(successive_changes) is not int or not 1 <= successive_changes <= 10:
        raise ValueError('Require 1..10 successive changes')
    if transverse_tolerance_ev is not None and (type(transverse_tolerance_ev) not in (int, float)
            or not math.isfinite(transverse_tolerance_ev) or transverse_tolerance_ev <= 0):
        raise ValueError('Invalid transverse tolerance')
    if len(reports) < 2: raise ValueError('At least two measured reports required')
    configs, samples = [], []
    for item in reports:
        path = Path(item) if isinstance(item, (str, Path)) else None
        r = load(path) if path else item
        if (r.get('format') != 'REMAG-EXPERIMENT-RESULT/1' or r.get('status') != 'measured'
                or r['request'].get('fixed_geometry') is not True
                or r['request']['source_sha256'] != r['source_sha256']):
            raise ValueError('A source-bound measured fixed-geometry report is required')
        if [s['stage'] for s in r['stages']] != ['scalar', 'soc-z', 'soc-x', 'soc-y'] or not all(s['accepted'] for s in r['stages']):
            raise ValueError('Native stage checks did not all pass')
        obs = r['observations']; z, x, y = [obs['soc-'+axis] for axis in ('z','x','y')]
        e = x['total_energy_ev']-z['total_energy_ev']
        f = x['free_energy_ev']-z['free_energy_ev']
        split = max(abs(y['total_energy_ev']-x['total_energy_ev']), abs(y['free_energy_ev']-x['free_energy_ev']))
        if not all(math.isfinite(v) for v in (e, f, split)):
            raise ValueError('Nonfinite directional energy')
        if not math.isclose(e, r['anisotropy']['mae_ev_cell'], rel_tol=0, abs_tol=1e-10):
            raise ValueError('Reported anisotropy differs from directional energies')
        config = numerical_configuration(r);configs.append(config)
        samples.append(dict(parameters={k: config[k] for k in VARIABLES[varied]},
            energy_anisotropy_ev=e, free_energy_anisotropy_ev=f, transverse_split_ev=split,
            report_sha256=sha256(path) if path else None))
    fixed = [{k:v for k,v in c.items() if k not in VARIABLES[varied]} for c in configs]
    if any(c != fixed[0] for c in fixed[1:]):
        raise ValueError('Unrelated scientific settings, source or executable differ')
    for a, b in zip(samples, samples[1:]):
        if a['parameters'] == b['parameters']: raise ValueError('Repeated parameter is not a refinement')
        if varied == 'kmesh' and any(j < i for i,j in zip(a['parameters']['kmesh'], b['parameters']['kmesh'])):
            raise ValueError('K meshes must refine componentwise in the supplied order')
        if varied == 'grid' and b['parameters']['mesh_ry'] <= a['parameters']['mesh_ry']:
            raise ValueError('Grid cutoff must increase')
        if varied == 'scf' and any(b['parameters'][k] > a['parameters'][k] for k in VARIABLES[varied]):
            raise ValueError('SCF tolerances must tighten')
    changes = [dict(energy_ev=b['energy_anisotropy_ev']-a['energy_anisotropy_ev'],
                    free_energy_ev=b['free_energy_anisotropy_ev']-a['free_energy_anisotropy_ev'])
               for a,b in zip(samples,samples[1:])]
    sufficient = len(changes) >= successive_changes
    stable = sufficient and all(abs(v) <= tolerance_ev for d in changes[-successive_changes:] for v in d.values())
    transverse = transverse_tolerance_ev is None or all(s['transverse_split_ev'] <= transverse_tolerance_ev for s in samples)
    return dict(format='REMAG-NUMERICAL-STABILITY/1', varied=varied, samples=samples, changes=changes,
        tolerance_ev=tolerance_ev, successive_changes_required=successive_changes,
        transverse_tolerance_ev=transverse_tolerance_ev, transverse_passed=transverse,
        sufficient_refinements=sufficient, observed_stability_passed=stable and transverse,
        fixed_configuration=fixed[0], physical_accuracy_qualified=False,
        scope='Observed E and F stability under the declared controls; no extrapolation, fitted error bar or physical-accuracy certificate')
