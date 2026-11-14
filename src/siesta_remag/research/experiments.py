"""Fixed-geometry, sequential scalar-seed and self-consistent SOC experiments."""
import os
from pathlib import Path
import shutil
import time

import numpy as np

from ..corpus import sha256
from ..runner import run_case
from ..discovery.common import digest, load, save
from ..discovery.config import Settings
from ..discovery.inputs import axes, prepare_case, pseudos_for, validate_atoms
from ..discovery.pipeline import lock, source_identity
from ..discovery.review import native_review, evaluated_structure, branch_check, directional_result
from ..discovery.seed import collinear_to_soc


def prepare_experiment(export, pseudos, output, *, settings=None, mesh=None,
                       basis_size='DZP', basis_energy_shift_ry=.02, parameter=None):
    """GPAC is optional at installation and required only at this authoring boundary."""
    from ase import Atoms
    from gpac.physical import verify_export
    verified = verify_export(export)
    settings = settings or Settings()
    g = verified['geometry']
    atoms = Atoms(numbers=g['atomic_numbers'], scaled_positions=g['fractional'], cell=g['cell'], pbc=True)
    validate_atoms(atoms, settings)
    pseudo_hashes = pseudos_for(atoms, pseudos, allow_psf=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output/'structure.json', verified)
    (output/'pseudos').mkdir()
    for name in pseudo_hashes:
        shutil.copyfile(Path(pseudos)/name, output/'pseudos'/name)
    case = prepare_case(atoms, output/'pseudos', output/'scalar', settings, mesh=mesh,
        allow_psf=True, basis_size=basis_size, basis_energy_shift_ry=basis_energy_shift_ry, diagnostics=True)
    manifest = load(case/'manifest.json')
    request = dict(format='REMAG-EXPERIMENT/1', source_sha256=verified['source']['sha256'],
        export_sha256=verified['sha256'], parameter=parameter, fixed_geometry=True,
        settings=settings.to_dict(), kmesh=manifest['kmesh'], basis_size=basis_size,
        basis_energy_shift_ry=basis_energy_shift_ry, pseudopotentials=pseudo_hashes,
        files={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    save(output/'experiment.json', request)
    return request


def verify_experiment(bundle):
    bundle = Path(bundle).resolve()
    request = load(bundle/'experiment.json')
    if request.get('format') != 'REMAG-EXPERIMENT/1' or request.get('fixed_geometry') is not True:
        raise ValueError('Expected a fixed-geometry REMAG experiment')
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file()} - {'experiment.json'}
    if actual != set(request['files']):
        raise ValueError('Experiment file inventory differs')
    for name, h in request['files'].items():
        if not (bundle/name).resolve().is_relative_to(bundle) or sha256(bundle/name) != h:
            raise ValueError('Experiment artifact changed: ' + name)
    structure = load(bundle/'structure.json')
    if structure['sha256'] != request['export_sha256'] or structure['source']['sha256'] != request['source_sha256']:
        raise ValueError('Experiment source binding differs')
    return request


def run_experiment(bundle, output, executable, *, executable_sha256, runtime_env=None,
                   threads=1, checkpoint=None, runner=run_case):
    """Run four bounded jobs in order; no relaxations, retries or implicit extensions."""
    from ase import Atoms
    bundle, output, executable = Path(bundle).resolve(), Path(output).resolve(), Path(executable).resolve()
    request = verify_experiment(bundle)
    if sha256(executable) != executable_sha256:
        raise ValueError('Executable checksum mismatch')
    settings = Settings.from_dict(request['settings'])
    env = {k: v for k, v in os.environ.items() if k.startswith('REMAG_')} | dict(runtime_env or {})
    code = source_identity() | {f'research/{p.name}': sha256(p) for p in Path(__file__).parent.glob('*.py')}
    identity = digest(dict(request=request, executable_sha256=executable_sha256, environment=env, threads=threads, code=code))
    started = time.monotonic()
    with lock(output.with_name(output.name+'.lock')):
        if output.exists():
            prior = load(output/'report.json')
            if prior['invocation_sha256'] != identity or prior['status'] == 'running':
                raise ValueError('Changed or interrupted experiment; use a new output for an explicit attempt')
            for name, h in prior['artifacts'].items():
                if sha256(output/name) != h:
                    raise ValueError('Completed evidence changed: '+name)
            return prior
        output.mkdir(parents=True)
        structure = load(bundle/'structure.json'); g = structure['geometry']
        atoms = Atoms(numbers=g['atomic_numbers'], cell=g['cell'], scaled_positions=g['fractional'], pbc=True)
        report = dict(format='REMAG-EXPERIMENT-RESULT/1', status='running', invocation_sha256=identity,
            source_sha256=request['source_sha256'], request=request, structure=structure,
            executable_sha256=executable_sha256, environment=env, source_code=code,
            stages=[], observations={}, artifacts={}, anisotropy=None, physical_accuracy_qualified=False)

        def flush():
            report['wall_seconds'] = time.monotonic()-started
            save(output/'report.json', report)
            if checkpoint: checkpoint()

        def run(label, prepared, axis=None, restart=None):
            left = settings.total_dft_seconds-(time.monotonic()-started)
            if left <= 0:
                raise TimeoutError('Experiment time budget exhausted')
            result = runner(prepared, executable, output/label, timeout=min(settings.soc_seconds, left),
                            threads=threads, extra_env=env, restart=restart)
            review = native_review(output/label, result, settings, axis=axis)
            review['stage'] = label
            report['stages'].append(review)
            report['observations'][label] = result['observables']
            save(output/label/'review.json', review)
            flush()
            if not review['accepted']:
                report['status'] = 'held_'+label
                return None
            actual = evaluated_structure(output/label, atoms.numbers)
            if not np.allclose(actual.cell.array, atoms.cell.array, atol=2e-8, rtol=0) or not np.allclose(actual.positions, atoms.positions, atol=2e-7, rtol=0):
                raise ValueError('Fixed geometry changed during '+label)
            return result['observables']

        try:
            scalar = run('scalar', bundle/'scalar')
            if scalar is None: return report
            rotations = axes(atoms); rows = {}
            for label, axis in rotations.items():
                seed = output/('seed-'+label+'.DM')
                seed_info = collinear_to_soc(output/'scalar/remag.DM', seed, axis)
                save(output/('seed-'+label+'.json'), seed_info)
                prepared = prepare_case(atoms, bundle/'pseudos', output/'inputs'/label, settings,
                    mesh=request['kmesh'], direction=axis, spins=np.array(scalar['atom_moments_muB'])[:, 2],
                    restart=True, allow_psf=True, basis_size=request['basis_size'],
                    basis_energy_shift_ry=request['basis_energy_shift_ry'], diagnostics=True)
                obs = run('soc-'+label, prepared, axis, dict(path=str(seed), sha256=sha256(seed)))
                if obs is None: return report
                branch = branch_check(obs, rows.get('z', scalar), axis, rotations['z'] if rows else [0, 0, 1], settings)
                report['stages'][-1]['branch'] = branch
                if not branch['passed']:
                    report['status'] = 'held_magnetic_branch'; return report
                rows[label] = obs
            integrations = {(tuple(s['native_grid']), s['native_kpoints']) for s in report['stages'][1:]}
            if len(integrations) != 1:
                raise ValueError('Directional integration grids differ')
            report['anisotropy'] = directional_result(rows, atoms.get_volume()) | dict(
                mae_ev_cell=rows['x']['total_energy_ev']-rows['z']['total_energy_ev'],
                transverse_split_ev_cell=rows['y']['total_energy_ev']-rows['x']['total_energy_ev'],
                convention='E[100] - E[001]; positive favors [001]', uncertainty_ev_cell=None)
            report['anisotropy']['axis_convention'] = 'Fixed authored Cartesian frame; [001] is c'
            report['status'] = 'measured'
        except Exception as exc:
            report.update(status='held_error', reason=str(exc))
        finally:
            report['artifacts'] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                                   if p.is_file() and p != output/'report.json'}
            flush()
        return report
