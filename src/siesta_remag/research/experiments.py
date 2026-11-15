"""Fixed-geometry, sequential scalar-seed and self-consistent SOC experiments."""
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone

import numpy as np

from ..corpus import sha256
from ..runner import run_case
from ..discovery.common import digest, load, save
from ..discovery.config import Settings
from ..discovery.inputs import axes, prepare_case, pseudos_for, validate_atoms
from ..discovery.pipeline import lock, source_identity
from ..discovery.review import native_review, evaluated_structure, branch_check, directional_result
from ..discovery.seed import collinear_to_soc
from .queries import validate_query, directions, angular_fit


def prepare_experiment(export, pseudos, output, *, settings=None, mesh=None,
                       basis_size='DZP', basis_energy_shift_ry=.02, parameter=None, basis_spec=None, query=None):
    """GPAC is optional at installation and required only at this authoring boundary."""
    from ase import Atoms
    from gpac.physical import verify_export
    verified = verify_export(export)
    selected_query = validate_query(query) if query is not None else None
    settings = settings or Settings()
    g = verified['geometry']
    atoms = Atoms(numbers=g['atomic_numbers'], scaled_positions=g['fractional'], cell=g['cell'], pbc=True)
    validate_atoms(atoms, settings)
    if basis_spec is not None:
        from .basis import validate_basis
        basis_spec = validate_basis(basis_spec, atoms.get_chemical_symbols())
        basis_size = 'explicit'
    pseudo_hashes = pseudos_for(atoms, pseudos, allow_psf=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output/'structure.json', verified)
    (output/'pseudos').mkdir()
    for name in pseudo_hashes:
        shutil.copyfile(Path(pseudos)/name, output/'pseudos'/name)
    case = prepare_case(atoms, output/'pseudos', output/'scalar', settings, mesh=mesh,
        allow_psf=True, basis_size=basis_size, basis_energy_shift_ry=basis_energy_shift_ry, diagnostics=True,
        basis_spec=basis_spec)
    manifest = load(case/'manifest.json')
    request = dict(format='REMAG-EXPERIMENT/1', source_sha256=verified['source']['sha256'],
        export_sha256=verified['sha256'], parameter=parameter, fixed_geometry=True,
        settings=settings.to_dict(), kmesh=manifest['kmesh'], basis_size=basis_size,
        basis_energy_shift_ry=basis_energy_shift_ry, pseudopotentials=pseudo_hashes,
        files={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    if basis_spec is not None: request['basis_spec'] = basis_spec
    if selected_query is not None: request['query'] = selected_query
    save(output/'experiment.json', request)
    return request


def verify_experiment(bundle):
    bundle = Path(bundle).resolve()
    request = load(bundle/'experiment.json')
    if request.get('format') != 'REMAG-EXPERIMENT/1' or request.get('fixed_geometry') is not True:
        raise ValueError('Expected a fixed-geometry REMAG experiment')
    if 'query' in request:
        validate_query(request['query'])
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file()} - {'experiment.json'}
    if actual != set(request['files']):
        raise ValueError('Experiment file inventory differs')
    for name, h in request['files'].items():
        if not (bundle/name).resolve().is_relative_to(bundle) or sha256(bundle/name) != h:
            raise ValueError('Experiment artifact changed: ' + name)
    structure = load(bundle/'structure.json')
    if structure['sha256'] != request['export_sha256'] or structure['source']['sha256'] != request['source_sha256']:
        raise ValueError('Experiment source binding differs')
    manifest = load(bundle/'scalar/manifest.json')
    if any(manifest.get(k) != request.get(k) for k in ['basis_spec', 'basis_size', 'basis_energy_shift_ry']):
        raise ValueError('Scalar and requested PAO basis differ')
    if request.get('basis_spec') is not None:
        from .basis import validate_basis
        spec = validate_basis(request['basis_spec'], manifest['geometry']['species'])
        if spec != request['basis_spec'] or request['basis_size'] != 'explicit':
            raise ValueError('Noncanonical explicit PAO request')
    return request


def run_experiment(bundle, output, executable, *, executable_sha256, runtime_env=None,
                   threads=1, checkpoint=None, runner=run_case):
    """Run the selected bounded jobs; historical requests retain scalar/z/x/y."""
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
            stages=[], observations={}, artifacts={}, anisotropy=None, physical_accuracy_qualified=False,
            started_at=datetime.now(timezone.utc).isoformat())

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
            review['execution_wall_seconds'] = result.get('wall_seconds')
            report['stages'].append(review)
            report['observations'][label] = result['observables']
            save(output/label/'review.json', review)
            flush()
            if not review['accepted']:
                report['status'] = 'held_'+label
                return None
            if request.get('basis_spec') is not None:
                from .basis import audit_native_basis
                audit = audit_native_basis(output/label, request['basis_spec'], atoms.get_chemical_symbols())
                prior_checks = report.setdefault('basis_checks', {})
                if prior_checks:
                    baseline = next(iter(prior_checks.values()))
                    if [s['orbitals'] for s in baseline['species']] != [s['orbitals'] for s in audit['species']]:
                        raise ValueError('Scalar/SOC native basis radii or shells differ')
                prior_checks[label] = audit
            actual = evaluated_structure(output/label, atoms.numbers)
            if not np.allclose(actual.cell.array, atoms.cell.array, atol=2e-8, rtol=0) or not np.allclose(actual.positions, atoms.positions, atol=2e-7, rtol=0):
                raise ValueError('Fixed geometry changed during '+label)
            return result['observables']

        try:
            scalar = run('scalar', bundle/'scalar')
            if scalar is None: return report
            query = validate_query(request.get('query', {'kind': 'anisotropy'}))
            rotations = directions(query, axes(atoms)); rows = {}
            reference_axis, reference_observation = [0, 0, 1], scalar
            for label, axis in rotations.items():
                seed = output/('seed-'+label+'.DM')
                seed_info = collinear_to_soc(output/'scalar/remag.DM', seed, axis)
                save(output/('seed-'+label+'.json'), seed_info)
                prepared = prepare_case(atoms, bundle/'pseudos', output/'inputs'/label, settings,
                    mesh=request['kmesh'], direction=axis, spins=np.array(scalar['atom_moments_muB'])[:, 2],
                    restart=True, allow_psf=True, basis_size=request['basis_size'],
                    basis_energy_shift_ry=request['basis_energy_shift_ry'], diagnostics=True,
                    basis_spec=request.get('basis_spec'))
                obs = run('soc-'+label, prepared, axis, dict(path=str(seed), sha256=sha256(seed)))
                if obs is None: return report
                branch = branch_check(obs, reference_observation, axis, reference_axis, settings)
                report['stages'][-1]['branch'] = branch
                if not branch['passed']:
                    report['status'] = 'held_magnetic_branch'; return report
                rows[label] = obs
                if len(rows) == 1:
                    reference_observation, reference_axis = obs, axis
            integrations = {(tuple(s['native_grid']), s['native_kpoints']) for s in report['stages'][1:]}
            if rotations and len(integrations) != 1:
                raise ValueError('Directional integration grids differ')
            if query['kind'] == 'anisotropy':
                report['anisotropy'] = directional_result(rows, atoms.get_volume()) | dict(
                    mae_ev_cell=rows['x']['total_energy_ev']-rows['z']['total_energy_ev'],
                    transverse_split_ev_cell=rows['y']['total_energy_ev']-rows['x']['total_energy_ev'],
                    convention='E[x] - E[z]; positive favors z; see axes_cartesian', uncertainty_ev_cell=None,
                    axes_cartesian=rotations)
                report['anisotropy']['axis_convention'] = 'z parallel to authored c; x perpendicular projection of authored a; y=z cross x'
            else:
                report['query'] = query
                report['axes_cartesian'] = rotations
                if query['kind'] == 'angular':
                    report['angular_fit'] = angular_fit(query, rows)
            report['status'] = 'measured'
        except Exception as exc:
            report.update(status='held_error', reason=str(exc))
        finally:
            report['finished_at'] = datetime.now(timezone.utc).isoformat()
            report['latency_scope'] = 'Local execution and review; submission, queue, allocation startup and retrieval excluded'
            report['artifacts'] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                                   if p.is_file() and p != output/'report.json'}
            flush()
        return report
