"""Idempotent finite pipeline. Completed stages are evidence, never retry hints."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from ..corpus import sha256
from ..runner import run_case
from .common import digest, load, save
from .config import Settings
from .inputs import axes, geometry, kmesh, MU0_MUB_PER_A3, prepare_case, pseudos_for, validate_atoms
from .review import branch_check, directional_result, evaluated_structure, native_review
from .seed import collinear_to_soc


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This candidate already has an active owner') from None
        yield


def source_identity():
    root = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in sorted(root.glob('*.py'))} | {
        '../runner.py': sha256(root.parent/'runner.py'), '../output.py': sha256(root.parent/'output.py')}


def prepare_bundle(source, pseudos, weights, bundle, settings=None, *, orb_device='cpu', orb_threads=4):
    """Run cheap Orb once on the caller, then stage a portable DFT bundle."""
    from ase.io import read, write
    settings = settings or Settings()
    bundle = Path(bundle).resolve()
    if orb_threads < 1 or orb_device not in ('cpu', 'cuda'):
        raise ValueError('Use cpu/cuda and a positive Orb thread count')
    source, weights, pseudos = Path(source).resolve(), Path(weights).resolve(), Path(pseudos).resolve()
    atoms = read(source)
    validate_atoms(atoms, settings)
    pseudo_hashes = pseudos_for(atoms, pseudos)
    request = dict(schema=1, input_sha256=sha256(source), orb_weights=str(weights),
                   orb_weights_sha256=sha256(weights), settings=settings.to_dict(),
                   pseudopotentials=pseudo_hashes, source_code=source_identity(),
                   orb_device=orb_device, orb_threads=orb_threads)
    identity = digest(request)
    with lock(bundle.with_name(bundle.name+'.lock')):
        if bundle.exists():
            old = load(bundle/'request.json')
            if digest(old) != identity: raise ValueError('Existing bundle belongs to different inputs/settings/code')
            if (bundle/'prepared.json').exists(): return verify_bundle(bundle)
            raise RuntimeError('Interrupted or failed Orb stage retained; use an explicitly new output for a new attempt')
        bundle.mkdir(parents=True)
        save(bundle/'request.json', request)
        write(bundle/'input.extxyz', atoms.copy())
        (bundle/'pseudos').mkdir()
        for name in pseudo_hashes: shutil.copyfile(pseudos/name, bundle/'pseudos'/name)
        started = time.monotonic()
        env = dict(os.environ)
        # Make an editable checkout available to the worker without relying on cwd.
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get('PYTHONPATH', '')
        with (bundle/'orb-worker.log').open('wb') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'siesta_remag.discovery.orb', str(bundle)],
                                    stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                code = proc.wait(timeout=settings.orb_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                code = proc.returncode
            except BaseException:
                os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=5)
                raise
        result_file = bundle/'orb/result.json'
        result = load(result_file) if result_file.exists() else dict(accepted=False, status='held_worker_failure_or_timeout')
        if code != 0: result.update(accepted=False, worker_returncode=code)
        result['worker_wall_seconds'] = time.monotonic()-started
        save(result_file, result)
        if not result['accepted']:
            save(bundle/'status.json', dict(status='held_orb', orb=result, next_stage=None))
            return load(bundle/'status.json')
        relaxed = read(bundle/'orb/relaxed.extxyz')
        validate_atoms(relaxed, settings)
        if list(relaxed.numbers) != list(atoms.numbers): raise ValueError('Orb changed species or atom order')
        # Check the requested mesh now, before any paid DFT allocation.
        mesh = kmesh(relaxed, settings)
        files = ['request.json', 'input.extxyz', 'orb/result.json', 'orb/relaxed.extxyz'] + ['pseudos/'+n for n in pseudo_hashes]
        receipt = dict(status='prepared', request_sha256=identity, kmesh=mesh,
                       files={n: sha256(bundle/n) for n in files},
                       orb_seconds=result['worker_wall_seconds'], physical_stability_qualified=False)
        save(bundle/'prepared.json', receipt)
        return receipt


def verify_bundle(bundle):
    bundle = Path(bundle)
    receipt = load(bundle/'prepared.json')
    request = load(bundle/'request.json')
    if receipt['status'] != 'prepared' or digest(request) != receipt['request_sha256']:
        raise ValueError('Invalid prepared bundle')
    required = {'request.json', 'input.extxyz', 'orb/result.json', 'orb/relaxed.extxyz'} | {'pseudos/'+n for n in request['pseudopotentials']}
    if set(receipt['files']) != required:
        raise ValueError('Incomplete prepared bundle manifest')
    for name, expected in receipt['files'].items():
        if not (bundle/name).resolve().is_relative_to(bundle.resolve()) or sha256(bundle/name) != expected:
            raise ValueError('Prepared bundle artifact changed: '+name)
    if not load(bundle/'orb/result.json')['accepted']:
        raise ValueError('Orb stage did not pass')
    from ase.io import read
    if receipt['kmesh'] != kmesh(read(bundle/'orb/relaxed.extxyz'), Settings.from_dict(request['settings'])):
        raise ValueError('Prepared k mesh changed')
    return receipt


def run_dft(bundle, output, executable, *, executable_sha256, runtime_env=None, threads=1, checkpoint=None, runner=run_case):
    """Relax and screen once, using one executable and sequential directions.

    A repeated completed call returns its report. Any interrupted allocation or
    failed scientific stage is held; it is never silently restarted/extended.
    """
    from ase.io import read, write
    bundle, output, executable = Path(bundle).resolve(), Path(output).resolve(), Path(executable).resolve()
    receipt = verify_bundle(bundle)
    request = load(bundle/'request.json')
    settings = Settings.from_dict(request['settings'])
    if sha256(executable) != executable_sha256: raise ValueError('Executable checksum mismatch')
    runtime_env = dict(runtime_env or {})
    # Capture inherited REMAG controls as well; never cache under an incomplete identity.
    runtime_env = {k: v for k, v in os.environ.items() if k.startswith('REMAG_')} | runtime_env
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in runtime_env.items()):
        raise ValueError('Runtime environment must map strings to strings')
    identity = digest(dict(bundle=receipt['request_sha256'], files=receipt['files'], executable=executable_sha256,
                           runtime_env=runtime_env, threads=threads, code=source_identity()))
    started = time.monotonic()
    deadline = started + settings.total_dft_seconds
    with lock(output.with_name(output.name+'.lock')):
        if output.exists():
            previous = load(output/'report.json')
            if previous['invocation_sha256'] != identity: raise ValueError('Output belongs to another invocation')
            if previous['status'] == 'running': raise RuntimeError('Interrupted run: preserve/recover its evidence; no automatic retry')
            for name, h in previous.get('accepted_artifacts', {}).items():
                if sha256(output/name) != h: raise ValueError('Cached accepted evidence changed: '+name)
            return previous
        output.mkdir(parents=True)
        report = dict(schema=1, status='running', invocation_sha256=identity, input_sha256=request['input_sha256'],
                      settings=settings.to_dict(), prepared_bundle=receipt, runtime_env=runtime_env,
                      executable_sha256=executable_sha256, source_code=source_identity(),
                      stages=[], accepted_artifacts={}, screen=None, K1=None, K2=None, kappa=None,
                      physical_accuracy_qualified=False, automatic_extensions=False,
                      scope='Finite discovery triage, not magnetic ground state or synthesis qualification')
        def flush():
            report['whole_pipeline_dft_seconds'] = time.monotonic()-started
            report['solver_seconds'] = sum(s.get('solver_seconds', 0) for s in report['stages'])
            report['orb_seconds'] = receipt['orb_seconds']
            save(output/'report.json', report)
            if checkpoint: checkpoint()
        def case(name, atoms, *, relax=False, axis=None, spins=None, restart=None, mesh=None):
            left = deadline-time.monotonic()
            if left <= 0: raise TimeoutError('Whole DFT pipeline time limit reached')
            limit = min(settings.dft_seconds if relax else settings.soc_seconds, left)
            prepared = prepare_case(atoms, bundle/'pseudos', output/'inputs'/name, settings, mesh=mesh,
                                    direction=axis, spins=spins, relax=relax, restart=restart is not None)
            report['current_stage'] = name; flush()
            result = runner(prepared, executable, output/name, timeout=limit, threads=threads,
                            extra_env=runtime_env, restart=restart)
            review = native_review(output/name, result, settings, relaxation=relax, axis=axis)
            review['stage'] = name
            save(output/name/'review.json', review)
            report['stages'].append(review); flush()
            if not review['accepted']:
                report.update(status='held_'+name, reason='; '.join(review['errors']), current_stage=None)
                return None
            for artifact, h in review['artifact_sha256'].items():
                report['accepted_artifacts'][name+'/'+artifact] = h
            return result['observables']
        try:
            atoms = read(bundle/'orb/relaxed.extxyz')
            scalar = case('dft-relax', atoms, relax=True, mesh=receipt['kmesh'])
            if scalar is None: return report
            relaxed = evaluated_structure(output/'dft-relax', atoms.numbers)
            validate_atoms(relaxed, settings)
            write(output/'dft-relaxed.extxyz', relaxed)
            write(output/'dft-relaxed.cif', relaxed)
            report['accepted_artifacts']['dft-relaxed.extxyz'] = sha256(output/'dft-relaxed.extxyz')
            report['relaxation'] = dict(geometry=geometry(relaxed), source='dft-relax/remag.STRUCT_OUT',
                Js_T=float(np.linalg.norm(scalar['total_moment_muB'])*MU0_MUB_PER_A3/relaxed.get_volume()),
                volume_ratio_to_orb=float(relaxed.get_volume()/atoms.get_volume()))
            if report['relaxation']['Js_T'] < settings.minimum_js_t:
                report.update(status='screened_low_moment', reason='SOC skipped below minimum_js_t', current_stage=None)
                return report
            dm = output/'dft-relax/remag.DM'
            dm_hash = sha256(dm)
            rotations = axes(relaxed); rows = {}
            # Keep the integer mesh used by the density-producing relaxation.
            # Recomputing it after a cell change would invalidate reuse of that density.
            for label, axis in rotations.items():
                if sha256(dm) != dm_hash: raise ValueError('Accepted scalar density changed')
                seed = output/('seed-'+label+'.DM')
                seed_check = collinear_to_soc(dm, seed, axis)
                save(output/('seed-'+label+'.json'), dict(seed_check, scalar_DM_sha256=dm_hash, seed_sha256=sha256(seed)))
                obs = case('soc-'+label, relaxed, axis=axis, spins=np.asarray(scalar['atom_moments_muB'])[:, 2],
                           mesh=receipt['kmesh'], restart=dict(path=str(seed), sha256=sha256(seed)))
                if obs is None: return report
                actual = evaluated_structure(output/('soc-'+label), relaxed.numbers)
                if not np.allclose(actual.cell.array, relaxed.cell.array, atol=2e-8, rtol=0) or not np.allclose(actual.positions, relaxed.positions, atol=2e-7, rtol=0):
                    raise ValueError('SOC geometry changed from the accepted DFT relaxation')
                branch = branch_check(obs, rows.get('z', scalar), axis, rotations['z'] if rows else [0, 0, 1], settings)
                report['stages'][-1]['branch'] = branch
                save(output/('soc-'+label)/'review.json', report['stages'][-1])
                if not branch['passed']:
                    report.update(status='held_magnetic_branch', reason='SOC changed the scalar magnetic branch', current_stage=None)
                    return report
                rows[label] = obs
                flush()
            integrations = {(tuple(s['native_grid']), s['native_kpoints']) for s in report['stages'][1:]}
            if len(integrations) != 1:
                raise ValueError('SOC directions used different native grids or k-point counts')
            report.update(status='screened', current_stage=None, screen=directional_result(rows, relaxed.get_volume()))
            return report
        except Exception as exc:
            report.update(status='held_error', reason=str(exc))
            return report
        finally:
            flush()
