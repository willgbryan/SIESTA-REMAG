"""Explicit final SOC direction after provider preemption of a source-bound relaxation."""
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from ..corpus import sha256
from ..discovery.collection import verify_archive
from ..discovery.common import digest, load, save
from ..discovery.pipeline import source_identity, verify_bundle
from ..discovery.seed import collinear_to_soc
from ..runner import run_case
from .input_reproduction import verify_prepared_input
from .runtime_policy import verify_saved

PREFIX = ('dft-relax', 'soc-z', 'soc-x')
LAST = 'soc-y'


def _unfinished(query):
    saved = load(query/'report.json')
    stages = saved['stages']
    last = query/LAST
    if (saved.get('status') != 'running' or saved.get('current_stage') != LAST
            or [s['stage'] for s in stages] != list(PREFIX)
            or any(s.get('accepted') is not True for s in stages)
            or any(s.get('branch', {}).get('passed') is not True for s in stages[1:])
            or any((last/name).exists() for name in ('result.json', 'review.json', '0_NORMAL_EXIT'))
            or 'Job completed' in (last/'stdout.log').read_text(errors='replace')):
        raise ValueError('Expected three accepted stages and an interrupted final direction')
    return saved


def snapshot_axis(volume_root, bundle, runtime, policy, run_id):
    """Pin the retained remote bytes and the scientific identity before allocation."""
    if not re.fullmatch(r'physics-[a-f0-9]{32}', run_id):
        raise ValueError('Invalid preempted volume run ID')
    volume_root, bundle = Path(volume_root), Path(bundle)
    query = volume_root/'query'
    cp = _unfinished(query)
    top = load(volume_root/'report.json')
    verify_saved(runtime, top['runtime_verification'])
    prepared = verify_bundle(bundle)
    code = source_identity()
    identity = dict(bundle=prepared['request_sha256'], files=prepared['files'],
                    executable=runtime['executable_sha256'], runtime_env=runtime.get('environment', {}),
                    threads=runtime.get('threads', 1), code=code, diagnostics=True)
    if (top.get('status') != 'running' or top.get('experiments') != []
            or top.get('policy') != policy or top.get('invocation_sha256') != digest(policy)
            or policy.get('operation') != 'relaxation' or policy.get('discovery_diagnostics') is not True
            or cp.get('prepared_bundle') != prepared or cp.get('source_code') != code
            or code != {k:v for k,v in policy['expected_source_code'].items() if not k.startswith('research/')}
            or cp.get('invocation_sha256') != digest(identity) or cp.get('diagnostics') is not True):
        raise ValueError('Preempted native checkpoint differs from its execution inputs')
    for name, expected in cp['accepted_artifacts'].items():
        path = query/name
        if not path.resolve().is_relative_to(query.resolve()) or sha256(path) != expected:
            raise ValueError('Preempted native artifact changed: '+name)
    from ase.io import read
    from .relaxation_execution import reproduce_stages
    request = load(bundle/'request.json')
    if cp.get('settings') != request['settings'] or cp.get('input_sha256') != request['input_sha256']:
        raise ValueError('Preemption checkpoint settings differ')
    reproduce_stages(query, bundle, request, prepared, cp, read(bundle/'orb/relaxed.extxyz').numbers, True)

    files = {}
    for stage, row in zip(PREFIX, cp['stages']):
        folder = query/stage
        names = set(row['artifact_sha256']) | set(row['effective_options_sha256'])
        names.update(['manifest.json', 'remag.KP', 'remag.EIG'])
        names.update(p.name for p in folder.glob('*.ion.xml'))
        files[stage] = {}
        for name in sorted(names):
            if not name or Path(name).name != name or name in ('.', '..'):
                raise ValueError('Invalid checkpoint artifact name')
            path = folder/name
            if path.is_symlink() or not path.resolve().is_relative_to(query.resolve()):
                raise ValueError('Invalid checkpoint artifact location')
            h = sha256(path)
            if (h != row['artifact_sha256'].get(name, h) or h != row['effective_options_sha256'].get(name, h)
                    or (name in row['artifact_sha256'] and h != cp['accepted_artifacts'].get(stage+'/'+name))):
                raise ValueError('Stage artifact does not match the saved review')
            files[stage][name] = h
    seeds = {}
    for stage in (*PREFIX[1:], LAST):
        name = 'seed-'+stage[-1]
        seed = load(query/(name+'.json'))
        if (seed['scalar_DM_sha256'] != files['dft-relax']['remag.DM']
                or seed['seed_sha256'] != sha256(query/(name+'.DM'))):
            raise ValueError('Preemption seed does not match its scalar source')
        # Reproduce the density for the original serialized input. A different
        # host can reconstruct a last-bit-different axis from the same cell.
        with TemporaryDirectory() as temp:
            check = Path(temp)/'seed.DM'
            direction = load(query/'inputs'/stage/'manifest.json')['direction']
            collinear_to_soc(query/'dft-relax/remag.DM', check, direction)
            if sha256(check) != seed['seed_sha256']:
                raise ValueError('Original input direction does not reproduce the saved SOC seed')
        seeds[stage] = seed['seed_sha256']
    folder = query/'inputs'/LAST
    input_names = {'input.fdf', 'manifest.json', *load(folder/'manifest.json')['pseudopotentials']}
    if set(p.name for p in folder.iterdir()) != input_names:
        raise ValueError('Unexpected saved final-axis input')
    return dict(run_id=run_id, report_sha256=sha256(volume_root/'report.json'),
                native_report_sha256=sha256(query/'report.json'), files=files, seeds=seeds,
                final_input={name:sha256(folder/name) for name in sorted(input_names)})


def admit_axis(attempt, owner, config, bundle, runtime, binding, ledger):
    """Bind a new single-direction grant to the stopped and reserved prior execution."""
    attempt = Path(attempt)
    old = load(attempt/'execution.json')
    auth = load(attempt/'authorization.json')
    prior = auth['config']
    receipt = load(attempt/'receipt.json')
    policy = load(attempt/'campaign/execution.json')
    app = config['previous_app_id']
    if (owner.get('status') != 'held_execution' or old.get('status') != 'held_execution'
            or any(v.get('app_id') != app for v in (old, owner, receipt))
            or not all(v.get('allocation_stopped') is True for v in (old, owner))
            or old.get('app_lifecycle', {}).get('state', '').lower() != 'stopped'
            or old['config_sha256'] != auth['config_sha256'] or config['authorization_id'] == prior['authorization_id']
            or any(prior.get(k) != config.get(k) for k in ('image','volume','runtime_sha256','budget_ledger'))
            or config.get('volume') != receipt['volume'] or receipt['image'] != config['image']
            or sha256(attempt/'runtime.json') != config['runtime_sha256'] or policy.get('source_binding') != binding
            or old['source_sha256'] != binding['source_sha256'] or 'final_axis_checkpoint' in policy
            or 'Container terminated due to preemption' not in (attempt/'launch.stdout').read_text()):
        raise ValueError('Previous owner is not the specified stopped preempted relaxation')
    rows = [r for r in ledger.snapshot()['reservations'] if r['id'] == receipt['run_id']]
    metadata = dict(app_id=app, receipt=str(attempt/'receipt.json'), policy_sha256=digest(policy))
    if (len(rows) != 1 or rows[0]['stage'] != prior['budget_stage']
            or any(rows[0]['metadata'].get(k) != v for k,v in metadata.items())):
        raise ValueError('Preempted receipt differs from its original ledger reservation')
    spec = snapshot_axis(attempt/'saved-volume', bundle, runtime, policy, receipt['run_id'])
    if digest(spec) != config['final_axis_checkpoint_sha256']:
        raise ValueError('Final-axis checkpoint differs from the reviewed grant')
    return spec


def review_seed_rejection(workspace, owner, config, bundle, runtime, binding, ledger):
    """Verify a stopped replay rejected a regenerated prefix seed before solver dispatch."""
    attempt = (workspace/owner['attempt']).resolve()
    if not attempt.is_relative_to(workspace):
        raise ValueError('Previous attempt is outside the workspace')
    execution = load(attempt/'execution.json')
    auth = load(attempt/'authorization.json')
    prior = auth['config']
    receipt = load(attempt/'receipt.json')
    policy = load(attempt/'campaign/execution.json')
    spec = policy.get('final_axis_checkpoint')
    if (owner.get('status') != 'held_review' or execution.get('status') != 'held_review'
            or any(x.get('app_id') != config['previous_app_id'] for x in (owner, receipt, execution))
            or not all(x.get('allocation_stopped') is True for x in (owner, execution))
            or execution.get('app_lifecycle', {}).get('state', '').lower() != 'stopped'
            or not spec or config['final_axis_checkpoint_sha256'] != digest(spec)
            or config['authorization_id'] == prior['authorization_id'] or 'seed_rejection_sha256' in prior
            or execution['config_sha256'] != auth['config_sha256'] or policy['source_binding'] != binding
            or any(config[k] != prior[k] for k in ('image','volume','runtime_sha256','budget_ledger','final_axis_checkpoint_sha256'))):
        raise ValueError('Expected a stopped first final-axis grant for the same checkpoint')
    reservations = {r['id']:r for r in ledger.snapshot()['reservations']}
    current = reservations[receipt['run_id']]
    if (current['metadata'].get('receipt') != str(attempt/'receipt.json') or current['stage'] != prior['budget_stage']
            or current['metadata'].get('policy_sha256') != digest(policy) or current['metadata'].get('app_id') != owner['app_id']):
        raise ValueError('Failed replay does not match its reservation')
    original = Path(reservations[spec['run_id']]['metadata']['receipt']).resolve().parent
    if not original.is_relative_to(workspace):
        raise ValueError('Preempted attempt escaped workspace')
    original_exec = load(original/'execution.json')
    if admit_axis(original, original_exec, prior, bundle, runtime, binding, ledger) != spec:
        raise ValueError('Failed replay checkpoint differs from original')
    collected = attempt/'collected'
    verify_archive(collected)
    q = collected/'query'
    report = load(q/'report.json')
    top = load(collected/'report.json')
    stage = report.get('current_stage')
    if (sha256(q/'report.json') != config['seed_rejection_sha256'] or top.get('policy') != policy
            or top.get('invocation_sha256') != digest(policy) or top.get('status') != 'held_experiment'
            or top.get('experiments') != [dict(id='query',status='held_error',report_sha256=sha256(q/'report.json'))]
            or report.get('reason') != 'Regenerated seed differs from the saved starting density'
            or report.get('status') != 'held_error' or stage not in PREFIX[1:]
            or [r['stage'] for r in report['stages']] != list(PREFIX[:PREFIX.index(stage)])
            or any((q/name).exists() for name in (stage,LAST,'inputs/'+LAST,'seed-y.DM'))):
        raise ValueError('Saved rejection is not a pre-dispatch seed mismatch')
    for row in report['stages']:
        for name, checksum in spec['files'][row['stage']].items():
            if sha256(q/row['stage']/name) != checksum:
                raise ValueError('Rejected replay changed retained evidence')
    previous_query = original/'saved-volume/query'
    verify_prepared_input(q/'inputs'/stage, previous_query/'inputs'/stage)
    seed_name = 'seed-'+stage[-1]+'.DM'
    with TemporaryDirectory() as temp:
        rebuilt = Path(temp)/seed_name
        collinear_to_soc(previous_query/'dft-relax/remag.DM', rebuilt,
                         load(q/'inputs'/stage/'manifest.json')['direction'])
        if sha256(rebuilt) != sha256(q/seed_name) or sha256(rebuilt) == spec['seeds'][stage]:
            raise ValueError('Saved prefix rejection does not reproduce from its input direction')
    return spec


def axis_runner(volume_root, spec, *, solver=run_case):
    """Reuse three immutable stage results; dispatch only the outstanding y input."""
    volume_root = Path(volume_root)
    query = volume_root/'query'
    _unfinished(query)
    if sha256(volume_root/'report.json') != spec['report_sha256'] or sha256(query/'report.json') != spec['native_report_sha256']:
        raise ValueError('Remote preemption checkpoint changed')
    remaining = list((*PREFIX, LAST))

    def run(case, executable, destination, **kwargs):
        stage = Path(destination).name
        if not remaining or remaining.pop(0) != stage:
            raise ValueError('Unexpected stage sequence in final-axis operation')
        retained = query/'inputs'/stage
        verify_prepared_input(case, retained)
        if stage != PREFIX[0]:
            restart = kwargs['restart']
            generated = Path(restart['path'])
            frozen = query/generated.name
            seed = load(frozen.with_suffix('.json'))
            expected = spec['seeds'][stage]
            if (sha256(generated) != restart['sha256'] or sha256(frozen) != expected
                    or seed['seed_sha256'] != expected or seed['scalar_DM_sha256'] != spec['files'][PREFIX[0]]['remag.DM']):
                raise ValueError('Final-axis operation has an invalid scalar-derived seed')
            save(generated.with_suffix('.reconstruction.json'), dict(host_regenerated_sha256=restart['sha256'],
                 original_sha256=expected, original_direction=seed['direction']))
            if restart['sha256'] != expected:
                shutil.copyfile(frozen, generated)
            save(generated.with_suffix('.json'), seed)
            kwargs['restart'] = dict(path=str(generated), sha256=expected)
        if stage == LAST:
            if (set(p.name for p in retained.iterdir()) != set(spec['final_input'])
                    or any(sha256(retained/n) != h for n,h in spec['final_input'].items())):
                raise ValueError('Saved final input changed before execution')
            return solver(retained, executable, destination, **kwargs)
        folder = query/stage
        verify_prepared_input(case, folder)
        destination.mkdir()
        for name, h in spec['files'][stage].items():
            target = destination/name
            shutil.copyfile(folder/name, target)
            if sha256(target) != h:
                raise ValueError('Recovered stage file differs: '+stage+'/'+name)
        return load(destination/'result.json')
    return run


def verify_collected_prefix(attempt, query, spec):
    """Require the combined archive to retain the original measured prefix."""
    auth = load(attempt/'authorization.json')
    if (auth['config'].get('final_axis_checkpoint_sha256') != digest(spec)
            or set(spec['files']) != set(PREFIX)):
        raise ValueError('Collected checkpoint differs from its operator authorization')
    for stage, files in spec['files'].items():
        for name, expected in files.items():
            if Path(name).name != name or sha256(query/stage/name) != expected:
                raise ValueError('Collected prefix differs from preempted native evidence')
    for name, expected in spec['final_input'].items():
        if Path(name).name != name or sha256(query/LAST/name) != expected:
            raise ValueError('Collected final-axis input differs from preempted run')
    for stage, expected in spec['seeds'].items():
        if stage not in (*PREFIX[1:], LAST) or sha256(query/('seed-'+stage[-1]+'.DM')) != expected:
            raise ValueError('Collected SOC seed differs from the admitted checkpoint')
