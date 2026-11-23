"""One explicitly enabled agent query through the existing guarded Modal path.

No ledger creation, reference build, reservation refund, retry, map or warm pool.
The operator supplies a fixed REMAG runtime/image and an existing authorized
budget stage. A policy can select prepared queries, never those execution grants.
"""
from datetime import datetime, timezone
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import threading

from ..corpus import sha256
from ..discovery.collection import verify_archive
from ..discovery.common import load, save, digest
from ..discovery.config import Settings
from ..discovery.pipeline import lock
from .budget import BudgetLedger, reservation_cents
from .experiments import verify_experiment, experiment_code
from .queries import requested_spectra
from .preflight import validate_runtime, preflight_query

SCALAR_SCF_FAILURE = 'previous_scalar_scf_failure_report_sha256'

def _path(root, value):
    if not isinstance(value,str) or not value:
        raise ValueError('Execution paths must be inside the workspace')
    path = (root/value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Execution path escapes workspace')
    return path


def read_execution_config(workspace, path):
    """Read-only preflight. An authorization file is operator input, not proof
    of human consent by itself; the CLI/session must explicitly enable it.
    """
    root = Path(workspace).resolve(); path = _path(root,str(path))
    c = load(path)
    required = {'format','authorization_id','authorization','budget_ledger','budget_stage','runtime',
                'runtime_sha256','image','previous_app_id','app_timeout_seconds',
                'hourly_reservation_usd','max_queries','max_reserved_cents'}
    endpoint = 'relaxation_endpoint_report_sha256'
    optional = {'volume', 'operation', 'driver_rejection_report_sha256', 'final_axis_checkpoint_sha256', 'seed_rejection_sha256', 'previous_scf_failure_report_sha256', endpoint, SCALAR_SCF_FAILURE}
    if (not isinstance(c,dict) or not required <= set(c) or set(c)-required-optional
            or c['format'] != 'REMAG-RESEARCH-EXECUTION/1'):
        raise ValueError('Expected explicit REMAG research execution configuration')
    if c.get('operation', 'query') not in ('query', 'relaxation'):
        raise ValueError('Unsupported execution operation')
    if SCALAR_SCF_FAILURE in c and (
            c.get('operation') != 'query' or c['max_queries'] != 1
            or (optional - {SCALAR_SCF_FAILURE, 'volume', 'operation'}) & c.keys()
            or not isinstance(c[SCALAR_SCF_FAILURE], str) or not re.fullmatch('[0-9a-f]{64}', c[SCALAR_SCF_FAILURE])):
        raise ValueError('A stopped scalar SCF failure requires one independent report-bound query grant')
    if endpoint in c and (c.get('operation', 'query') != 'query' or c['max_queries'] != 1
            or (optional - {endpoint, 'volume', 'operation'}) & c.keys()
            or not isinstance(c[endpoint], str) or not re.fullmatch('[0-9a-f]{64}', c[endpoint])):
        raise ValueError('An endpoint force measurement requires one report-bound scalar query grant')
    if 'previous_scf_failure_report_sha256' in c and (
            c.get('operation') != 'relaxation' or c['max_queries'] != 1
            or {'seed_rejection_sha256', 'final_axis_checkpoint_sha256', 'driver_rejection_report_sha256'} & c.keys()
            or not isinstance(c['previous_scf_failure_report_sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', c['previous_scf_failure_report_sha256'])):
        raise ValueError('Independent composition requires one report-bound relaxation grant')
    if 'seed_rejection_sha256' in c and ('final_axis_checkpoint_sha256' not in c
            or not isinstance(c['seed_rejection_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', c['seed_rejection_sha256'])):
        raise ValueError('Seed rejection review requires a report-bound final-axis grant')
    if 'final_axis_checkpoint_sha256' in c and (
            c.get('operation') != 'relaxation' or c['max_queries'] != 1 or not c.get('volume')
            or 'driver_rejection_report_sha256' in c or not isinstance(c['final_axis_checkpoint_sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', c['final_axis_checkpoint_sha256'])):
        raise ValueError('Final SOC direction requires a single checkpoint-bound relaxation grant')
    if 'driver_rejection_report_sha256' in c and (c.get('operation') not in ('query', 'relaxation') or c['max_queries'] != 1
            or not isinstance(c['driver_rejection_report_sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', c['driver_rejection_report_sha256'])):
        raise ValueError('Driver rejection continuation requires one report-bound execution grant')
    if 'volume' in c and (not isinstance(c['volume'],str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',c['volume'])):
        raise ValueError('Expected a named operator-selected result volume')
    if (not isinstance(c['authorization_id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',c['authorization_id'])
            or not isinstance(c['authorization'],str) or not c['authorization'].strip()
            or not isinstance(c['budget_stage'],str) or not c['budget_stage']):
        raise ValueError('Named operator authorization and existing budget stage required')
    for key,prefix in [('image','im-'),('previous_app_id','ap-')]:
        if not isinstance(c[key],str) or not re.fullmatch(prefix+r'[A-Za-z0-9]+',c[key]):
            raise ValueError('Explicit existing image and previous app identities required')
    if type(c['max_queries']) is not int or not 1 <= c['max_queries'] <= 32:
        raise ValueError('Execution authorization allows 1..32 single queries')
    cents = reservation_cents(c['app_timeout_seconds'],c['hourly_reservation_usd'])
    if cents > 10000 or type(c['max_reserved_cents']) is not int or not cents <= c['max_reserved_cents'] <= 100000:
        raise ValueError('Invalid finite execution reservation ceiling')
    runtime_path = _path(root,c['runtime']); runtime = load(runtime_path)
    if sha256(runtime_path) != c['runtime_sha256']:
        raise ValueError('Authorized REMAG runtime changed')
    validate_runtime(runtime)
    ledger_path = _path(root,c['budget_ledger'])
    if not ledger_path.is_file():
        raise ValueError('An existing authorized budget ledger is required')
    ledger = BudgetLedger(ledger_path)
    if ledger.remaining_cents(c['budget_stage']) < cents:
        raise ValueError('Insufficient existing stage reservation capacity')
    return dict(config=c, sha256=sha256(path), runtime=runtime, reserved_cents=cents,
                runtime_path=runtime_path, ledger=ledger,
                # Bound upload/client collection and leave time for authoritative
                # stop observation and local review before the session deadline.
                required_session_seconds=c['app_timeout_seconds']+480)


class ModalExecution:
    """Production transport; all launches pass through scripts/run_modal.py."""
    def __init__(self, repo=None):
        self.repo = Path(repo or Path(__file__).resolve().parents[3])

    def info(self, app_id):
        result = subprocess.run([sys.executable,'-m','modal','app','info',app_id,'--json'],
            check=True,capture_output=True,text=True,timeout=15)
        row = json.loads(result.stdout)
        if row.get('app_id') != app_id or not isinstance(row.get('lifecycle'),dict):
            raise ValueError('Authoritative app identity differs')
        return row

    def stop(self, app_id):
        subprocess.run([sys.executable,'-m','modal','app','stop',app_id,'--yes'],check=True,
                       capture_output=True,text=True,timeout=15)

    def stopped(self, app_id):
        try:row = self.info(app_id)
        except Exception:row = {}
        if row.get('lifecycle',{}).get('state','').lower() != 'stopped':
            try:self.stop(app_id)
            except Exception:pass  # A concurrent terminal transition can make stop return nonzero.
        deadline = time.monotonic()+45
        while True:
            try:
                row = self.info(app_id)
                if row['lifecycle'].get('state','').lower() == 'stopped':return row
            except Exception:
                pass  # Read failures never prove release; repeat only observation.
            if time.monotonic() >= deadline:
                raise RuntimeError('Owned app shutdown unconfirmed; do not submit another query')
            time.sleep(2)

    def launch(self, attempt, config):
        env = dict(os.environ, REMAG_RESEARCH_CAMPAIGN=str(attempt/'campaign'),
            REMAG_RESEARCH_RUNTIME=str(attempt/'runtime.json'), REMAG_RESEARCH_IMAGE=config['image'],
            REMAG_RESEARCH_BUDGET_LEDGER=config['budget_ledger'])
        command = [sys.executable,'scripts/run_modal.py','--module-file','scripts/research_modal.py',
                   '--receipt',str(attempt/'receipt.json'),'--output',str(attempt/'collected')]
        with (attempt/'launch.stdout').open('xb') as out,(attempt/'launch.stderr').open('xb') as err:
            p = subprocess.Popen(command,cwd=self.repo,env=env,stdout=out,stderr=err,start_new_session=True)
            try:
                code = p.wait(timeout=config['app_timeout_seconds']+300)
            except BaseException:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise
        if code:
            raise RuntimeError('Guarded launch/collection failed; inspect retained streams and receipt')


def _review(attempt, request, runtime):
    from .evidence import verify_report as _verified_report
    root = attempt/'collected'; verify_archive(root)
    top = load(root/'report.json'); policy = load(attempt/'campaign/execution.json')
    if top['invocation_sha256'] != digest(policy) or top.get('policy') != policy:
        raise ValueError('Collected allocation belongs to another execution policy')
    if runtime.get('capabilities') is not None:
        from .runtime_policy import verify_saved
        verify_saved(runtime, top.get('runtime_verification'))
    report = _verified_report(root/'query/report.json',root/'archive-manifest.json')
    if (report['request'] != request or report['source_sha256'] != request['source_sha256']
            or report['executable_sha256'] != runtime['executable_sha256']
            or report['environment'] != runtime.get('environment',{})
            or report['source_code'] != policy['expected_source_code']
            or report['invocation_sha256'] != digest(dict(request=request,executable_sha256=runtime['executable_sha256'],
                environment=runtime.get('environment',{}),threads=runtime.get('threads',1),code=policy['expected_source_code']))):
        raise ValueError('Collected query source/input/executable/runtime differs')
    expected = top.get('experiments',[])
    if len(expected) != 1 or expected[0] != dict(id='query',status=report['status'],report_sha256=sha256(root/'query/report.json')):
        raise ValueError('Collected query report binding differs')
    healthy = (top['status'] == 'measured' and report['status'] == 'measured' and bool(report['stages'])
               and all(s['accepted'] and s.get('branch',{}).get('passed',True) for s in report['stages']))
    spectra = []
    if healthy:
        if request.get('soc_matrix',False):
            expected_soc={s['stage'] for s in report['stages'] if s['stage']!='scalar'}
            if not expected_soc or report.get('soc_operator_checks')!={s:{'native_pair_verified':True} for s in expected_soc}:
                raise ValueError('Requested native SOC operator checks are missing')
            for stage in expected_soc:
                if any(not re.fullmatch('[0-9a-f]{64}',report['artifacts'].get(stage+'/'+name,''))
                       for name in ('remag.SOC.HSX','remag.SOC.json')):
                    raise ValueError('Requested native SOC operator artifacts are missing')
        from .inspection import load_spectrum
        for selected in policy.get('spectra',[]):
            stage,spin=selected['stage'],selected['spin']
            path=root/'query'/('electronic-'+stage+'-spin'+str(spin))/'electronic.json'
            meta,_=load_spectrum(path)
            if (meta['hsx_sha256'] != report['artifacts'].get(stage+'/remag.HSX')
                    or meta['kp_sha256'] != report['artifacts'].get(stage+'/remag.KP')
                    or meta['eig_sha256'] != report['artifacts'].get(stage+'/remag.EIG')
                    or meta['collinear_spin'] != (spin if stage=='scalar' else None)):
                raise ValueError('Requested spectrum belongs to another native stage/channel')
            spectra.append(dict(**selected,path=str(path.relative_to(attempt)),sha256=sha256(path)))
    return dict(healthy=healthy, native_status=report['status'], report='collected/query/report.json',
        report_sha256=sha256(root/'query/report.json'), collection='collected/archive-manifest.json',
        observations={stage:{k:v for k,v in row.items() if k != 'eigenvalues'} for stage,row in report['observations'].items()},
        anisotropy=report.get('anisotropy'), angular_fit=report.get('angular_fit'),
        native_wall_seconds=report.get('wall_seconds'), remote_timing=top.get('timing'), spectra=spectra,verification=report['verification'],
        runtime_verification=top.get('runtime_verification'))


@contextmanager
def _stop_on_signal():
    # Let the tool worker reach owned-app cleanup when its parent terminates
    # the local client. A hard kill still leaves durable unresolved ownership.
    enabled = threading.current_thread() is threading.main_thread()
    def interrupted(signum, frame):
        raise InterruptedError('Execution worker interrupted')
    prior = signal.signal(signal.SIGTERM,interrupted) if enabled else None
    try:yield
    finally:
        if enabled:signal.signal(signal.SIGTERM,prior)


def execute_query(workspace, execution_config, bundle, output, *, spectra=None, transport=None):
    with _stop_on_signal():
        return _execute_query(workspace,execution_config,bundle,output,spectra=spectra,transport=transport)


def execute_relaxation(workspace, execution_config, structure, bundle, output, *, transport=None):
    """Operator-only source-bound discovery through the same allocation owner.

    Requires an explicit relaxation grant. This is not a direct-policy tool.
    """
    with _stop_on_signal():
        return _execute_query(workspace, execution_config, bundle, output,
                              transport=transport, operation='relaxation', structure=structure)


def _review_driver_rejection(root, owner, config, preflight, source_binding, *, query_bundle=None, spectra=None):
    """Admit a fresh operator grant for the same case after a pre-solver host rejection."""
    from .runtime_policy import check, validate
    attempt = _path(root, owner['attempt'])
    collected = attempt/'collected'
    verify_archive(collected)
    report_path = collected/'report.json'
    report = load(report_path)
    old = load(attempt/'execution.json')
    policy = load(attempt/'campaign/execution.json')
    authorization = load(attempt/'authorization.json')
    old_config = authorization['config']
    app_id = config['previous_app_id']
    operation = config['operation']
    if operation == 'query':
        if query_bundle is None or source_binding is not None or policy.get('spectra') != [] or spectra:
            raise ValueError('Driver rejection continuation requires the same fixed query without spectra')
        saved = attempt/'campaign/query'
        request = verify_experiment(saved)
        if (request != verify_experiment(query_bundle) or request.get('query') not in ({'kind':'scalar'}, {'kind':'anisotropy'})
                or request.get('fixed_geometry') is not True or sha256(saved/'experiment.json') != preflight['request_sha256']
                or old_config.get('operation', 'query') != operation):
            raise ValueError('Driver rejection continuation requires an identical fixed scalar or anisotropy query')
        if request['query'] == {'kind':'anisotropy'}:
            limits = ('volume', 'app_timeout_seconds', 'hourly_reservation_usd', 'max_reserved_cents')
            if ('driver_rejection_report_sha256' in old_config or request.get('soc_matrix', False)
                    or any(old_config.get(key) != config.get(key) for key in limits)):
                raise ValueError('A fixed anisotropy query permits one host replacement with its original limits')
        source_sha = request['source_sha256']
    else:
        source_sha = source_binding['source_sha256']
    if (owner.get('status') != 'held_review' or owner.get('allocation_stopped') is not True
            or owner.get('app_id') != app_id or old.get('app_id') != app_id
            or old.get('status') != 'held_review' or old.get('allocation_stopped') is not True
            or old['config_sha256'] != authorization['config_sha256']
            or old['source_sha256'] != source_sha
            or old.get('app_lifecycle', {}).get('state','').lower() != 'stopped'
            or load(attempt/'receipt.json').get('app_id') != app_id
            or old_config['authorization_id'] == config['authorization_id']
            or old_config['image'] != config['image'] or old_config['runtime_sha256'] != config['runtime_sha256']
            or sha256(attempt/'runtime.json') != config['runtime_sha256']
            or sha256(report_path) != config['driver_rejection_report_sha256']
            or report.get('status') != 'held_error' or report.get('experiments') != []
            or report.get('timing', {}).get('native_seconds') != 0
            or set(load(collected/'archive-manifest.json')['files']) != {'report.json'}
            or report.get('policy') != policy or report.get('invocation_sha256') != digest(policy)
            or policy.get('operation', 'query') != operation or policy.get('source_binding') != source_binding
            or load(attempt/'preflight.json') != preflight or old['request_sha256'] != preflight['request_sha256']):
        raise ValueError('Previous attempt is not the authorized stopped pre-solver driver rejection')
    prefix = 'Native runtime profile is not admitted: '
    reason = report.get('reason', '')
    if not reason.startswith(prefix):
        raise ValueError('Previous error is not a driver rejection')
    observed = json.loads(reason[len(prefix):])
    runtime = load(attempt/'runtime.json')
    policy = validate(runtime['driver_policy'])
    if not isinstance(observed, dict) or 'format' in observed:
        raise ValueError('Invalid saved driver observation')
    validate({**observed, 'format': policy['format']})
    if len(observed['nvidia_driver_versions']) != 1:
        raise ValueError('Expected one observed GPU driver')
    try:
        check(policy, observed)
    except ValueError as exc:
        if str(exc) == reason:
            return str(report_path.relative_to(root))
    raise ValueError('Previous runtime rejection does not reproduce')


def _stopped_scf_attempt(root, owner, plan, report_field):
    """Bind the recorded failure to its stopped owner, receipt and reservation."""
    config = plan['config']
    previous = config['previous_app_id']
    attempt = _path(root, owner['attempt'])
    result = load(attempt/'execution.json')
    receipt = load(attempt/'receipt.json')
    grant = load(attempt/'authorization.json')
    report = attempt/'collected/query/report.json'
    if (owner.get('status') != 'held_native_review' or owner.get('allocation_stopped') is not True
            or owner.get('app_id') != previous or receipt.get('app_id') != previous
            or any(result.get(k) != owner[k] for k in ('app_id', 'status', 'allocation_stopped'))
            or result.get('app_lifecycle', {}).get('state', '').lower() != 'stopped'
            or grant['config']['authorization_id'] == config['authorization_id']
            or grant['config_sha256'] != result['config_sha256']
            or sha256(report) != config[report_field]
            or result['review']['report_sha256'] != sha256(report)):
        raise ValueError('Independent composition requires its reviewed stopped predecessor')
    reservation = next((row for row in plan['ledger'].snapshot()['reservations']
                        if row['id'] == receipt.get('run_id')), None)
    if (not reservation or reservation['metadata'].get('app_id') != previous
            or reservation['metadata'].get('receipt') != str(attempt/'receipt.json')):
        raise ValueError('Previous receipt differs from its reservation')
    return attempt, result, reservation


def _require_different_composition(prior_structure, next_structure):
    from collections import Counter
    from fractions import Fraction

    def composition(path):
        numbers = load(path)['geometry']['atomic_numbers']
        return {z: Fraction(n, len(numbers)) for z, n in Counter(numbers).items()}

    if composition(prior_structure) == composition(next_structure):
        raise ValueError('This grant requires a different chemical composition')


def _review_prior_scf_failure(root, owner, plan, structure):
    """Admit an independent composition after a reproducible, stopped SCF rejection."""
    from .relaxation_execution import review_relaxation

    attempt, result, _ = _stopped_scf_attempt(root, owner, plan, 'previous_scf_failure_report_sha256')
    report = attempt/'collected/query/report.json'
    stage = report.parent/'dft-relax'
    reproduced = review_relaxation(attempt, load(attempt/'campaign/query/request.json'),
                                   load(attempt/'runtime.json'), require_current_code=False)
    exit_status = load(stage/'execution-status.json')
    marker = 'REMAG device SCF export: Final electronic state requires convergence'
    if (reproduced['healthy'] or reproduced['native_status'] != 'held_dft-relax'
            or reproduced['source_binding']['source_sha256'] != result['source_sha256']
            or exit_status['timed_out'] is not False or exit_status['returncode'] != -6
            or marker not in (stage/'stderr.log').read_text().splitlines()):
        raise ValueError('Prior native artifacts do not establish the terminal SCF convergence rejection')

    _require_different_composition(attempt/'campaign/structure.json', structure)
    return str(report.relative_to(root))


def _review_fixed_scalar_failure(root, owner, plan, bundle):
    from ase.io.siesta import read_fdf

    attempt, result, reservation = _stopped_scf_attempt(root, owner, plan, SCALAR_SCF_FAILURE)
    config = plan['config']
    old = load(attempt/'authorization.json')['config']
    prior = attempt/'campaign/query'
    request = verify_experiment(prior)
    policy = load(attempt/'campaign/execution.json')
    if (result.get('format') != 'REMAG-QUERY-EXECUTION/1' or old.get('operation', 'query') != 'query'
            or request.get('query') != {'kind': 'scalar'} or result['source_sha256'] != request['source_sha256']
            or result['request_sha256'] != sha256(prior/'experiment.json')
            or sha256(attempt/'runtime.json') != config['runtime_sha256']
            or any(old.get(k) != config.get(k) for k in ('runtime_sha256', 'image', 'volume', 'budget_ledger'))
            or reservation['metadata'].get('policy_sha256') != digest(policy)
            or reservation.get('stage') != old['budget_stage']
            or reservation.get('cents') != result['reservation_recorded_cents']):
        raise ValueError('Fixed scalar request, runtime or reservation does not match the stopped attempt')
    reproduced = _review(attempt, request, load(attempt/'runtime.json'))
    path = attempt/'collected/query/report.json'
    report = load(path)
    stages = report['stages']
    stage = path.parent/'scalar'
    exit_status = load(stage/'execution-status.json')
    marker = 'SCF_NOT_CONV: SCF did not converge in maximum number of steps (required).'
    if (reproduced['healthy'] or reproduced['native_status'] != 'held_scalar' or len(stages) != 1
            or stages[0]['stage'] != 'scalar' or stages[0]['accepted'] is not False
            or stages[0]['scf_iterations'] != request['settings']['max_scf']
            or report['observations']['scalar']['converged'] is not False
            or exit_status['timed_out'] is not False or exit_status['returncode'] != -6
            or marker not in (stage/'stderr.log').read_text().splitlines()):
        raise ValueError('Fixed scalar artifacts do not establish exhaustion of the native SCF limit')
    for prepared in (prior, bundle):
        numbers = load(prepared/'structure.json')['geometry']['atomic_numbers']
        native = read_fdf(str(prepared/'scalar/input.fdf'))
        species = {index: int(z) for index, z, label in native['chemicalspecieslabel']}
        ordered = [species[row[3]] for row in native['atomiccoordinatesandatomicspecies']]
        if ordered != numbers or native['numberofatoms'] != [str(len(numbers))]:
            raise ValueError('Fixed scalar native input and authored species differ')
    _require_different_composition(prior/'structure.json', bundle/'structure.json')
    return str(path.relative_to(root))


def _diagnostic_review(root, owner, plan, preflight, source_binding=None):
    """A narrow operator review, never a successful-outcome rewrite or retry grant."""
    path = root/'.research-execution/diagnostic-reviews'/(plan['sha256']+'.json')
    if not path.is_file():
        raise ValueError('Previous execution needs explicit evidence review; no resubmission or follow-on query')
    review = load(path)
    if (review.get('format') != 'REMAG-CHANGED-RUNTIME-DIAGNOSTIC-REVIEW/1'
            or review.get('owner') != owner or review.get('next_config_sha256') != plan['sha256']
            or review.get('next_request_sha256') != preflight['request_sha256']
            or review.get('operation', 'query') != plan['config'].get('operation', 'query')
            or review.get('next_source_binding') != source_binding
            or review.get('native_fix_claimed') is not False):
        raise ValueError('Diagnostic review does not match the held owner or next grant')
    for name, expected in review['evidence_sha256'].items():
        if sha256(_path(root, name)) != expected:
            raise ValueError('Diagnostic review evidence changed')
    return str(path.relative_to(root))


def review_failed_runtime_for_diagnostic(workspace, execution_config, bundle, *, evidence, reason, transport=None, structure=None):
    """Operator-only: bind one changed-runtime diagnostic to a reviewed stopped failure.

    This API is deliberately absent from research-policy and browser tool catalogs.
    Calling it requires separate owner authorization; a reason is not consent.
    It neither launches nor edits the failed outcome, ownership or budget history.
    """
    root = Path(workspace).resolve(); plan = read_execution_config(root, execution_config)
    c = plan['config']; control = root/'.research-execution'
    operation = c.get('operation', 'query'); source_binding = None
    if operation == 'relaxation':
        from .relaxation_execution import preflight_relaxation, verify_relaxation_source
        if structure is None: raise ValueError('A relaxation diagnostic needs its exact authored source')
        preflight = preflight_relaxation(_path(root, str(bundle)), plan['runtime'])
        source_binding = verify_relaxation_source(_path(root, str(structure)), _path(root, str(bundle)))
    else:
        preflight = preflight_query(_path(root, str(bundle)), plan['runtime'])
    if (c['max_queries'] != 1 or not isinstance(reason, str) or not 20 <= len(reason) <= 8000
            or not isinstance(evidence, list) or not 1 <= len(evidence) <= 32):
        raise ValueError('One diagnostic grant, explicit review reason and saved evidence required')
    transport = transport or ModalExecution()
    with lock(control/'owner.lock'):
        owner = load(control/'owner.json')
        if owner.get('status') not in ('held_native_review', 'held_review') or owner.get('allocation_stopped') is not True or not owner.get('app_id'):
            raise ValueError('Only a stopped native-review failure can receive this diagnostic review')
        prior = _path(root, owner['attempt']); result = load(prior/'execution.json')
        if any(result.get(k) != owner.get(k) for k in ('status', 'app_id', 'allocation_stopped')):
            raise ValueError('Held owner does not match its retained execution')
        old_runtime = load(prior/'runtime.json'); new_runtime = plan['runtime']
        verify_archive(prior/'collected')
        if owner['status'] == 'held_review':
            # A failed Python review is not by itself proof of native failure.
            # Admit only the retained relaxation-native failure pattern; keep
            # its original status and report untouched.
            top = load(prior/'collected/report.json')
            native_path = prior/'collected/query/dft-relax/execution-status.json'
            if (result.get('format') != 'REMAG-RELAXATION-EXECUTION/1'
                    or top.get('policy', {}).get('operation') != 'relaxation'
                    or not native_path.is_file()):
                raise ValueError('Held review needs verified native relaxation failure evidence')
            native = load(native_path); report = load(prior/'collected/query/report.json')
            if (report.get('executable_sha256') != old_runtime['executable_sha256']
                    or report.get('runtime_env') != old_runtime.get('environment', {})
                    or report.get('status') != 'held_error'
                    or type(native.get('timed_out')) is not bool
                    or not (native['timed_out'] or (type(native.get('returncode')) is int and native['returncode'] != 0))):
                raise ValueError('Held review does not establish a matching native failure')
        identity = lambda r: (r['executable_sha256'], r.get('capabilities', {}).get('library_sha256'))
        if identity(old_runtime) == identity(new_runtime):
            raise ValueError('Diagnostic requires changed executable/library bytes, not a path or environment relabel')
        if (control/'authorizations'/(c['authorization_id']+'.json')).exists():
            raise ValueError('Diagnostic must use a new unattempted authorization')
        paths = [str((prior/name).relative_to(root)) for name in ('execution.json','runtime.json','receipt.json','collected/archive-manifest.json')] + evidence
        bindings = {name:sha256(_path(root,name)) for name in paths}
        observation = transport.info(owner['app_id'])
        if observation.get('app_id') != owner['app_id'] or observation.get('lifecycle',{}).get('state','').lower() != 'stopped':
            raise ValueError('Reviewed owner is not authoritatively STOPPED')
        review = dict(format='REMAG-CHANGED-RUNTIME-DIAGNOSTIC-REVIEW/1',
            recorded_at=datetime.now(timezone.utc).isoformat(), owner=owner, next_config_sha256=plan['sha256'],
            next_request_sha256=preflight['request_sha256'],
            operation=operation, next_source_binding=source_binding,
            evidence_sha256=bindings, reason=reason, native_fix_claimed=False, stopped_observation=observation)
        path = control/'diagnostic-reviews'/(plan['sha256']+'.json'); path.parent.mkdir(exist_ok=True)
        with path.open('x') as stream:json.dump(review,stream,indent=2);stream.write('\n')
        return review


def _execute_query(workspace, execution_config, bundle, output, *, spectra=None, transport=None,
                   operation='query', structure=None):
    """One finite submission/collection/shutdown/review. Never retries an attempt.

Transport injection is for engineering tests, not an agent-selectable backend.
The owner file blocks later submissions after uncertain launch or failed health
review. Disappearing local sessions never count as allocation release.
"""
    requested = time.monotonic(); requested_at = datetime.now(timezone.utc).isoformat()
    root = Path(workspace).resolve(); plan = read_execution_config(root,execution_config); c = plan['config']
    if c.get('operation', 'query') != operation:
        raise ValueError('Execution operation is not authorized by this configuration')
    bundle = _path(root,str(bundle)); output = _path(root,str(output))
    source_binding = None
    if operation == 'relaxation':
        from .relaxation_execution import preflight_relaxation, verify_relaxation_source
        structure = _path(root, str(structure))
        preflight = preflight_relaxation(bundle, plan['runtime'])
        source_binding = verify_relaxation_source(structure, bundle)
        request = load(bundle/'request.json')
        source_sha = source_binding['source_sha256']
        code = experiment_code(); spectra = []
    else:
        request = verify_experiment(bundle)
        preflight = preflight_query(bundle,plan['runtime'])
        from .experiments import validate_soc_request
        validate_soc_request(request.get('soc_matrix',False),request.get('query'),plan['runtime'].get('environment',{}))
        spectra = requested_spectra(request.get('query',{'kind':'anisotropy'}),[] if spectra is None else spectra)
        source_sha = request['source_sha256']; code = experiment_code()
    settings = Settings.from_dict(request['settings'])
    final_axis = 'final_axis_checkpoint_sha256' in c
    native_limit = settings.soc_seconds if final_axis else settings.total_dft_seconds
    if native_limit+60 > c['app_timeout_seconds']:
        raise ValueError('Query native ceiling does not fit the authorized app deadline')
    if output.exists() or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError('Use a new attempt directory separate from the prepared bundle')
    transport = transport or ModalExecution()
    control = _path(root,'.research-execution'); control.mkdir(exist_ok=True)
    if output.is_relative_to(control):raise ValueError('Attempt output cannot replace controller ownership state')
    with lock(control/'owner.lock'):
        owner_path = control/'owner.json'; previous = c['previous_app_id']
        continuation_review = None
        checkpoint_spec = None
        if final_axis and not owner_path.exists():
            raise ValueError('Missing owner for final SOC direction')
        if 'driver_rejection_report_sha256' in c and not owner_path.exists():
            raise ValueError('No owned driver rejection for this grant')
        if ('previous_scf_failure_report_sha256' in c or SCALAR_SCF_FAILURE in c) and not owner_path.exists():
            raise ValueError('No owned SCF rejection for this grant')
        if 'relaxation_endpoint_report_sha256' in c and not owner_path.exists():
            raise ValueError('No owned relaxation endpoint for this grant')
        if owner_path.exists():
            owner = load(owner_path)
            if 'relaxation_endpoint_report_sha256' in c:
                from .endpoint_diagnostic import review_endpoint_query
                continuation_review = review_endpoint_query(
                    root, _path(root, owner['attempt']), owner, plan, bundle)
            elif SCALAR_SCF_FAILURE in c:
                continuation_review = _review_fixed_scalar_failure(root, owner, plan, bundle)
            elif 'previous_scf_failure_report_sha256' in c:
                continuation_review = _review_prior_scf_failure(root, owner, plan, structure)
            elif 'seed_rejection_sha256' in c:
                from .preempted_axis import review_seed_rejection
                checkpoint_spec = review_seed_rejection(root, owner, c, bundle, plan['runtime'], source_binding, plan['ledger'])
                continuation_review = owner['attempt']+'/collected/query/report.json'
            elif final_axis:
                from .preempted_axis import admit_axis
                checkpoint_spec = admit_axis(_path(root,owner['attempt']), owner, c, bundle, plan['runtime'],
                                             source_binding, plan['ledger'])
                continuation_review = owner['attempt']+'/saved-volume/query/report.json'
            elif 'driver_rejection_report_sha256' in c:
                continuation_review = _review_driver_rejection(root, owner, c, preflight, source_binding,
                    query_bundle=bundle if operation=='query' else None, spectra=spectra)
            elif owner['status'] != 'measured' or not owner.get('allocation_stopped') or not owner.get('app_id'):
                from .relaxation_recovery import verified_execution, NAME as RECOVERY
                previous_attempt = _path(root, owner['attempt'])
                if owner['status']=='held_review' and (previous_attempt/RECOVERY).exists():
                    recovered = verified_execution(previous_attempt)
                    if (recovered['status']!='measured' or recovered['app_id']!=owner.get('app_id') or
                            recovered['allocation_stopped'] is not True or owner.get('allocation_stopped') is not True):
                        raise ValueError('Recovered prior execution is not a stopped measured result')
                    continuation_review = str((previous_attempt/RECOVERY).relative_to(root))
                else:
                    continuation_review = _diagnostic_review(root, owner, plan, preflight, source_binding)
            previous = owner['app_id']
        prior = transport.info(previous)
        if prior.get('app_id') != previous or prior.get('lifecycle',{}).get('state','').lower() != 'stopped':
            raise ValueError('Previous app is not authoritatively STOPPED')
        authorizations = _path(root,'.research-execution/authorizations');authorizations.mkdir(exist_ok=True)
        auth_path = authorizations/(c['authorization_id']+'.json')
        auth = load(auth_path) if auth_path.exists() else dict(config_sha256=plan['sha256'],attempts=[])
        if auth['config_sha256'] != plan['sha256']:
            raise ValueError('An execution authorization cannot be edited or reset')
        if len(auth['attempts']) >= c['max_queries'] or (len(auth['attempts'])+1)*plan['reserved_cents'] > c['max_reserved_cents']:
            raise ValueError('Execution authorization query/reservation limit reached')
        output.mkdir(parents=True,exist_ok=False); (output/'campaign').mkdir()
        shutil.copytree(bundle,output/'campaign/query')
        if operation == 'relaxation':
            shutil.copyfile(structure, output/'campaign/structure.json')
            if (preflight_relaxation(output/'campaign/query', plan['runtime']) != preflight or
                    verify_relaxation_source(output/'campaign/structure.json', output/'campaign/query') != source_binding):
                raise ValueError('Prepared relaxation changed while snapshotting')
        elif verify_experiment(output/'campaign/query') != request:
            raise ValueError('Prepared query changed while snapshotting')
        shutil.copyfile(plan['runtime_path'],output/'runtime.json')
        if sha256(output/'runtime.json') != c['runtime_sha256']:
            raise ValueError('Authorized runtime changed while snapshotting')
        policy = dict(experiments=['query'],timeout_seconds=c['app_timeout_seconds'],
            hourly_reservation_usd=c['hourly_reservation_usd'],budget_usd=plan['reserved_cents']/100,
            budget_stage=c['budget_stage'],spectra=spectra,expected_source_code=code)
        if operation == 'relaxation':
            policy.update(operation=operation, source_binding=source_binding, discovery_diagnostics=True)
        if checkpoint_spec is not None:
            policy['final_axis_checkpoint'] = checkpoint_spec
        if 'volume' in c:policy['volume'] = c['volume']
        save(output/'campaign/execution.json',policy)
        save(output/'preflight.json',preflight)
        save(output/'authorization.json',dict(config=c,config_sha256=plan['sha256'],prior_app=prior,
            diagnostic_review=continuation_review))
        started = time.monotonic()
        result = dict(format='REMAG-RELAXATION-EXECUTION/1' if operation == 'relaxation' else 'REMAG-QUERY-EXECUTION/1',
            status='submitting',config_sha256=plan['sha256'],
            source_sha256=source_sha,request_sha256=preflight['request_sha256'],
            requested_at=requested_at,request_to_submission_seconds=started-requested,
            submitted_at=datetime.now(timezone.utc).isoformat(),app_id=None,allocation_stopped=False,
            reserved_ceiling_cents=plan['reserved_cents'],reservation_recorded_cents=None,actual_billed_usd=None,
            retry_authorized=False,physical_accuracy_qualified=False)
        # Persist both intent and ownership before any launch, including an
        # indeterminate failure between app creation and receipt publication.
        auth['attempts'].append(str(output.relative_to(root))); save(auth_path,auth)
        def flush():
            save(output/'execution.json',result)
            save(owner_path,dict(attempt=str(output.relative_to(root)),status=result['status'],
                app_id=result['app_id'],allocation_stopped=result['allocation_stopped']))
        flush()
        try:
            transport.launch(output,c | {'budget_ledger':str(plan['ledger'].path)})
            result['submission_to_collection_seconds'] = time.monotonic()-started
            result['status'] = 'collected'
        except BaseException as exc:
            result.update(status='held_execution',error=dict(type=type(exc).__name__,message=str(exc)))
        finally:
            if (output/'receipt.json').exists():
                try:
                    receipt = load(output/'receipt.json')
                    app_id = receipt.get('app_id')
                    if not isinstance(app_id,str) or not re.fullmatch(r'ap-[A-Za-z0-9]+',app_id):
                        raise ValueError('No owned app identity in receipt')
                    # Only the existing launcher/ledger may establish ownership.
                    ledger = plan['ledger'].snapshot()
                    reservations = [r for r in ledger['reservations'] if r['id'] == receipt.get('run_id')]
                    if len(reservations) != 1 or reservations[0]['metadata'].get('receipt') != str(output/'receipt.json') or reservations[0]['metadata'].get('app_id') != app_id:
                        raise ValueError('Receipt is not bound to the owned ledger reservation')
                    result['app_id'] = app_id; result['reservation_recorded_cents'] = reservations[0]['cents']; flush()
                    lifecycle = transport.stopped(app_id)
                    if lifecycle.get('app_id') != app_id or lifecycle.get('lifecycle',{}).get('state','').lower() != 'stopped':
                        raise ValueError('Owned app shutdown was not authoritative')
                    result.update(allocation_stopped=True,app_lifecycle=lifecycle['lifecycle'])
                    plan['ledger'].record(receipt['run_id'],dict(status='agent_query_stopped',lifecycle=lifecycle['lifecycle']))
                    if (reservations[0]['cents'] != plan['reserved_cents'] or reservations[0].get('stage') != c['budget_stage']
                            or reservations[0]['metadata'].get('policy_sha256') != digest(policy)):
                        raise ValueError('Owned reservation policy/stage/ceiling differs')
                except Exception as exc:
                    result.update(status='held_shutdown',shutdown_error=dict(type=type(exc).__name__,message=str(exc)))
            else:
                result.update(status='held_unknown_owner',shutdown_error=dict(message='No receipt; allocation release is unconfirmed'))
            flush()
        if result['status'] == 'collected' and result['allocation_stopped']:
            try:
                if operation == 'relaxation':
                    from .relaxation_execution import review_relaxation
                    result['review'] = review_relaxation(output, request, plan['runtime'])
                else:
                    result['review'] = _review(output,request,plan['runtime'])
                result['status'] = 'measured' if result['review']['healthy'] else 'held_native_review'
                result['review']['report'] = str((output/result['review']['report']).relative_to(root))
                result['review']['collection'] = str((output/result['review']['collection']).relative_to(root))
                for spectrum in result['review']['spectra']:
                    spectrum['path']=str((output/spectrum['path']).relative_to(root))
            except Exception as exc:
                result.update(status='held_review',review_error=dict(type=type(exc).__name__,message=str(exc)))
        result.update(finished_at=datetime.now(timezone.utc).isoformat(),
            submission_through_review_seconds=time.monotonic()-started,
            request_through_review_seconds=time.monotonic()-requested,
            scope='One controlled REMAG query; measured means native acceptance, not physical convergence or calibrated exploratory accuracy. Queue/startup remain unseparated unless timing evidence provides them.')
        result['usable_result_seconds'] = result['request_through_review_seconds'] if result['status'] == 'measured' else None
        if operation == 'relaxation':
            result['scope'] = ('One controlled relaxation and magnetic screen; measured includes a completed negative screen. '
                               'Exact authored source is separate from relaxed observations. No synthesis, stability, '
                               'ground-state, convergence or calibrated-latency qualification.')
        flush()
        return result
