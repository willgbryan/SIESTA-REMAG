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
    if (not isinstance(c,dict) or not required <= set(c) or set(c)-required-{'volume'}
            or c['format'] != 'REMAG-RESEARCH-EXECUTION/1'):
        raise ValueError('Expected explicit REMAG research execution configuration')
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
    if (set(runtime)-{'executable','executable_sha256','environment','threads'}
            or not isinstance(runtime.get('executable'),str) or not runtime['executable'].startswith('/')
            or not re.fullmatch(r'[0-9a-f]{64}',runtime.get('executable_sha256',''))
            or type(runtime.get('threads',1)) is not int or not 1 <= runtime.get('threads',1) <= 8
            or not isinstance(runtime.get('environment',{}),dict)
            or any(not isinstance(k,str) or not isinstance(v,str) for k,v in runtime.get('environment',{}).items())):
        raise ValueError('Expected a pinned REMAG runtime manifest')
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
    from .tools import _verified_report
    root = attempt/'collected'; verify_archive(root)
    top = load(root/'report.json'); policy = load(attempt/'campaign/execution.json')
    if top['invocation_sha256'] != digest(policy) or top.get('policy') != policy:
        raise ValueError('Collected allocation belongs to another execution policy')
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
        native_wall_seconds=report.get('wall_seconds'), remote_timing=top.get('timing'), spectra=spectra,verification=report['verification'])


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


def _execute_query(workspace, execution_config, bundle, output, *, spectra=None, transport=None):
    """One finite submission/collection/shutdown/review. Never retries an attempt.

Transport injection is for engineering tests, not an agent-selectable backend.
The owner file blocks later submissions after uncertain launch or failed health
review. Disappearing local sessions never count as allocation release.
"""
    requested = time.monotonic(); requested_at = datetime.now(timezone.utc).isoformat()
    root = Path(workspace).resolve(); plan = read_execution_config(root,execution_config); c = plan['config']
    bundle = _path(root,str(bundle)); output = _path(root,str(output))
    request = verify_experiment(bundle); settings = Settings.from_dict(request['settings'])
    from .experiments import validate_soc_request
    validate_soc_request(request.get('soc_matrix',False),request.get('query'),plan['runtime'].get('environment',{}))
    spectra = requested_spectra(request.get('query',{'kind':'anisotropy'}),[] if spectra is None else spectra)
    if settings.total_dft_seconds+60 > c['app_timeout_seconds']:
        raise ValueError('Query native ceiling does not fit the authorized app deadline')
    if output.exists() or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError('Use a new attempt directory separate from the prepared bundle')
    transport = transport or ModalExecution()
    control = _path(root,'.research-execution'); control.mkdir(exist_ok=True)
    if output.is_relative_to(control):raise ValueError('Attempt output cannot replace controller ownership state')
    with lock(control/'owner.lock'):
        owner_path = control/'owner.json'; previous = c['previous_app_id']
        if owner_path.exists():
            owner = load(owner_path)
            if owner['status'] != 'measured' or not owner.get('allocation_stopped') or not owner.get('app_id'):
                raise ValueError('Previous execution needs explicit evidence review; no resubmission or follow-on query')
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
        if verify_experiment(output/'campaign/query') != request:
            raise ValueError('Prepared query changed while snapshotting')
        shutil.copyfile(plan['runtime_path'],output/'runtime.json')
        if sha256(output/'runtime.json') != c['runtime_sha256']:
            raise ValueError('Authorized runtime changed while snapshotting')
        policy = dict(experiments=['query'],timeout_seconds=c['app_timeout_seconds'],
            hourly_reservation_usd=c['hourly_reservation_usd'],budget_usd=plan['reserved_cents']/100,
            budget_stage=c['budget_stage'],spectra=spectra,expected_source_code=experiment_code())
        if 'volume' in c:policy['volume'] = c['volume']
        save(output/'campaign/execution.json',policy)
        save(output/'authorization.json',dict(config=c,config_sha256=plan['sha256'],prior_app=prior))
        started = time.monotonic()
        result = dict(format='REMAG-QUERY-EXECUTION/1',status='submitting',config_sha256=plan['sha256'],
            source_sha256=request['source_sha256'],request_sha256=sha256(bundle/'experiment.json'),
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
        flush()
        return result
