"""Synthetic transport exercises real bundle/review/collection/ledger code.

No native executable, cloud API, GPU, model API or campaign ledger is used.
"""
from pathlib import Path
import sys
import pytest
from ase import Atoms
from test_discovery import backend
from siesta_remag.corpus import sha256
from siesta_remag.discovery.collection import pack,unpack
from siesta_remag.discovery.common import load,save,digest
from siesta_remag.discovery.config import Settings
from siesta_remag.discovery.inputs import prepare_case
from siesta_remag.research.budget import BudgetLedger
from siesta_remag.research.execution import execute_query,read_execution_config,ModalExecution
from siesta_remag.research.experiments import run_experiment
from siesta_remag.research.queries import requested_spectra

@pytest.fixture
def prepared(tmp_path):
    # Minimal synthetic boundary payload; GPAC's actual export boundary has
    # separate tests. The execution controller must work without GPAC installed.
    bundle=tmp_path/'bundle';bundle.mkdir();pseudo=bundle/'pseudos';pseudo.mkdir()
    (pseudo/'Fe.psf').write_text(' Fe pb rel pcec\nfixture only\n')
    atoms=Atoms('Fe2',scaled_positions=[[0,0,0],[.5,.5,.5]],cell=[3,3,3],pbc=True)
    settings=Settings(total_dft_seconds=30,soc_seconds=10)
    prepare_case(atoms,pseudo,bundle/'scalar',settings,mesh=[3,3,3],allow_psf=True,diagnostics=True,basis_size='DZP',basis_energy_shift_ry=.02)
    save(bundle/'structure.json',dict(sha256='b'*64,source=dict(sha256='a'*64),geometry=dict(
        atomic_numbers=atoms.numbers.tolist(),cell=atoms.cell.array.tolist(),fractional=atoms.get_scaled_positions().tolist())))
    request=dict(format='REMAG-EXPERIMENT/1',source_sha256='a'*64,export_sha256='b'*64,
        parameter=None,fixed_geometry=True,settings=settings.to_dict(),kmesh=[3,3,3],basis_size='DZP',
        basis_energy_shift_ry=.02,pseudopotentials={'Fe.psf':sha256(pseudo/'Fe.psf')},
        files={str(p.relative_to(bundle)):sha256(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'experiment.json',request)
    binary=tmp_path/'binary';binary.write_text('synthetic fixture; never executed')
    save(tmp_path/'runtime.json',dict(executable=str(binary),executable_sha256=sha256(binary),environment={},threads=1))
    ledger=BudgetLedger(tmp_path/'budget.json');ledger.create(100,authorization='synthetic test only')
    ledger.define_stage('queries',100,authorization='synthetic test only')
    config=dict(format='REMAG-RESEARCH-EXECUTION/1',authorization_id='fixture',authorization='synthetic test only',
        budget_ledger='budget.json',budget_stage='queries',runtime='runtime.json',runtime_sha256=sha256(tmp_path/'runtime.json'),
        image='im-fixture',previous_app_id='ap-previous',app_timeout_seconds=120,hourly_reservation_usd=3.,max_queries=2,max_reserved_cents=20)
    save(tmp_path/'execution-config.json',config)
    return bundle

class SyntheticTransport:
    def __init__(self,*,failure=None):
        self.failure=failure;self.calls=[];self.info_calls=[];self.stops=[];self.native=[]

    def info(self,app):
        self.info_calls.append(app)
        return dict(app_id=app,lifecycle=dict(state='running' if self.failure=='prior_active' else 'stopped'))

    def launch(self,attempt,config):
        self.calls.append(attempt)
        if self.failure=='unknown':raise OSError('Client lost before receipt')
        policy=load(attempt/'campaign/execution.json');ledger=BudgetLedger(config['budget_ledger'])
        run_id='fixture-'+attempt.name; app='ap-'+attempt.name
        ledger.reserve(run_id,10,metadata=dict(receipt=str(attempt/'receipt.json'),app_id=app,policy_sha256=digest(policy)),stage='queries')
        save(attempt/'receipt.json',dict(app_id=app,run_id=run_id,reserved_cents=10,function_call_id='fc-fixture'))
        if self.failure=='client':raise RuntimeError('Collection connection failed')
        if self.failure=='interrupt':raise KeyboardInterrupt()
        runtime=load(attempt/'runtime.json');remote=attempt/'synthetic-remote';remote.mkdir()
        r=run_experiment(attempt/'campaign/query',remote/'query',runtime['executable'],executable_sha256=runtime['executable_sha256'],
                         runtime_env=runtime['environment'],runner=backend(self.native,fail='soc-x' if self.failure=='native' else None))
        if self.failure=='runtime':r['environment']={'REMAG_DIFFERENT':'1'};save(remote/'query/report.json',r)
        top=dict(status='measured' if r['status']=='measured' else 'held_experiment',invocation_sha256=digest(policy),policy=policy,
            experiments=[dict(id='query',status=r['status'],report_sha256=sha256(remote/'query/report.json'))])
        save(remote/'report.json',top);unpack(pack(remote,include_arrays=True),attempt/'collected')

    def stopped(self,app):
        self.stops.append(app)
        if self.failure=='shutdown':raise TimeoutError('Authoritative observation unavailable')
        return dict(app_id=app,lifecycle=dict(state='stopped',stopped_at='fixture'))

def test_controlled_query_collects_reviews_and_next_query_rechecks_app(prepared,tmp_path):
    transport=SyntheticTransport()
    first=execute_query(tmp_path,'execution-config.json','bundle','first',transport=transport)
    assert first['status']=='measured',first
    assert first['allocation_stopped'] and first['reservation_recorded_cents']==10
    assert first['actual_billed_usd'] is None and first['usable_result_seconds']>0
    assert first['review']['report']=='first/collected/query/report.json'
    second=execute_query(tmp_path,'execution-config.json','bundle','second',transport=transport)
    assert second['status']=='measured' and transport.info_calls==['ap-previous','ap-first']
    assert transport.stops==['ap-first','ap-second'] and len(transport.native)==8
    ledger=BudgetLedger(tmp_path/'budget.json').snapshot()
    assert [r['cents'] for r in ledger['reservations']]==[10,10]
    with pytest.raises(ValueError,match='limit reached'):
        execute_query(tmp_path,'execution-config.json','bundle','third',transport=transport)
    assert len(transport.calls)==2

@pytest.mark.parametrize('failure,status', [('client','held_execution'),('interrupt','held_execution'),
    ('native','held_native_review'),('runtime','held_review'),('shutdown','held_shutdown'),('unknown','held_unknown_owner')])
def test_failure_never_resubmits_or_refunds(prepared,tmp_path,failure,status):
    transport=SyntheticTransport(failure=failure)
    r=execute_query(tmp_path,'execution-config.json','bundle','attempt',transport=transport)
    assert r['status']==status,r
    assert r['usable_result_seconds'] is None
    assert r['allocation_stopped'] == (failure not in ('shutdown','unknown'))
    assert len(transport.calls)==1
    assert len(transport.stops)==(0 if failure=='unknown' else 1)
    with pytest.raises(ValueError,match='Previous execution needs'):
        execute_query(tmp_path,'execution-config.json','bundle','next',transport=transport)
    assert len(transport.calls)==1
    rows=BudgetLedger(tmp_path/'budget.json').snapshot()['reservations']
    assert sum(r['cents'] for r in rows)==(0 if failure=='unknown' else 10)

def test_prior_active_or_changed_runtime_prevents_launch(prepared,tmp_path):
    transport=SyntheticTransport(failure='prior_active')
    with pytest.raises(ValueError,match='authoritatively STOPPED'):
        execute_query(tmp_path,'execution-config.json','bundle','attempt',transport=transport)
    assert not transport.calls and not (tmp_path/'attempt').exists()
    (tmp_path/'runtime.json').write_text('{}')
    with pytest.raises(ValueError,match='runtime changed'):read_execution_config(tmp_path,'execution-config.json')

def test_operator_configuration_cannot_be_reset(prepared,tmp_path):
    transport=SyntheticTransport();execute_query(tmp_path,'execution-config.json','bundle','first',transport=transport)
    c=load(tmp_path/'execution-config.json');c['max_queries']=3;save(tmp_path/'execution-config.json',c)
    with pytest.raises(ValueError,match='cannot be edited or reset'):
        execute_query(tmp_path,'execution-config.json','bundle','second',transport=transport)

def test_operator_volume_is_bound_to_policy_and_immutable_authorization(prepared,tmp_path):
    config=load(tmp_path/'execution-config.json');config['volume']='acceptance-isolated'
    save(tmp_path/'execution-config.json',config)
    result=execute_query(tmp_path,'execution-config.json','bundle','first',transport=SyntheticTransport())
    assert result['status']=='measured'
    policy=load(tmp_path/'first/campaign/execution.json')
    assert policy['volume']=='acceptance-isolated'
    reservation=BudgetLedger(tmp_path/'budget.json').snapshot()['reservations'][0]
    assert reservation['metadata']['policy_sha256']==digest(policy)
    config['volume']='another-volume';save(tmp_path/'execution-config.json',config)
    with pytest.raises(ValueError,match='cannot be edited or reset'):
        execute_query(tmp_path,'execution-config.json','bundle','second',transport=SyntheticTransport())

@pytest.mark.parametrize('volume',[None,False,'','../results','/results','bad volume','a'*65])
def test_invalid_operator_volume_prevents_launch(prepared,tmp_path,volume):
    config=load(tmp_path/'execution-config.json');config['volume']=volume
    save(tmp_path/'execution-config.json',config)
    with pytest.raises(ValueError,match='operator-selected result volume'):
        read_execution_config(tmp_path,'execution-config.json')

def test_requested_spectrum_missing_is_a_failed_query(prepared,tmp_path):
    r=execute_query(tmp_path,'execution-config.json','bundle','attempt',spectra=[dict(stage='soc-z',spin=0)],transport=SyntheticTransport())
    assert r['status']=='held_review' and r['allocation_stopped']
    assert r['usable_result_seconds'] is None

@pytest.mark.parametrize('selection',[[dict(stage='soc-z',spin=1)],[dict(stage='missing',spin=0)],
    [dict(stage='scalar',spin=True)],[dict(stage='scalar',spin=0)]*2,False])
def test_spectrum_selection_cannot_add_work(selection):
    with pytest.raises(ValueError):requested_spectra(dict(kind='scalar'),selection)

def test_explicit_scalar_and_soc_channels():
    rows=[dict(stage='scalar',spin=0),dict(stage='scalar',spin=1),dict(stage='soc-x',spin=0)]
    assert requested_spectra(dict(kind='anisotropy'),rows)==rows

def test_transient_stop_observation_is_not_release(monkeypatch):
    t=ModalExecution();events=[];responses=iter([TimeoutError(),dict(app_id='ap-owned',lifecycle=dict(state='running')),
        dict(app_id='ap-owned',lifecycle=dict(state='stopped'))])
    def info(app):
        events.append(('info',app));r=next(responses)
        if isinstance(r,Exception):raise r
        return r
    monkeypatch.setattr(t,'info',info);monkeypatch.setattr(t,'stop',lambda a:events.append(('stop',a)))
    monkeypatch.setattr('siesta_remag.research.execution.time.sleep',lambda _:None)
    assert t.stopped('ap-owned')['lifecycle']['state']=='stopped'
    assert events==[('info','ap-owned'),('stop','ap-owned'),('info','ap-owned'),('info','ap-owned')]

def test_production_transport_uses_only_shared_guard_and_fixed_entry(tmp_path,monkeypatch):
    calls=[]
    class Process:
        def __init__(self,command,**kwargs):calls.append((command,kwargs))
        def wait(self,timeout):return 0
    monkeypatch.setattr('siesta_remag.research.execution.subprocess.Popen',Process)
    ModalExecution(tmp_path).launch(tmp_path,dict(image='im-fixed',budget_ledger='/fixed/ledger',app_timeout_seconds=120))
    cmd,kw=calls[0]
    assert cmd[:4]==[sys.executable,'scripts/run_modal.py','--module-file','scripts/research_modal.py']
    assert kw['env']['REMAG_RESEARCH_BUDGET_LEDGER']=='/fixed/ledger'
    assert kw['start_new_session'] and len(calls)==1

def test_client_deadline_terminates_its_process_group_once(tmp_path,monkeypatch):
    import subprocess,signal
    events=[]
    class Process:
        pid=123
        def __init__(self,*args,**kwargs):self.waits=0
        def wait(self,timeout):
            self.waits+=1
            if self.waits==1:raise subprocess.TimeoutExpired('synthetic',timeout)
            return -15
    monkeypatch.setattr('siesta_remag.research.execution.subprocess.Popen',Process)
    monkeypatch.setattr('siesta_remag.research.execution.os.killpg',lambda *args:events.append(args))
    with pytest.raises(subprocess.TimeoutExpired):
        ModalExecution(tmp_path).launch(tmp_path,dict(image='im-fixed',budget_ledger='/fixed/ledger',app_timeout_seconds=120))
    assert events==[(123,signal.SIGTERM)]

def test_exhausted_existing_stage_prevents_any_account_or_launch_call(prepared,tmp_path):
    ledger=BudgetLedger(tmp_path/'budget.json')
    ledger.reserve('prior',100,metadata={'synthetic':True},stage='queries')
    transport=SyntheticTransport()
    with pytest.raises(ValueError,match='Insufficient existing stage'):
        execute_query(tmp_path,'execution-config.json','bundle','attempt',transport=transport)
    assert not transport.calls and not transport.info_calls
    assert ledger.snapshot()['reservations'][0]['cents']==100

@pytest.mark.parametrize('selection', [None,[dict(stage='scalar',spin=0),dict(stage='scalar',spin=1)],[dict(stage='soc-x',spin=0)]])
def test_remote_calculation_uses_selected_channels_and_retains_legacy_diagnostics(prepared,tmp_path,monkeypatch,selection):
    import ast,os,types
    from siesta_remag.research import experiments,electronic
    original=experiments.run_experiment;calls=[];exports=[]
    def run(*args,**kwargs):return original(*args,**kwargs,runner=backend(calls))
    def export(directory,output,*,collinear_spin):
        exports.append((directory.name,collinear_spin));output.mkdir();save(output/'electronic.json',dict(synthetic=True))
    monkeypatch.setattr(experiments,'run_experiment',run);monkeypatch.setattr(electronic,'spectrum_from_run',export)
    policy=dict(experiments=['query'],timeout_seconds=120,hourly_reservation_usd=3)
    if selection is not None:policy['spectra']=selection
    campaign=tmp_path/'remote-campaign';campaign.mkdir()
    import shutil
    shutil.copytree(prepared,campaign/'query');save(campaign/'execution.json',policy)
    results=tmp_path/'remote-results';results.mkdir()
    def path(value):
        return {'/opt/research/campaign':campaign,'/opt/research/campaign/execution.json':campaign/'execution.json',
                '/opt/research/runtime.json':tmp_path/'runtime.json','/results':results}.get(str(value),Path(value))
    module=ast.parse((Path(__file__).parents[1]/'scripts/research_modal.py').read_text())
    fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='calculate');fn.decorator_list=[]
    ns=dict(Path=path,os=types.SimpleNamespace(environ=dict(os.environ)),load=lambda value:load(path(value)),save=save,digest=digest,
            volume=types.SimpleNamespace(commit=lambda:None))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'remote-calculate','exec'),ns)
    payload=ns['calculate']('synthetic');unpack(payload,tmp_path/'collected')
    report=load(tmp_path/'collected/report.json')
    assert report['status']=='measured',report
    assert exports==([('soc-z',0),('soc-x',0),('soc-y',0)] if selection is None else [(r['stage'],r['spin']) for r in selection])
    assert report['timing']['native_seconds']>0 and report['timing']['diagnostics_seconds']>=0
    assert report['timing']['queue_seconds'] is None and report['timing']['startup_seconds'] is None


@pytest.fixture
def scalar_limit(prepared, monkeypatch, request):
    """Retain an actual synthetic collection, stopped owner and ledger entry."""
    import test_research_execution as suite
    root = prepared.parent
    original = backend
    reason = getattr(request, 'param', 'limit')
    selected = load(prepared/'experiment.json')
    save(prepared/'experiment.json', dict(selected, query={'kind':'scalar'}))

    def backend_factory(calls, **kwargs):
        wrapped = original(calls, **kwargs)
        def runner(source, exe, destination, **options):
            out = wrapped(source, exe, destination, **options)
            code = -11 if reason == 'segfault' else -6
            msg = 'SCF_NOT_CONV: SCF did not converge in maximum number of steps (required).'
            (destination/'stderr.log').write_text('other abort\n' if reason=='other' else msg+'\n')
            (destination/'stdout.log').write_text((destination/'stdout.log').read_text().replace('SCF cycle converged','SCF_NOT_CONV'))
            save(destination/'execution-status.json', dict(returncode=code, timed_out=reason=='timeout'))
            out.update(valid=False,returncode=code)
            out['observables'].update(converged=False,scf_iterations=50 if reason=='early' else selected['settings']['max_scf'])
            save(destination/'result.json',out)
            return out
        return runner

    transport = SyntheticTransport()
    with monkeypatch.context() as patch:
        patch.setattr(suite, 'backend', backend_factory)
        first = execute_query(root,'execution-config.json','bundle','failed',transport=transport)
    assert first['status']=='held_native_review' and first['allocation_stopped']
    config = load(root/'execution-config.json')
    config.update(authorization_id='independent', operation='query', max_queries=1, max_reserved_cents=10,
        previous_app_id=first['app_id'], previous_scalar_scf_failure_report_sha256=first['review']['report_sha256'])
    save(root/'separate.json',config)
    return root, transport


def independent_ni_bundle(root):
    src = load(root/'bundle/structure.json')
    src.update(source=dict(sha256='c'*64),sha256='d'*64)
    src['geometry']['atomic_numbers'] = [28,28]
    folder = root/'ni'
    pseudos = folder/'pseudos'
    pseudos.mkdir(parents=True)
    (pseudos/'Ni.psf').write_text(' Ni pb rel pcec\nfixture only\n')
    atom = Atoms('Ni2', scaled_positions=src['geometry']['fractional'], cell=src['geometry']['cell'], pbc=True)
    req = load(root/'bundle/experiment.json')
    prepare_case(atom,pseudos,folder/'scalar',Settings.from_dict(req['settings']),mesh=req['kmesh'],allow_psf=True,
                 diagnostics=True,basis_size=req['basis_size'],basis_energy_shift_ry=req['basis_energy_shift_ry'])
    save(folder/'structure.json',src)
    save(folder/'experiment.json',dict(req,source_sha256=src['source']['sha256'],export_sha256=src['sha256'],
        pseudopotentials={'Ni.psf':sha256(pseudos/'Ni.psf')}, query={'kind':'anisotropy'},
        files={str(p.relative_to(folder)):sha256(p) for p in folder.rglob('*') if p.is_file()}))
    return folder


def test_stopped_scalar_limit_permits_one_independent_query(scalar_limit):
    root, transport = scalar_limit
    bundle = independent_ni_bundle(root)
    with pytest.raises(ValueError,match='different chemical composition'):
        execute_query(root,'separate.json','bundle','repeated',transport=transport)
    transport.failure='prior_active'
    with pytest.raises(ValueError,match='authoritatively STOPPED'):
        execute_query(root,'separate.json',bundle,'overlap',transport=transport)
    transport.failure=None
    result = execute_query(root,'separate.json',bundle,'independent',transport=transport)
    assert result['status']=='measured' and result['allocation_stopped']
    assert len(transport.calls)==2
    assert load(root/'independent/authorization.json')['diagnostic_review']=='failed/collected/query/report.json'
    assert load(root/'failed/execution.json')['status']=='held_native_review'
    assert len(BudgetLedger(root/'budget.json').snapshot()['reservations'])==2
    with pytest.raises(ValueError):
        execute_query(root,'separate.json',bundle,'another',transport=transport)
    assert len(transport.calls)==2


@pytest.mark.parametrize('scalar_limit',['segfault','other','timeout','early'],indirect=True)
def test_independent_query_requires_native_scalar_limit(scalar_limit):
    root, transport = scalar_limit
    with pytest.raises(ValueError,match='SCF limit'):
        execute_query(root,'separate.json',independent_ni_bundle(root),'independent',transport=transport)
    assert len(transport.calls)==1 and not (root/'independent').exists()


def test_scalar_limit_grant_checks_actual_native_species(scalar_limit):
    root, transport = scalar_limit
    bundle = independent_ni_bundle(root)
    fdf = bundle/'scalar/input.fdf'
    fdf.write_bytes((root/'bundle/scalar/input.fdf').read_bytes())
    request = load(bundle/'experiment.json')
    request['files']['scalar/input.fdf'] = sha256(fdf)
    save(bundle/'experiment.json',request)
    with pytest.raises(ValueError,match='authored species differ'):
        execute_query(root,'separate.json',bundle,'independent',transport=transport)
    assert len(transport.calls)==1 and not (root/'independent').exists()


@pytest.mark.parametrize('change', [dict(max_queries=2),dict(operation='relaxation'),
    dict(previous_scf_failure_report_sha256='a'*64),dict(previous_scalar_scf_failure_report_sha256='invalid')])
def test_scalar_limit_grant_rejects_incompatible_config(prepared, change):
    root=prepared.parent
    config=load(root/'execution-config.json')
    config.update(operation='query',max_queries=1,previous_scalar_scf_failure_report_sha256='a'*64)
    config.update(change)
    save(root/'separate.json',config)
    with pytest.raises(ValueError):
        read_execution_config(root,'separate.json')
