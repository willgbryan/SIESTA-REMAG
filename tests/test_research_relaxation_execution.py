"""Controller + actual remote function with synthetic native transport only.

Ni/Co fixtures exercise arbitrary species, not material predictions or DFT.
No model, cloud, GPU, Orb weights or historical campaign ledger is used.
"""
import ast
import os
from pathlib import Path
import types
import pytest
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import write
from test_discovery import backend
from test_research_execution import SyntheticTransport
from siesta_remag.corpus import sha256
from siesta_remag.discovery import pipeline
from siesta_remag.discovery.collection import pack, unpack
from siesta_remag.discovery.common import digest, load, save
from siesta_remag.discovery.config import Settings
from siesta_remag.discovery.inputs import kmesh
from siesta_remag.research.budget import BudgetLedger
from siesta_remag.research.execution import execute_query, execute_relaxation
from siesta_remag.research.relaxation_execution import preflight_relaxation, verify_relaxation_source

@pytest.fixture(params=['Ni', 'Co'])
def discovery(tmp_path, request):
    pytest.importorskip('gpac')
    from gpac import ir, physical
    symbol = request.param
    program = ir.author_program(dict(hall=1, gram=[[9,0,0],[0,9,0],[0,0,9]],
        orbits=[dict(z=atomic_numbers[symbol], wyckoff_index=0, parameters=[x]*3) for x in [0,'1/2']]))
    save(tmp_path/'structure.json', physical.export_geometry(program, length_unit='angstrom'))
    b = tmp_path/'bundle'; (b/'pseudos').mkdir(parents=True); (b/'orb').mkdir()
    pseudo = b/'pseudos'/f'{symbol}.psml'
    pseudo.write_text(f'<psml><pseudo-atom-spec atomic-label="{symbol}" relativity="dirac"><exchange-correlation><functional id="101"/><functional id="130"/></exchange-correlation></pseudo-atom-spec></psml>')
    atoms = Atoms(symbol+'2', scaled_positions=[[0,0,0],[.5,.5,.5]], cell=[3,3,3], pbc=True)
    atoms.set_initial_magnetic_moments([2., 2.])
    write(b/'input.extxyz', atoms); write(b/'orb/relaxed.extxyz', atoms)
    save(b/'orb/result.json', dict(accepted=True))
    settings = Settings(total_dft_seconds=30, dft_seconds=10, soc_seconds=5)
    save(b/'request.json', dict(settings=settings.to_dict(), input_sha256=sha256(b/'input.extxyz'),
                               pseudopotentials={pseudo.name:sha256(pseudo)}))
    refresh_bundle(b)
    binary = tmp_path/'binary'; binary.write_text('synthetic, never executed')
    save(tmp_path/'runtime.json', dict(executable=str(binary), executable_sha256=sha256(binary), environment={}, threads=1))
    ledger = BudgetLedger(tmp_path/'budget.json'); ledger.create(100, authorization='synthetic')
    ledger.define_stage('queries', 100, authorization='synthetic')
    save(tmp_path/'config.json', dict(format='REMAG-RESEARCH-EXECUTION/1', operation='relaxation',
        authorization_id='synthetic', authorization='synthetic test only', budget_ledger='budget.json',
        budget_stage='queries', runtime='runtime.json', runtime_sha256=sha256(tmp_path/'runtime.json'),
        image='im-fixture', previous_app_id='ap-previous', app_timeout_seconds=120,
        hourly_reservation_usd=3., max_queries=2, max_reserved_cents=20))
    return b

def refresh_bundle(b):
    from ase.io import read
    request = load(b/'request.json')
    files = ['request.json', 'input.extxyz', 'orb/result.json', 'orb/relaxed.extxyz'] + ['pseudos/'+s for s in request['pseudopotentials']]
    save(b/'prepared.json', dict(status='prepared', request_sha256=digest(request),
        files={name:sha256(b/name) for name in files}, orb_seconds=1,
        kmesh=kmesh(read(b/'orb/relaxed.extxyz'), Settings.from_dict(request['settings']))))

class DiscoveryTransport(SyntheticTransport):
    def __init__(self, monkeypatch, *, relaxed_cell_scale=1., **kwargs):
        super().__init__(**kwargs)
        original = pipeline.run_dft
        base = backend(self.native, fail='optimizer' if self.failure in ('optimizer', 'step_limit') else
                       'soc-x' if self.failure == 'native' else None, branch=self.failure == 'branch')
        def native(case, executable, destination, **kw):
            result = base(case, executable, destination, **kw)
            m = load(destination/'manifest.json'); g = m['geometry']
            if self.failure == 'step_limit':
                with (destination/'stdout.log').open('a') as stream:
                    for move in range(m['settings']['dft_steps']+1):
                        stream.write(f'\nBegin Broyden opt. move = {move}\n')
                save(destination/'execution-status.json', dict(returncode=0, timed_out=False))
            cell = [[v*(relaxed_cell_scale if destination.name == 'dft-relax' else 1) for v in row]
                    for row in g['lattice_A']]
            (destination/'remag.STRUCT_OUT').write_text('\n'.join(' '.join(map(str,r)) for r in cell)+
                '\n2\n'+'\n'.join('1 '+str(atomic_numbers[s])+' '+' '.join(map(str,r))
                                   for s,r in zip(g['species'],g['fractional_coordinates']))+'\n')
            # Retention fixture only; not a basis accepted by electronic tools.
            (destination/(g['species'][0]+'.ion.xml')).write_text('<ion>synthetic retention fixture</ion>')
            return result
        monkeypatch.setattr(pipeline, 'run_dft', lambda *a, **kw: original(*a, **kw, runner=native))

    def launch(self, attempt, config):
        self.calls.append(attempt)
        if self.failure == 'unknown': raise OSError('No receipt')
        policy = load(attempt/'campaign/execution.json')
        run_id = 'fixture-'+attempt.name
        ledger = BudgetLedger(config['budget_ledger'])
        ledger.reserve(run_id, 10, stage='queries', metadata=dict(receipt=str(attempt/'receipt.json'),
                       app_id='ap-'+attempt.name, policy_sha256=digest(policy)))
        save(attempt/'receipt.json', dict(app_id='ap-'+attempt.name, run_id=run_id))
        if self.failure == 'client': raise OSError('Lost collection')
        results = attempt/'remote-results'; results.mkdir()
        campaign = attempt/'campaign'
        def path(value):
            return {'/opt/research/campaign':campaign, '/opt/research/campaign/execution.json':campaign/'execution.json',
                    '/opt/research/runtime.json':attempt/'runtime.json', '/results':results}.get(str(value), Path(value))
        tree = ast.parse((Path(__file__).parents[1]/'scripts/research_modal.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'calculate'); fn.decorator_list=[]
        ns = dict(Path=path, os=types.SimpleNamespace(environ=dict(os.environ)),
                  load=lambda v:load(path(v)), save=save, digest=digest,
                  volume=types.SimpleNamespace(commit=lambda:None))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'remote-calculate', 'exec'), ns)
        payload = ns['calculate'](run_id)
        if self.failure in ('review', 'runtime', 'screen', 'missing_native', 'input'):
            root = results/run_id; report = load(root/'query/report.json')
            if self.failure == 'review': report['stages'][0]['max_force_ev_ang'] = 42
            if self.failure == 'runtime': report['runtime_env'] = {'REMAG_FAKE':'1'}
            if self.failure == 'screen': report['screen']['signal'] = 'forged'
            if self.failure == 'missing_native': (root/'query/dft-relax/remag.DM').unlink()
            if self.failure == 'input':
                folder = root/'query/dft-relax'; fdf=folder/'input.fdf'
                fdf.write_text(fdf.read_text().replace('MeshCutoff 300', 'MeshCutoff 301'))
                m=load(folder/'manifest.json'); m['input_sha256']=sha256(fdf); save(folder/'manifest.json',m)
                report['stages'][0]['artifact_sha256']['input.fdf']=sha256(fdf)
                report['accepted_artifacts']['dft-relax/input.fdf']=sha256(fdf)
            save(root/'query/report.json', report)
            top = load(root/'report.json'); top['experiments'][0]['report_sha256'] = sha256(root/'query/report.json')
            save(root/'report.json', top)
            payload = pack(root, include_arrays=True, include_native=True)
        unpack(payload, attempt/'collected')

def run(tmp_path, transport, output='attempt'):
    return execute_relaxation(tmp_path, 'config.json', 'structure.json', 'bundle', output, transport=transport)

def test_non_fept_remote_collection_feedback_and_shared_owner(discovery, tmp_path, monkeypatch):
    t = DiscoveryTransport(monkeypatch)
    before = (tmp_path/'structure.json').read_bytes()
    result = run(tmp_path, t)
    assert result['status'] == 'measured', result
    assert result['review']['native_status'] == 'screened'
    assert result['review']['feedback']['feedback']['symmetry']['all_authored_operations_preserved']
    assert result['review']['screen']['synthesizability_qualified'] is False
    assert result['review']['verification']['native_reviews_reproduced']
    assert result['review']['report'] == 'attempt/collected/query/report.json'
    assert (tmp_path/'attempt/collected/query/dft-relax/remag.DM').exists()
    assert (tmp_path/'structure.json').read_bytes() == before
    assert result['allocation_stopped'] and result['reservation_recorded_cents'] == 10
    assert result['actual_billed_usd'] is None and result['usable_result_seconds'] > 0
    second = run(tmp_path, t, 'second')
    assert second['status'] == 'measured', second
    assert t.info_calls == ['ap-previous', 'ap-attempt'] and len(t.native) == 8
    with pytest.raises(ValueError, match='limit reached'): run(tmp_path, t, 'third')
    assert len(t.calls) == 2

def test_negative_screen_is_a_completed_observation(discovery, tmp_path, monkeypatch):
    request = load(discovery/'request.json'); request['settings']['minimum_js_t'] = 100
    save(discovery/'request.json', request); refresh_bundle(discovery)
    t = DiscoveryTransport(monkeypatch); result = run(tmp_path, t)
    assert result['status'] == 'measured', result
    assert result['review']['native_status'] == 'screened_low_moment'
    assert result['review']['screen'] is None and len(t.native) == 1

@pytest.mark.parametrize('failure,status', [('optimizer','held_native_review'), ('native','held_native_review'),
    ('branch','held_native_review'), ('client','held_execution'), ('unknown','held_unknown_owner'),
    ('shutdown','held_shutdown'), ('review','held_review'), ('runtime','held_review'),
    ('screen','held_review'), ('missing_native','held_review'), ('input','held_review')])
def test_failed_attempt_is_retained_never_retried(discovery, tmp_path, monkeypatch, failure, status):
    t = DiscoveryTransport(monkeypatch, failure=failure); result = run(tmp_path, t)
    assert result['status'] == status, result
    assert result['usable_result_seconds'] is None and len(t.calls) == 1
    assert result['allocation_stopped'] == (failure not in ('shutdown', 'unknown'))
    with pytest.raises(ValueError, match='Previous execution needs'): run(tmp_path, t, 'next')
    assert len(t.calls) == 1
    assert sum(r['cents'] for r in BudgetLedger(tmp_path/'budget.json').snapshot()['reservations']) == (0 if failure == 'unknown' else 10)

def test_operation_grant_cannot_be_repurposed(discovery, tmp_path):
    t = SyntheticTransport()
    with pytest.raises(ValueError, match='operation is not authorized'):
        execute_query(tmp_path, 'config.json', 'bundle', 'attempt', transport=t)
    c = load(tmp_path/'config.json'); c.pop('operation'); save(tmp_path/'config.json', c)
    with pytest.raises(ValueError, match='operation is not authorized'): run(tmp_path, t)
    assert not t.calls and not t.info_calls

def test_mismatched_exact_source_is_rejected_before_ownership_or_reservation(discovery, tmp_path):
    from gpac import ir, physical
    p = ir.author_program(dict(hall=1, gram=[[16,0,0],[0,16,0],[0,0,16]],
        orbits=[dict(z=28, wyckoff_index=0, parameters=[0,0,0])]))
    save(tmp_path/'structure.json', physical.export_geometry(p, length_unit='angstrom'))
    t = SyntheticTransport(); before = (tmp_path/'budget.json').read_bytes()
    with pytest.raises(ValueError, match='Authored source differs'): run(tmp_path, t)
    assert not t.info_calls and not t.calls and not (tmp_path/'.research-execution').exists()
    assert (tmp_path/'budget.json').read_bytes() == before

def test_preflight_checks_compiled_capacity_before_account_calls(discovery, tmp_path):
    runtime = load(tmp_path/'runtime.json')
    runtime['environment'] = {'REMAG_DEVICE_SCF':'1', 'LD_LIBRARY_PATH':'/synthetic'}
    runtime['capabilities'] = dict(format='REMAG-BOUND-CAPABILITIES/1', executable_sha256=runtime['executable_sha256'],
        library='/synthetic/lib.so', library_sha256='a'*64, device_scf=dict(format='REMAG-DEVICE-SCF-CAPABILITIES/1',
        max_kpoints=128, max_spatial_orbitals=2048, max_spinor_dimension=4096))
    save(tmp_path/'runtime.json', runtime)
    c=load(tmp_path/'config.json'); c['runtime_sha256']=sha256(tmp_path/'runtime.json'); save(tmp_path/'config.json', c)
    t = SyntheticTransport()
    with pytest.raises(ValueError, match='compiled device-SCF limit is 128'): run(tmp_path, t)
    assert not t.info_calls and not t.calls

def test_read_only_preflight_and_input_psp_compatibility(discovery, tmp_path):
    result = preflight_relaxation(discovery, load(tmp_path/'runtime.json'))
    assert result['paid_actions'] == 0 and result['physical_conditions']['cell'] == 'fully_variable'
    assert result['magnetic_start'] == dict(moments_muB=[2., 2.], provenance='explicit')
    assert verify_relaxation_source(tmp_path/'structure.json', discovery)['source_sha256']
    pseudo = next((discovery/'pseudos').iterdir()); pseudo.write_text(pseudo.read_text().replace('dirac', 'scalar'))
    request = load(discovery/'request.json'); request['pseudopotentials'][pseudo.name]=sha256(pseudo)
    save(discovery/'request.json', request); refresh_bundle(discovery)
    with pytest.raises(ValueError, match='fully relativistic'): preflight_relaxation(discovery, load(tmp_path/'runtime.json'))

def test_cpu_preparation_cannot_lose_magnetic_start(discovery, tmp_path):
    from ase.io import read
    atoms=read(discovery/'orb/relaxed.extxyz'); atoms.set_initial_magnetic_moments([2., -2.])
    write(discovery/'orb/relaxed.extxyz', atoms); refresh_bundle(discovery)
    t=SyntheticTransport()
    with pytest.raises(ValueError, match='magnetic starting branch'): run(tmp_path, t)
    assert not t.calls and not t.info_calls

def test_previous_running_app_prevents_discovery_launch(discovery, tmp_path):
    t=SyntheticTransport(failure='prior_active')
    with pytest.raises(ValueError, match='authoritatively STOPPED'): run(tmp_path, t)
    assert not t.calls and not (tmp_path/'attempt').exists()

def test_broyden_input_audits_and_unknown_consumers_before_account(discovery, tmp_path, monkeypatch):
    runtime = load(tmp_path/'runtime.json')
    runtime['environment'] = {'REMAG_INTEGRAL_FORCE':'1', 'REMAG_GATE_H_EXPORT':'1', 'REMAG_GPU_HISTORY':'1'}
    save(tmp_path/'runtime.json', runtime)
    config = load(tmp_path/'config.json'); config['runtime_sha256'] = sha256(tmp_path/'runtime.json')
    save(tmp_path/'config.json', config)
    audit = preflight_relaxation(discovery, runtime)['input_audits']
    assert audit['hamiltonian_exports'] == 4 and audit['density_exports'] == 2
    assert audit['density_history'] == dict(depth=1, mode=2)
    from siesta_remag.research import relaxation_execution
    original = relaxation_execution.prepare_case
    def unsupported(*args, **kwargs):
        folder = original(*args, **kwargs)
        with (folder/'input.fdf').open('a') as f: f.write('\nOpticalCalculation true\n')
        return folder
    monkeypatch.setattr(relaxation_execution, 'prepare_case', unsupported)
    t = DiscoveryTransport(monkeypatch); before = (tmp_path/'budget.json').read_bytes()
    with pytest.raises(ValueError, match='analysis consumer'):
        run(tmp_path, t)
    assert not t.calls and not t.info_calls and not t.native
    assert not (tmp_path/'attempt').exists() and not (tmp_path/'.research-execution').exists()
    assert (tmp_path/'budget.json').read_bytes() == before
