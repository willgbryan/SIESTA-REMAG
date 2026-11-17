import pytest
from test_research_relaxation_execution import discovery, DiscoveryTransport, run
from test_research_execution import SyntheticTransport
from siesta_remag.discovery import pipeline
from siesta_remag.discovery.common import load,save
from siesta_remag.corpus import sha256
from siesta_remag.research.execution import review_failed_runtime_for_diagnostic,execute_relaxation
from siesta_remag.research.budget import BudgetLedger


def held(root,monkeypatch,code=-11):
    original=pipeline.run_dft;t=DiscoveryTransport(monkeypatch)
    def failed_native(case,exe,output,**kwargs):
        output.mkdir(parents=True)
        if code is not None:save(output/'execution-status.json',dict(returncode=code,timed_out=False,wall_seconds=1.0))
        raise ValueError('Retained post-execution audit failure fixture')
    monkeypatch.setattr(pipeline,'run_dft',lambda *a,**kw:original(*a,**kw,runner=failed_native))
    result=run(root,t,'failed');assert result['status']=='held_review'
    monkeypatch.setattr(pipeline,'run_dft',original)
    config=load(root/'config.json');runtime=load(root/'runtime.json')
    (root/'new-binary').write_text('changed fixture, never executed')
    runtime.update(executable=str(root/'new-binary'),executable_sha256=sha256(root/'new-binary'))
    save(root/'new-runtime.json',runtime)
    config.update(authorization_id='changed-runtime',runtime='new-runtime.json',runtime_sha256=sha256(root/'new-runtime.json'),max_queries=1)
    save(root/'next.json',config);save(root/'diagnosis.json',dict(cause='unresolved',new_runtime='profile guard'))


def review(root,**kwargs):
    return review_failed_runtime_for_diagnostic(root,'next.json','bundle',structure='structure.json',
        evidence=['diagnosis.json'],reason='Explicit authorized changed-runtime validation; preserve the failed attempt without a causal repair claim.',
        transport=SyntheticTransport(),**kwargs)


def test_review_binds_relaxation_source_preserves_failure_and_admits_one_case(discovery,tmp_path,monkeypatch):
    held(tmp_path,monkeypatch)
    original=sha256(tmp_path/'failed/execution.json');owner=(tmp_path/'.research-execution/owner.json').read_bytes()
    budget=BudgetLedger(tmp_path/'budget.json').snapshot()
    row=review(tmp_path)
    assert row['operation']=='relaxation' and row['next_source_binding']['source_sha256']
    assert (tmp_path/'.research-execution/owner.json').read_bytes()==owner
    assert BudgetLedger(tmp_path/'budget.json').snapshot()==budget
    result=execute_relaxation(tmp_path,'next.json','structure.json','bundle','validated',transport=DiscoveryTransport(monkeypatch))
    assert result['status']=='measured' and sha256(tmp_path/'failed/execution.json')==original
    with pytest.raises(ValueError,match='limit reached'):
        execute_relaxation(tmp_path,'next.json','structure.json','bundle','again',transport=DiscoveryTransport(monkeypatch))


@pytest.mark.parametrize('code',[None,0])
def test_review_error_alone_does_not_establish_native_failure(discovery,tmp_path,monkeypatch,code):
    held(tmp_path,monkeypatch,code)
    with pytest.raises(ValueError,match='native.*failure'):
        review(tmp_path)
    assert not (tmp_path/'.research-execution/diagnostic-reviews').exists()


def test_relaxation_review_rejects_unchanged_runtime(discovery,tmp_path,monkeypatch):
    held(tmp_path,monkeypatch)
    save(tmp_path/'new-runtime.json',load(tmp_path/'runtime.json'))
    c=load(tmp_path/'next.json');c['runtime_sha256']=sha256(tmp_path/'new-runtime.json');save(tmp_path/'next.json',c)
    with pytest.raises(ValueError,match='changed executable'):
        review(tmp_path)


@pytest.mark.parametrize('field',['operation','next_source_binding'])
def test_relaxation_review_cannot_be_reused_with_wrong_source_or_operation(discovery,tmp_path,monkeypatch,field):
    held(tmp_path,monkeypatch);review(tmp_path)
    path=next((tmp_path/'.research-execution/diagnostic-reviews').glob('*.json'))
    row=load(path);row[field]='query' if field=='operation' else None;save(path,row)
    budget=BudgetLedger(tmp_path/'budget.json').snapshot();transport=DiscoveryTransport(monkeypatch)
    with pytest.raises(ValueError,match='does not match'):
        execute_relaxation(tmp_path,'next.json','structure.json','bundle','denied',transport=transport)
    assert not transport.calls and not (tmp_path/'denied').exists()
    assert BudgetLedger(tmp_path/'budget.json').snapshot()==budget
