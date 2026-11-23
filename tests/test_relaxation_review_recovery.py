import pytest
from test_research_relaxation_execution import discovery, DiscoveryTransport, run
from siesta_remag.discovery.common import load,save
from siesta_remag.corpus import sha256
from siesta_remag.research import relaxation_execution
from siesta_remag.research.relaxation_recovery import recover_saved_review,verified_execution,NAME
from siesta_remag.research.discovery_observation import import_relaxation_result,verified_observation


def held(root,monkeypatch):
    original=relaxation_execution.review_relaxation
    def fail(*a,**k):raise ValueError('Synthetic serialization review error; native fixture succeeded')
    monkeypatch.setattr(relaxation_execution,'review_relaxation',fail)
    t=DiscoveryTransport(monkeypatch);assert run(root,t)['status']=='held_review'
    monkeypatch.setattr(relaxation_execution,'review_relaxation',original)
    return t


def test_saved_review_preserves_original_imports_and_allows_separate_case(discovery,tmp_path,monkeypatch):
    t=held(tmp_path,monkeypatch);p=tmp_path/'attempt'
    before={n:sha256(tmp_path/n) for n in ['attempt/execution.json','attempt/collected/archive-manifest.json','budget.json','.research-execution/owner.json']}
    row=recover_saved_review(p,reason='Local serialization validator repaired; full saved evidence independently rechecked.')
    assert row['status']=='measured' and row['original_status']=='held_review' and len(t.native)==4
    assert row['usable_result_seconds']>=load(p/'execution.json')['request_through_review_seconds']
    assert before=={n:sha256(tmp_path/n) for n in before}
    assert recover_saved_review(p,reason='Repeated read must not replace the append-only review.')==row
    result=import_relaxation_result(p)
    observed=verified_observation(p/'research-observation/report.json')
    assert observed['status']=='measured' and NAME in observed['provenance']['attempt_files']
    assert 'through saved-only recovery' in observed['latency_scope']
    assert run(tmp_path,t,'separate')['status']=='measured'
    assert len(t.native)==8 and sha256(p/'execution.json')==before['attempt/execution.json']


@pytest.mark.parametrize('tamper',['summary','original','time'])
def test_recovery_tampering_is_rejected_before_import_or_new_execution(discovery,tmp_path,monkeypatch,tamper):
    t=held(tmp_path,monkeypatch);p=tmp_path/'attempt'
    recover_saved_review(p,reason='Local serialization validator repaired; full saved evidence independently rechecked.')
    path=p/(NAME if tamper!='original' else 'execution.json');d=load(path)
    if tamper=='summary':d['review']['screen']['Js_T']+=.1
    elif tamper=='time':d['request_through_recovery_seconds']=0
    else:d['source_sha256']='0'*64
    save(path,d);before=sha256(tmp_path/'budget.json')
    with pytest.raises(ValueError):verified_execution(p)
    with pytest.raises(ValueError):import_relaxation_result(p)
    with pytest.raises(ValueError):run(tmp_path,t,'denied')
    assert len(t.calls)==1 and sha256(tmp_path/'budget.json')==before
