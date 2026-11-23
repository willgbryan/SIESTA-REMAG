import pytest
from test_research_execution import prepared,SyntheticTransport
from siesta_remag.discovery.common import load,save
from siesta_remag.corpus import sha256
from siesta_remag.research.execution import execute_query,review_failed_runtime_for_diagnostic
from siesta_remag.research.budget import BudgetLedger


def held(root):
    result=execute_query(root,'execution-config.json','bundle','failed',transport=SyntheticTransport(failure='native'))
    assert result['status']=='held_native_review'
    config=load(root/'execution-config.json');runtime=load(root/'runtime.json')
    (root/'new-binary').write_text('changed synthetic executable; never executed')
    runtime.update(executable=str(root/'new-binary'),executable_sha256=sha256(root/'new-binary'))
    save(root/'new-runtime.json',runtime)
    config.update(authorization_id='diagnostic',runtime='new-runtime.json',runtime_sha256=sha256(root/'new-runtime.json'),max_queries=1)
    save(root/'next.json',config);save(root/'diagnosis.json',dict(cause='unresolved',instrumentation='new'))


def review(root,transport=None):
    return review_failed_runtime_for_diagnostic(root,'next.json','bundle',evidence=['diagnosis.json'],
        reason='Explicit owner-approved diagnostic after retained failure review; no causal repair claimed.',transport=transport or SyntheticTransport())


def test_explicit_review_preserves_failure_and_allows_only_one_bound_diagnostic(prepared,tmp_path):
    held(tmp_path);frozen=sha256(tmp_path/'failed/execution.json');owner=load(tmp_path/'.research-execution/owner.json')
    before=BudgetLedger(tmp_path/'budget.json').snapshot();row=review(tmp_path)
    assert row['native_fix_claimed'] is False
    assert load(tmp_path/'.research-execution/owner.json')==owner
    assert BudgetLedger(tmp_path/'budget.json').snapshot()==before
    result=execute_query(tmp_path,'next.json','bundle','diagnostic',transport=SyntheticTransport())
    assert result['status']=='measured' and sha256(tmp_path/'failed/execution.json')==frozen
    assert load(tmp_path/'diagnostic/authorization.json')['diagnostic_review']
    with pytest.raises(ValueError,match='limit reached'):
        execute_query(tmp_path,'next.json','bundle','second',transport=SyntheticTransport())


@pytest.mark.parametrize('change',['same_runtime','active','uncertain','evidence_changed','wrong_bundle'])
def test_review_cannot_release_unchanged_uncertain_or_tampered_attempt(prepared,tmp_path,change):
    held(tmp_path)
    if change=='same_runtime':
        save(tmp_path/'new-runtime.json',load(tmp_path/'runtime.json'))
        c=load(tmp_path/'next.json');c['runtime_sha256']=sha256(tmp_path/'new-runtime.json');save(tmp_path/'next.json',c)
        with pytest.raises(ValueError,match='changed executable'):review(tmp_path)
    elif change=='active':
        with pytest.raises(ValueError,match='STOPPED'):review(tmp_path,SyntheticTransport(failure='prior_active'))
    elif change=='uncertain':
        p=tmp_path/'.research-execution/owner.json';o=load(p);o['allocation_stopped']=False;save(p,o)
        with pytest.raises(ValueError,match='stopped native-review'):review(tmp_path)
    else:
        review(tmp_path)
        if change=='evidence_changed':save(tmp_path/'diagnosis.json',dict(changed=True))
        else:
            p=tmp_path/'bundle/experiment.json';r=load(p);r['parameter']='changed';save(p,r)
        transport=SyntheticTransport()
        with pytest.raises(ValueError,match='evidence changed|does not match'):
            execute_query(tmp_path,'next.json','bundle','diagnostic',transport=transport)
        assert not transport.calls and not (tmp_path/'diagnostic').exists()
    assert len(BudgetLedger(tmp_path/'budget.json').snapshot()['reservations'])==1
