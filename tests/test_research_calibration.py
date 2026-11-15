from copy import deepcopy

import pytest

from test_research_convergence import report
from siesta_remag.discovery.common import save
from siesta_remag.research.calibration import calibrate_response


def tier(tmp_path,name,energies):
    paths=[]
    for i,e in enumerate(energies):
        r=report(8 if name=='cheap' else 24,e,e-.0001)
        r['source_sha256']=r['request']['source_sha256']=str(i)*64
        r['artifacts']={}
        p=tmp_path/name/str(i)/'report.json';save(p,r);paths.append(p)
    return paths


def test_calibration_detects_response_sign_and_rank_errors(tmp_path):
    a=tier(tmp_path,'cheap',[.002,.003,.001])
    b=tier(tmp_path,'reference',[.003,.004,.0035])
    r=calibrate_response(a,b,baseline_source_sha256='0'*64,resolution_ev=.00001)
    q=r['observables']['total_energy_ev']
    assert q['resolved_sign_agreements']==1 and q['resolved_sign_disagreements']==1
    assert q['max_abs_response_error_ev']==pytest.approx(.0015)
    assert set(r['changed_settings'])=={'kmesh'}
    assert not r['latency_target_demonstrated']


def test_unresolved_response_is_never_a_sign_success(tmp_path):
    a=tier(tmp_path,'cheap',[.002,.002001,.001])
    b=tier(tmp_path,'reference',[.003,.003002,.002])
    r=calibrate_response(a,b,baseline_source_sha256='0'*64,resolution_ev=.00001)
    assert r['observables']['total_energy_ev']['unresolved_signs']==1
    assert r['observables']['total_energy_ev']['resolved_sign_agreements']==1
    with pytest.raises(ValueError,match='3..100'):calibrate_response(a[:1],b[:1],baseline_source_sha256='0'*64,resolution_ev=.00001)


def test_mismatched_sources_and_native_failures_rejected(tmp_path):
    a=tier(tmp_path,'cheap',[.002,.003,.001]);b=tier(tmp_path,'reference',[.003,.004,.002])
    from siesta_remag.discovery.common import load
    r=load(b[-1]);r['source_sha256']=r['request']['source_sha256']='a'*64;save(b[-1],r)
    with pytest.raises(ValueError,match='matching structures'):calibrate_response(a,b,baseline_source_sha256='0'*64,resolution_ev=.00001)
    r['stages'][-1]['accepted']=False;save(b[-1],r)
    with pytest.raises(ValueError,match='accepted directional'):calibrate_response(a,b,baseline_source_sha256='0'*64,resolution_ev=.00001)
