from copy import deepcopy

import pytest

from test_research_convergence import report
from siesta_remag.discovery.common import save
from siesta_remag.research.calibration import calibrate_response
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load
from siesta_remag.discovery.collection import pack, unpack
from siesta_remag.research.toolhost import ToolHost
from siesta_remag.research.tools import ResearchTools


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


def compact(paths, output):
    """Package synthetic directional fixtures through the production collector."""
    endpoints=[]
    for i,path in enumerate(paths):
        r=load(path);r['invocation_sha256']='f'*64
        binary=path.parent/'remote.DM';binary.write_bytes(b'synthetic omitted native binary')
        r['artifacts']['remote.DM']=sha256(binary)
        for stage in r['stages']:
            for name in ('result.json','review.json','stdout.log','input.fdf','fdf-1.log'):
                p=path.parent/stage['stage']/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text('{}\n' if name.endswith('.json') else 'synthetic fixture\n')
                r['artifacts'][str(p.relative_to(path.parent))]=sha256(p)
        save(path,r)
        destination=output/str(i);unpack(pack(path.parent),destination)
        endpoints.append(dict(report=destination/'report.json',collection=destination/'archive-manifest.json'))
    return endpoints


def test_compact_tiers_match_full_results_and_retain_omissions(tmp_path):
    a=tier(tmp_path,'cheap',[.002,.003,.001]);b=tier(tmp_path,'reference',[.003,.004,.0035])
    ca=compact(a,tmp_path/'collected-cheap');cb=compact(b,tmp_path/'collected-reference')
    full=calibrate_response(a,b,baseline_source_sha256='0'*64,resolution_ev=.00001)
    collected=calibrate_response(ca,cb,baseline_source_sha256='0'*64,resolution_ev=.00001)
    assert collected['observables']==full['observables']
    assert collected['changed_settings']==full['changed_settings']
    for row in collected['exploratory'].values():
        assert row['verification']['omitted_artifacts']==['remote.DM']
        assert row['verification']['mode']=='compact collection'
        assert len(row['verification']['collection_manifest_sha256'])==64
    mixed=calibrate_response([a[0],ca[1],ca[2]],cb,baseline_source_sha256='0'*64,resolution_ev=.00001)
    assert mixed['observables']==full['observables']
    # No implicit collection discovery and no silent missing-byte exemption.
    with pytest.raises(FileNotFoundError):
        calibrate_response([e['report'] for e in ca],cb,baseline_source_sha256='0'*64,resolution_ev=.00001)


@pytest.mark.parametrize('failure',['changed_artifact','wrong_collection','failed_branch'])
def test_compact_corruption_and_native_failures_stop_calibration(tmp_path,failure):
    a=tier(tmp_path,'cheap',[.002,.003,.001]);b=tier(tmp_path,'reference',[.003,.004,.002])
    if failure=='failed_branch':
        r=load(a[0]);r['stages'][-1]['branch']={'passed':False};save(a[0],r)
    ca=compact(a,tmp_path/'collected');expected='accepted directional'
    if failure=='changed_artifact':
        (ca[0]['report'].parent/'soc-x/stdout.log').write_text('corruption');expected='artifact binding'
    if failure=='wrong_collection':
        ca[0]['collection']=ca[1]['collection'];expected='outside the collection'
    with pytest.raises(ValueError,match=expected):
        calibrate_response(ca,b,baseline_source_sha256='0'*64,resolution_ev=.00001)


def test_direct_tools_accept_archives_without_losing_workspace_or_holdout_gates(tmp_path):
    a=tier(tmp_path,'cheap',[.002,.003,.001]);b=tier(tmp_path,'reference',[.003,.004,.002])
    ca=compact(a,tmp_path/'collected-cheap');cb=compact(b,tmp_path/'collected-reference')
    relative=lambda points:[{k:str(v.relative_to(tmp_path)) for k,v in e.items()} for e in points]
    args=dict(exploratory=relative(ca),reference=relative(cb),baseline_source_sha256='0'*64,resolution_ev=.00001)
    response=ToolHost(tmp_path).call(dict(tool='calibrate_exploratory_response',args=args,
        rationale='Inspect saved synthetic calibration through the chat interface'),'calls/calibration')
    assert response['ok'] and not response['result']['latency_target_demonstrated']
    with pytest.raises(ValueError,match='held-out'):
        ResearchTools(tmp_path,withheld_sources=['0'*64]).call('calibrate_exploratory_response',args)
    escaped=deepcopy(args);escaped['exploratory'][0]['collection']='../archive-manifest.json'
    with pytest.raises(ValueError,match='escapes'):
        ResearchTools(tmp_path).call('calibrate_exploratory_response',escaped)
    unknown=deepcopy(args);unknown['exploratory'][0]['skip_checks']=True
    with pytest.raises(ValueError,match='endpoint'):
        ResearchTools(tmp_path).call('calibrate_exploratory_response',unknown)
