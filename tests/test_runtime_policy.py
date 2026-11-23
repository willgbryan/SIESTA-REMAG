import copy
import pytest
from siesta_remag.research import runtime_policy as rp
from siesta_remag.research.preflight import build_runtime, verify_runtime_files
from siesta_remag.discovery.common import load, save
from siesta_remag.corpus import sha256
from test_research_preflight import compiled
from test_research_relaxation_execution import discovery, DiscoveryTransport, run

OBS = dict(cuda_runtime=12060,cuda_driver_api=13030,nvidia_driver_versions=['610.57.04'])


def test_compiled_policy_bound_and_cannot_be_removed_or_widened(compiled, monkeypatch):
    exe,lib=compiled
    r=build_runtime(exe,lib,{'REMAG_DEVICE_SCF':'1','LD_LIBRARY_PATH':str(lib.parent)})
    assert r['driver_policy']==dict(format='REMAG-SCF-RUNTIME-POLICY/1',**OBS)
    monkeypatch.setattr(rp,'observe',lambda:dict(OBS))
    proof=verify_runtime_files(r,require_deployment=True)
    rp.verify_saved(r,proof)
    for mutation in ('remove','widen'):
        changed=copy.deepcopy(r)
        if mutation=='remove':del changed['driver_policy']
        else:changed['driver_policy']['nvidia_driver_versions'].append('620.70.06')
        with pytest.raises(ValueError,match='policy differs'):verify_runtime_files(changed)
    with pytest.raises(ValueError,match='Missing native runtime admission'):
        rp.verify_saved(r,verify_runtime_files(r))


@pytest.mark.parametrize('key,value', [('cuda_runtime',12080),('cuda_driver_api',13050),
    ('cuda_driver_api',None),('cuda_driver_api',True),('nvidia_driver_versions',[]),
    ('nvidia_driver_versions',['620.70.06']),('nvidia_driver_versions',['610.57.04','610.57.04'])])
def test_unknown_or_mismatched_runtime_fails_closed(compiled,key,value):
    policy=rp.probe(compiled[1]);observed=dict(OBS);observed[key]=value
    with pytest.raises(ValueError,match='not admitted'):rp.check(policy,observed)


@pytest.mark.parametrize('mutation',['missing','wrong_driver','forged_success'])
def test_saved_proof_is_recomputed(compiled,monkeypatch,mutation):
    exe,lib=compiled;r=build_runtime(exe,lib,{'REMAG_DEVICE_SCF':'1','LD_LIBRARY_PATH':str(lib.parent)})
    monkeypatch.setattr(rp,'observe',lambda:dict(OBS))
    proof=verify_runtime_files(r,require_deployment=True)
    if mutation=='missing':proof.pop('deployment')
    elif mutation=='wrong_driver':proof['deployment']['observed']['cuda_driver_api']=13050
    else:proof['deployment']['scientific_acceptance']=True
    with pytest.raises(ValueError):rp.verify_saved(r,proof)


def test_actual_remote_entry_rejects_before_native_solver(discovery,compiled,tmp_path,monkeypatch):
    exe,lib=compiled
    runtime=build_runtime(exe,lib,{'REMAG_DEVICE_SCF':'1','LD_LIBRARY_PATH':str(lib.parent)})
    save(tmp_path/'runtime.json',runtime)
    config=load(tmp_path/'config.json');config['runtime_sha256']=sha256(tmp_path/'runtime.json');save(tmp_path/'config.json',config)
    monkeypatch.setattr(rp,'observe',lambda:dict(OBS,cuda_driver_api=13050,nvidia_driver_versions=['620.70.06']))
    t=DiscoveryTransport(monkeypatch);result=run(tmp_path,t)
    assert result['allocation_stopped'] and t.native==[]
    assert not (tmp_path/'attempt/collected/query').exists()
    report=load(tmp_path/'attempt/collected/report.json')
    assert 'not admitted' in str(report)
    assert result['usable_result_seconds'] is None
