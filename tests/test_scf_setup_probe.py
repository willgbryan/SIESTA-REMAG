import copy
import json
import pytest
from siesta_remag.research.scf_setup_probe import CASES,validate_policy,summarize_case
from siesta_remag.discovery.collection import pack,unpack
from siesta_remag.discovery.common import save


def policy():
    return dict(format='REMAG-SCF-SETUP-PROBE/1',cases=copy.deepcopy(CASES),timeout_seconds=420,
        reservation_cents=53,retries=0,gpu_count=1,cycle_launch=False,image='im-example',previous_app='ap-example',
        volume='existing-volume',files={'probe_scf_setup.cpp':'a'*64,'runtime.json':'b'*64},
        module_sha256='c'*64,helper_sha256='d'*64,held_owner_sha256='e'*64)


def test_finite_shape_plan():
    assert validate_policy(policy())['cases']==CASES


@pytest.mark.parametrize('key,value',[('timeout_seconds',421),('gpu_count',2),('retries',1),
    ('cycle_launch',True),('cases',CASES+[CASES[0]]),('reservation_cents',0),('files',{'../escape':'a'*64})])
def test_scope_change_rejected(key,value):
    p=policy();p[key]=value
    with pytest.raises(ValueError):validate_policy(p)


def output():
    return '\n'.join(map(json.dumps,[dict(stage='setup',n=76,nk=132,mesh=4),
        dict(stage='construct_begin',cycle_launch_authorized=False),
        dict(stage='constructed',graph_nodes=27501,forbidden_nodes=0,jobs=264,lanes=8)]))


@pytest.mark.parametrize('code,timeout,tail',[(-11,False,''),(0,True,''),(2,False,''),(0,False,'\n{broken')])
def test_failed_process_cannot_pass(code,timeout,tail):
    result=summarize_case(CASES[1],output()+tail,code,timeout)
    assert not result['constructed'] and not result['native_runtime_repaired']


def test_setup_pass_is_not_repair_or_scientific_acceptance(tmp_path):
    result=summarize_case(CASES[1],output(),0,False)
    assert result['constructed'] and not result['scientific_acceptance'] and not result['native_runtime_repaired']
    save(tmp_path/'report.json',dict(status='diagnostic_complete',invocation_sha256='a'*64,cases=[result]))
    (tmp_path/'stdout.log').write_text(output())
    (tmp_path/'stderr.log').write_text('saved raw process stderr')
    recovered=tmp_path.parent/(tmp_path.name+'-collected')
    report=unpack(pack(tmp_path),recovered)
    assert report['cases'][0]==result
    assert (recovered/'stderr.log').read_text()=='saved raw process stderr'


def test_wrong_shape_and_missing_end_are_not_passes():
    assert not summarize_case(CASES[0],output(),0,False)['constructed']
    assert not summarize_case(CASES[1],output().splitlines()[0],0,False)['constructed']
