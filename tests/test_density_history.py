import copy
import json
from pathlib import Path
import pytest
from siesta_remag.density_history import policy, record_errors
from siesta_remag.terminal_exports import requested_density_exports


def test_history_policy_preserves_cg_default_and_first_label():
    assert policy('DM.History.Depth 3\n') == dict(depth=1, mode=2)
    assert policy('DM.History.Depth 3\nDM.AllowExtrapolation true\nDM_AllowExtrapolation false') == dict(depth=3, mode=2)
    assert policy('DM.History.Depth 3\nDM.AllowReuse false') == dict(depth=0, mode=2)
    assert policy('UseDIISforDMExtrapolation false\nUseSVDExperimental true')['mode'] == 0
    assert policy('UseSVDExperimental true')['mode'] == 3
    assert policy('UseSVD false')['mode'] == 1
    with pytest.raises(ValueError):
        policy('%include controls.fdf')


def test_density_consumers_distinguish_restart_from_tsde_and_netcdf():
    base='Write.DM false\n'
    assert requested_density_exports(base, True) == 0
    assert requested_density_exports(base, False) == 1
    assert requested_density_exports(base+'TS.DE.Save true\n', True) == 4
    assert requested_density_exports(base+'CDF.Save true\n', True) == 0
    assert requested_density_exports(base+'CDF.Save true\nWrite.DM.End.Of.Cycle true', True) == 10
    assert requested_density_exports(base+'Spin spin-orbit\nWriteOrbMom true', True) == 16
    assert requested_density_exports(base, True, shadow=True) == 32
    with pytest.raises(ValueError):
        requested_density_exports('Write.DM true', True)


def test_history_requires_force_epoch_and_zero_reuse_upload():
    # Two geometry leases with depth one: atomic D first, retained D second.
    history=[
        dict(token=1,epoch=3,atoms=2,rows=3,capacity=5,spins=2,depth=1,mode=2,previous_count=0,count=1,
             reused=False,bound=True,finished=True,saved=True,forces_accepted=True,status=0,rank=0,fallback=0,
             setup_bytes=188,setup_copies=6,initial_status_bytes=80,initial_status_copies=2,
             post_status_bytes=40,post_status_copies=1,coefficient_export_bytes=0,workspace_bytes=4096),
        dict(token=2,epoch=7,atoms=2,rows=3,capacity=5,spins=2,depth=1,mode=2,previous_count=1,count=1,
             reused=True,bound=True,finished=True,saved=True,forces_accepted=True,status=0,rank=1,fallback=0,
             setup_bytes=140,setup_copies=5,initial_status_bytes=80,initial_status_copies=2,
             post_status_bytes=40,post_status_copies=1,coefficient_export_bytes=8,workspace_bytes=4096)]
    cycles=[dict(epoch=e,converged=True,terminal=dict(capacity=5,spins=2)) for e in (3,7)]
    initial=[dict(epoch=3,rows=3,atoms=2,density_from_history=False,density_upload_bytes=80),
             dict(epoch=7,rows=3,atoms=2,density_from_history=True,density_upload_bytes=0)]
    forces=[dict(epoch=3),dict(epoch=7)]
    expected=dict(depth=1,mode=2)
    assert not record_errors(history,cycles,initial,forces,expected)
    for key,value in [('count',2),('previous_count',0),('reused',False),('finished',False),('post_status_bytes',0),('setup_bytes',220)]:
        broken=copy.deepcopy(history);broken[1][key]=value
        assert record_errors(broken,cycles,initial,forces,expected), key
    initial[1]['density_upload_bytes']=80
    assert record_errors(history,cycles,initial,forces,expected)
    initial[1]['density_upload_bytes']=0
    forces[1]['epoch']=3
    assert record_errors(history,cycles,initial,forces,expected)


def test_constrained_svd_rank_uses_augmented_system_from_gpu_regression():
    fixture=json.loads((Path(__file__).parent/'fixtures/density-history-v91-ranks.json').read_text())
    args=[fixture[key] for key in ('history','cycles','initial','forces','expected')]
    assert any(r['rank']==r['previous_count']+1 for r in args[0] if r['previous_count']>1)
    assert not record_errors(*args)
    for index,rank in ((0,1),(1,2),(2,4),(4,5)):
        broken=copy.deepcopy(args);broken[0][index]['rank']=rank
        assert record_errors(*broken)
    broken=copy.deepcopy(args);broken[0][4]['fallback']=1
    assert record_errors(*broken)
