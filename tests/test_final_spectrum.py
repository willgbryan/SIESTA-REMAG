from copy import deepcopy
from siesta_remag.final_spectrum import record_errors


def trajectory():
    cycles=[dict(epoch=i,converged=True,terminal=dict(bank_size=12,deferred_spectrum=True,spectrum_bytes=[0,0])) for i in (1,2)]
    records=[dict(epoch=i,bank_size=12,retained_device_bytes=96,eigenvalue_export_bytes=96 if i==2 else 0,
        occupation_export_bytes=0,consumed=i==2,discarded=True) for i in (1,2)]
    forces=[dict(epoch=i,status=0) for i in (1,2)]
    return records,cycles,forces


def test_final_spectrum_rejects_stale_or_missing_state():
    args=trajectory();assert not record_errors(*args)
    for item,key,value in ((0,'epoch',1),(0,'consumed',False),(0,'eigenvalue_export_bytes',0),
                           (1,'converged',False),(2,'status',1)):
        bad=deepcopy(args);bad[item][-1][key]=value
        assert record_errors(*bad)
    assert record_errors(args[0][1:],args[1],args[2])


def test_final_spectrum_rejects_unnecessary_downloads():
    args=trajectory()
    bad=deepcopy(args);bad[0][0]['eigenvalue_export_bytes']=96
    assert record_errors(*bad)
    bad=deepcopy(args);bad[0][-1]['occupation_export_bytes']=96
    assert record_errors(*bad)
    bad=deepcopy(args);bad[1][0]['terminal']['spectrum_bytes']=[96,96]
    assert record_errors(*bad)
