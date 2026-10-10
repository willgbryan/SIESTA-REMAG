from copy import deepcopy
import io
import tarfile
import numpy as np
import pytest
from siesta_remag.nonlocal_h0 import audit
from siesta_remag.attempts import package_evidence


@pytest.mark.parametrize('soc',[False,True])
def test_raw_matrix_agreement_and_lifetime_rejections(tmp_path,soc):
    # Two sparse entries; real and imaginary SOC changes are checked separately.
    h=np.array([[0.,0.],[.25,-.5],[.25,-.5]],dtype='<f8')
    z=np.zeros((3,8),dtype='<c16');z[1:]=np.arange(8)+.125j
    raw=h.tobytes()+(z.tobytes() if soc else b'');name='remag-nonlocal-h0-0001.bin'
    (tmp_path/name).write_bytes(raw)
    owner=dict(epoch=1,rows=1,columns=2,capacity=2,consumer=3 if soc else 2,queries=2,neighbors=2)
    record=dict(owner,preparation_epoch=0,split=0,local_neighbors=1,graph_nodes=3,status=0,
        setup_bytes=160 if soc else 32,device_allocation_bytes=224 if soc else 96,
        export_bytes=148 if soc else 20,raw_bytes=len(raw),raw_file=name,compared=True,passed=True,
        bound=True,force_borrowed=True,complete=True,scalar_max_error=0.,soc_max_error=0.,
        derivative_upload_bytes=0,integral_download_bytes=0)
    report=dict(schema=1,cycles=[record]);geometry=dict(cycles=[owner])
    trace=dict(intervals=[dict(phase='initialization',boundaries={'nonlocal_h0_shadow':dict(
        host_to_device_bytes=160 if soc else 32,device_to_host_bytes=148 if soc else 20,copy_calls=8 if soc else 6)})])
    result=audit(report,tmp_path,[1],geometry,trace);assert result['passed'],result
    for key,value in (('force_borrowed',False),('complete',False),('preparation_epoch',1),('integral_download_bytes',64)):
        bad=deepcopy(report);bad['cycles'][0][key]=value
        assert not audit(bad,tmp_path,[1],geometry,trace)['passed']
    bad=deepcopy(trace);bad['intervals'][0]['phase']='scf'
    assert not audit(report,tmp_path,[1],geometry,bad)['passed']
    if soc:z[2,3]+=.01j
    else:h[2,1]+=.01
    (tmp_path/name).write_bytes(h.tobytes()+(z.tobytes() if soc else b''))
    assert not audit(report,tmp_path,[1],geometry,trace)['passed']


def test_matrix_evidence_survives_returned_archive(tmp_path):
    name='remag-nonlocal-h0-0001.bin';payload=b'raw scalar and complex H0 matrices'
    (tmp_path/name).write_bytes(payload)
    with tarfile.open(fileobj=io.BytesIO(package_evidence(tmp_path))) as archive:
        assert archive.extractfile(name).read()==payload


@pytest.mark.parametrize('soc',[False,True])
@pytest.mark.parametrize('queries',[0,2])
@pytest.mark.parametrize('poisoned',[False,True])
@pytest.mark.parametrize('compact',[False,True])
def test_resident_h0_requires_scf_and_force_leases_without_matrix_exports(tmp_path,soc,queries,poisoned,compact):
    local=1 if queries else 0;values=144 if soc else 16
    owner=dict(epoch=2,rows=1,columns=2,capacity=2,consumer=3 if soc else 2,
               queries=queries,neighbors=local)
    staged=4*(2+local)+4
    r=dict(owner,preparation_epoch=1,split=0,local_neighbors=local,
        graph_nodes=3 if queries else 2,status=0,setup_bytes=staged,
        device_allocation_bytes=staged+values+(0 if local else 4)+8*max(1,4*queries),
        export_bytes=4,raw_bytes=0,raw_file='',compared=False,passed=True,
        bound=True,force_borrowed=True,complete=True,scalar_max_error=None,soc_max_error=None,
        derivative_upload_bytes=0,integral_download_bytes=0,production=True,
        scf_borrowed=True,scf_released=True,poison_requested=poisoned,host_poisoned=poisoned)
    report=dict(schema=1,cycles=[r]);geometry=dict(cycles=[owner])
    if compact:
        owner.update(compact_queries=True,centers=1 if queries else 0)
        r.update(compact_queries=True,query_expansion_nodes=2 if queries else 0)
        r['graph_nodes']+=r['query_expansion_nodes']
    cycles=[dict(epoch=2,nonlocal_h0=dict(device=True,retained_bytes=values))]
    trace=dict(intervals=[dict(phase='initialization',boundaries={'nonlocal_h0':dict(
        host_to_device_bytes=staged,device_to_host_bytes=4,copy_calls=3+bool(local))})])
    def check(report=report,cycles=cycles,trace=trace):
        return audit(report,tmp_path,[2],geometry,trace,production=True,cycle_records=cycles,poisoned=poisoned,compact_queries=compact)
    assert check()['passed']
    for key,value in [('scf_borrowed',False),('scf_released',False),('force_borrowed',False),
                      ('compared',True),('production',False),('export_bytes',4+values),
                      ('scalar_max_error',0.),('raw_bytes',3*values),
                      ('poison_requested',not poisoned),('host_poisoned',not poisoned)]:
        bad=deepcopy(report);bad['cycles'][0][key]=value
        assert not check(report=bad)['passed'],key
    assert not check(cycles=None)['passed']
    bad=deepcopy(cycles);bad[0]['nonlocal_h0']['retained_bytes']+=8
    assert not check(cycles=bad)['passed']
    bad=deepcopy(trace);bad['intervals'][0]['phase']='scf'
    assert not check(trace=bad)['passed']
    if compact:
        for key,value in [('compact_queries',False),('query_expansion_nodes',7),('graph_nodes',99)]:
            bad=deepcopy(report);bad['cycles'][0][key]=value
            assert not check(report=bad)['passed'],key
    (tmp_path/'remag-nonlocal-h0-0001.bin').write_bytes(b'unexpected matrix export')
    assert not check()['passed']
