"""Operator algebra, native component compilation and evidence-bound handoff."""
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest
from scipy.linalg import eigh

from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save
from siesta_remag.research.electronic import RY_EV
from siesta_remag.research.soc import NATIVE_METADATA, export_soc, project_soc, validate_native_soc
from siesta_remag.research.tools import ResearchTools
from siesta_remag.research.experiments import validate_soc_request
from test_research_mechanisms import native_fixture
from test_research import experiment
from test_discovery import backend


def soc_fixture(root):
    report, native=native_fixture(root)
    # Mixed real/imaginary spin blocks, with an exactly zero spatial spin trace.
    v=np.array([[.3,.04j,.02,.01j],[-.04j,-.3,.03j,-.02],
                [.02,-.03j,-.1,.05j],[-.01j,-.02,-.05j,.1]])
    blocks=[v[::2,::2].real,v[1::2,1::2].real,v[::2,1::2].real,-v[::2,1::2].imag,
            v[::2,::2].imag,v[1::2,1::2].imag,v[1::2,::2].real,v[1::2,::2].imag]
    raw=(native/'remag.HSX').read_bytes(); records=[]; position=0
    while position<len(raw):
        n=struct.unpack('<i',raw[position:position+4])[0]
        records.append(raw[position+4:position+4+n]);position+=n+8
    records[12:28]=[np.asarray(row/RY_EV,dtype='<f8').tobytes() for block in blocks for row in block.T]
    (native/'remag.SOC.HSX').write_bytes(b''.join(struct.pack('<i',len(r))+r+struct.pack('<i',len(r)) for r in records))
    save(native/'remag.SOC.json',NATIVE_METADATA)
    data=load(report);data['artifacts']={'soc-z/'+p.name:sha256(p) for p in native.iterdir()};save(report,data)
    return report,native,v


def test_nonorthogonal_projection_and_phase_covariance():
    s=np.kron([[1.,.3],[.3,1.]],np.eye(2))
    w,u=eigh(s);root=(u/np.sqrt(w))@u.T
    v=np.diag([.3,-.3,-.1,.1]);f=np.array([1.,.5,0.,0.])
    got=project_soc(v,s,root,f)
    expected=root.T@v@root
    assert np.allclose(got['state_matrix_ev'],expected)
    assert not np.allclose(expected,root.T@s@v@root)
    assert got['occupied_expectation_ev']==pytest.approx(float(f@expected.diagonal()))
    phases=np.exp(1j*np.arange(4));changed=project_soc(v,s,root*phases,f)
    assert np.allclose(abs(changed['state_matrix_ev']),abs(expected))
    assert changed['occupied_expectation_ev']==pytest.approx(got['occupied_expectation_ev'])
    # A completely occupied degenerate subspace may be rotated without changing
    # its trace, while individual diagonal elements need not survive the rotation.
    rotation=np.eye(4,dtype=complex);rotation[:2,:2]=[[1/np.sqrt(2),1j/np.sqrt(2)],[1j/np.sqrt(2),1/np.sqrt(2)]]
    a=project_soc(v,s,root,[1,1,0,0]);b=project_soc(v,s,root@rotation,[1,1,0,0])
    assert a['occupied_expectation_ev']==pytest.approx(b['occupied_expectation_ev'])
    assert not np.allclose(a['state_matrix_ev'].diagonal(),b['state_matrix_ev'].diagonal())


@pytest.mark.parametrize('bad',['trace','hermitian','metric','normalization','occupation','nan'])
def test_invalid_operator_or_states_are_rejected(bad):
    v=np.diag([1.,-1]);s=np.eye(2);c=np.eye(2);f=np.array([1.,0])
    if bad=='trace':v[1,1]=1
    if bad=='hermitian':v[0,1]=1
    if bad=='metric':s[0,0]=-1
    if bad=='normalization':c[0,0]=2
    if bad=='occupation':f[0]=2
    if bad=='nan':v[0,0]=np.nan
    with pytest.raises(ValueError):project_soc(v,s,c,f)


def test_native_export_tool_and_cli(tmp_path):
    report,native,v=soc_fixture(tmp_path/'native')
    tool=ResearchTools(tmp_path)
    result=tool.call('export_soc_matrices',dict(report='native/report.json',stage='soc-z',native_directory='native/soc-z',output='inspection',k_indices=[1]))
    assert result['selected_k_weight']==pytest.approx(.5)
    assert result['native_k_count']==2 and result['physical_accuracy_qualified'] is False
    with np.load(tmp_path/'inspection/soc.npz') as arrays:
        assert np.allclose(arrays['operator_ev'][0],v)
        assert np.allclose(np.trace(arrays['operator_ev'][0].reshape(2,2,2,2),axis1=1,axis2=3),0)
        assert result['selected_weighted_expectation_ev']==pytest.approx(.5*arrays['occupied_expectation_ev'][0])
    from siesta_remag.research.__main__ import main
    assert main(['soc-matrices',str(report),'--stage','soc-z','--native-directory',str(native),
                 '--output',str(tmp_path/'cli'),'--k-indices','1'])==0
    assert load(tmp_path/'cli/soc.json')['occupied_expectations_ev']==result['occupied_expectations_ev']
    with pytest.raises(ValueError,match='scalar profile'):
        ResearchTools(tmp_path,information_profile='scalar').call('export_soc_matrices',{})
    with pytest.raises(ValueError,match='held-out'):
        ResearchTools(tmp_path,withheld_sources=['a'*64]).call('export_soc_matrices',
            dict(report='native/report.json',stage='soc-z',native_directory='native/soc-z',output='held'))


@pytest.mark.parametrize('mutation',['missing','hash','definition','total_h','overlap','branch','eig'])
def test_saved_soc_identity_and_native_health_fail_closed(tmp_path,mutation):
    report,native,_=soc_fixture(tmp_path/'native')
    if mutation=='missing':(native/'remag.SOC.HSX').unlink()
    if mutation=='hash':(native/'remag.SOC.HSX').write_bytes(b'invalid')
    if mutation=='definition':save(native/'remag.SOC.json',NATIVE_METADATA|{'operator':'total-H'})
    if mutation=='total_h':shutil.copyfile(native/'remag.HSX',native/'remag.SOC.HSX')
    if mutation=='overlap':
        p=native/'remag.SOC.HSX';raw=bytearray(p.read_bytes());raw[-20:-12]=struct.pack('<d',.8);p.write_bytes(raw)
    if mutation=='branch':
        data=load(report);data['stages'][0]['branch']['passed']=False;save(report,data)
    if mutation=='eig':(native/'remag.EIG').write_text('0\n4 8 2\n1 0 0 0 0\n2 0 0 0 0\n')
    if mutation not in ('missing','hash'):
        data=load(report);data['artifacts']={'soc-z/'+p.name:sha256(p) for p in native.iterdir()};save(report,data)
    with pytest.raises((ValueError,FileNotFoundError)):
        export_soc(report,'soc-z',native,tmp_path/'inspection')


def test_soc_request_remains_explicit_and_requires_complete_device_h0_owner():
    validate_soc_request(False,{'kind':'scalar'})
    validate_soc_request(True,{'kind':'anisotropy'},{'REMAG_DEVICE_SCF':'1'})
    owner={k:'1' for k in ('REMAG_NONLOCAL_H0','REMAG_DEVICE_SCF','REMAG_GPU_INITIALIZATION','REMAG_INTEGRAL_FORCE')}
    validate_soc_request(True,{'kind':'anisotropy'},owner)
    validate_soc_request(True,{'kind':'anisotropy'},owner|{'REMAG_NONLOCAL_H0_POISON_HOST':'1'})
    for key in owner:
        with pytest.raises(ValueError):validate_soc_request(True,None,{k:v for k,v in owner.items() if k!=key}|{'REMAG_NONLOCAL_H0_POISON_HOST':'1'})
    for flag in ('REMAG_NONLOCAL_H0','REMAG_NONLOCAL_H0_POISON_HOST'):
        with pytest.raises(ValueError,match='host nonlocal'):validate_soc_request(True,None,{flag:'1'})
    for enabled,q in [(1,None),('true',None),(True,{'kind':'scalar'})]:
        with pytest.raises(ValueError):validate_soc_request(enabled,q)


def test_native_device_snapshot_metadata_and_bound_operator_are_required(tmp_path):
    report,native,_=soc_fixture(tmp_path/'data');operator=validate_native_soc(native)
    m=NATIVE_METADATA|dict(format='REMAG-NATIVE-SOC/2',producer='remag-device-nonlocal-h0',host_nonlocal_h0=False,
        snapshot_epoch=2,snapshot_rows=int(operator['header'][1]),
        snapshot_columns=int(operator['header'][1]*np.prod(operator['header'][4:])),
        snapshot_capacity=len(operator['columns']),snapshot_phase='post_scf',snapshot_stage='before_forces')
    save(native/'remag.SOC.json',m);validate_native_soc(native)
    r=load(report);r['artifacts']['soc-z/remag.SOC.json']=sha256(native/'remag.SOC.json');save(report,r)
    result=export_soc(report,'soc-z',native,tmp_path/'device-inspection')
    assert result['evidence']['metadata']==sha256(native/'remag.SOC.json')
    assert result['native_ownership']['snapshot_epoch']==2
    for key,value in [('snapshot_epoch',False),('snapshot_capacity',1),('snapshot_rows',999),
                      ('snapshot_columns',999),('snapshot_phase','scf'),('host_nonlocal_h0',0),('producer','unknown')]:
        save(native/'remag.SOC.json',m|{key:value})
        with pytest.raises(ValueError,match='SOC producer|ownership'):validate_native_soc(native)


def test_compiled_native_components_remove_only_nonlocal_spin_trace(tmp_path):
    compiler=shutil.which('gfortran')
    if compiler is None:pytest.skip('gfortran required for native arithmetic control')
    root=Path(__file__).resolve().parents[1]
    shutil.copyfile(root/'native/remag_soc_components.F90.in',tmp_path/'components.f90')
    (tmp_path/'check.f90').write_text('''program check
+ use m_remag_soc_components
+ use iso_c_binding,only:c_double
+ use, intrinsic :: ieee_arithmetic,only:ieee_value,ieee_quiet_nan
+ implicit none
+ complex(c_double)::off(2,4),original(2,4)
+ real(c_double)::h(2,8),bad(2,7),expected(8)
+ integer::status
+ off(1,:)=[cmplx(3.,4.,c_double),cmplx(1.,2.,c_double),cmplx(5.,6.,c_double),cmplx(7.,8.,c_double)]
+ off(2,:)=off(1,:);off(2,1:2)=off(2,1:2)+cmplx(10.,20.,c_double)
+ original=off;expected=[1.,-1.,5.,6.,1.,-1.,7.,-8.]
+ call remag_soc_components(off,h,status)
+ if(status/=0.or.any(off/=original))stop 1
+ if(any(h(1,:)/=expected).or.any(h(2,:)/=expected))stop 2
+ call remag_soc_components(off,bad,status)
+ if(status==0)stop 3
+ off(1,1)=ieee_value(0._c_double,ieee_quiet_nan)
+ call remag_soc_components(off,h,status)
+ if(status==0)stop 4
+end program
'''.replace('\n+','\n'))
    subprocess.run([compiler,'-Wall','-Wextra','-fcheck=all','components.f90','check.f90','-o','check'],cwd=tmp_path,check=True,capture_output=True,text=True)
    subprocess.run([str(tmp_path/'check')],check=True,capture_output=True,text=True)


@pytest.fixture(scope='module')
def compiled_export(tmp_path_factory):
    compiler=shutil.which('gfortran')
    if compiler is None:pytest.skip('gfortran required for native adapter compilation')
    root=Path(__file__).resolve().parents[1]; build=tmp_path_factory.mktemp('soc-export-build')
    for name in ('remag_soc_components','remag_soc_export'):
        shutil.copyfile(root/'native'/f'{name}.F90.in',build/f'{name}.f90')
    subprocess.run([compiler,'-fcheck=all',str(root/'tests/fixtures/soc_export_stubs.f90'),
        'remag_soc_components.f90','remag_soc_export.f90',str(root/'tests/fixtures/soc_export_driver.f90'),
        '-o','export'],cwd=build,check=True,capture_output=True,text=True)
    return build/'export'


@pytest.mark.parametrize('mode',['valid','disabled','parallel','onsite','collinear','device','device_poison',
    'device_missing','device_epoch','device_geometry','device_topology','device_copy_failure','poison','external','missing','nonfinite'])
def test_native_export_adapter_ownership_and_component_handoff(tmp_path,compiled_export,mode):
    result=subprocess.run([str(compiled_export),mode],cwd=tmp_path,capture_output=True,text=True)
    if mode in ('valid','disabled','device','device_poison'):
        assert result.returncode==0,result.stdout+result.stderr
        if mode=='valid':assert load(tmp_path/'remag.SOC.json')==NATIVE_METADATA
        elif mode.startswith('device'):
            m=load(tmp_path/'remag.SOC.json');assert m['format']=='REMAG-NATIVE-SOC/2'
            assert m['snapshot_epoch']==2 and not m['host_nonlocal_h0'] and m['snapshot_capacity']==1
        else:assert not (tmp_path/'remag.SOC.json').exists()
    else:
        assert result.returncode!=0
        assert 'REMAG SOC' in result.stdout+result.stderr
        assert not (tmp_path/'remag.SOC.json').exists()


def test_opt_in_query_stops_if_native_executable_does_not_supply_operator(experiment,tmp_path):
    from siesta_remag.research.experiments import prepare_experiment,run_experiment
    bundle=tmp_path/'soc-query'
    request=prepare_experiment(load(experiment/'structure.json'),experiment/'pseudos',bundle,
        mesh=[3,3,3],query={'kind':'anisotropy'},soc_matrix=True)
    assert request['soc_matrix'] is True
    assert 'REMAG.WriteSOCMatrix' not in (bundle/'scalar/input.fdf').read_text()
    binary=tmp_path/'binary';binary.write_text('engineering fixture only')
    calls=[]
    result=run_experiment(bundle,tmp_path/'result',binary,executable_sha256=sha256(binary),runner=backend(calls))
    assert result['status']=='held_error'
    assert [c[0] for c in calls]==['scalar','soc-z']
    assert 'REMAG.WriteSOCMatrix true' in (tmp_path/'result/inputs/z/input.fdf').read_text()
    assert 'SOC' in result['reason']
    assert result['anisotropy'] is None
    with pytest.raises(ValueError,match='host nonlocal'):
        run_experiment(bundle,tmp_path/'forbidden',binary,executable_sha256=sha256(binary),
            runtime_env={'REMAG_NONLOCAL_H0':'1'},runner=backend([]))
    assert not (tmp_path/'forbidden').exists()


def test_policy_session_can_select_soc_tool(tmp_path):
    import sys
    from siesta_remag.research.agent import run_session
    soc_fixture(tmp_path/'native')
    policy=tmp_path/'policy.py'
    policy.write_text('''import json,sys
context=json.load(sys.stdin)
if context['history']:
 print(json.dumps({'finish':'Fixture SOC tool returned; no scientific inference'}))
else:
 print(json.dumps({'tool':'export_soc_matrices','args':{'report':'native/report.json',
 'stage':'soc-z','native_directory':'native/soc-z','output':'inspection','k_indices':[0]},
 'rationale':'Inspect the synthetic nonlocal operator'}))
''')
    result=run_session(tmp_path,'session',objective='Synthetic SOC inspection integration',
        policy_command=[sys.executable,str(policy)],max_steps=2,max_seconds=30)
    assert result['status']=='completed',result
    assert load(tmp_path/'inspection/soc.json')['operator']==NATIVE_METADATA['operator']
    assert result['native_submissions']==0
