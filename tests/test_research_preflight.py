"""Real compiled capability metadata and no-spend rejection controls."""
import copy
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import types

import pytest

from test_research_execution import prepared, SyntheticTransport
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load,save
from siesta_remag.research.execution import execute_query
from siesta_remag.research.preflight import (build_runtime,probe_library,validate_runtime,
    verify_runtime_files,preflight_query,check_mesh_capacity)


@pytest.fixture(scope='module')
def compiled(tmp_path_factory):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('C++ compiler required')
    repo=Path(__file__).resolve().parents[1];root=tmp_path_factory.mktemp('capacity')
    out=root/'capacity';lib=root/'libremag_capacity_test.so'
    flags=[compiler,'-std=c++17','-Wall','-Wextra','-Werror','-I'+str(repo/'native')]
    subprocess.run([*flags,str(repo/'tests/fixtures/scf_capacity_driver.cpp'),'-o',str(out)],check=True,capture_output=True,text=True)
    subprocess.run([str(out)],check=True,capture_output=True,text=True)
    subprocess.run([*flags,'-shared','-fPIC',str(repo/'native/capabilities.cpp'),'-o',str(lib)],check=True,capture_output=True,text=True)
    return out,lib


def pin_runtime(tmp_path,compiled,limit=None):
    exe,lib=compiled
    runtime=build_runtime(exe,lib,{'REMAG_DEVICE_SCF':'1','LD_LIBRARY_PATH':str(lib.parent)})
    if limit is not None:runtime['capabilities']['device_scf']['max_kpoints']=limit
    save(tmp_path/'runtime.json',runtime)
    config=load(tmp_path/'execution-config.json');config['runtime_sha256']=sha256(tmp_path/'runtime.json')
    save(tmp_path/'execution-config.json',config)
    return runtime


def remesh(bundle,n):
    p=bundle/'scalar/input.fdf';text=p.read_text()
    start=text.index('%block kgrid_Monkhorst_Pack');end=text.index('%endblock kgrid_Monkhorst_Pack')
    text=text[:start]+'%block kgrid_Monkhorst_Pack\n'+''.join(' '.join(str(n if i==j else 0) for j in range(3))+' 0.0\n' for i in range(3))+text[end:]
    p.write_text(text);m=load(bundle/'scalar/manifest.json');m['kmesh']=[n]*3;m['input_sha256']=sha256(p);save(bundle/'scalar/manifest.json',m)
    r=load(bundle/'experiment.json');r['kmesh']=[n]*3
    for name in r['files']:r['files'][name]=sha256(bundle/name)
    save(bundle/'experiment.json',r)


def test_real_compiled_contract_covers_scalar_and_soc_acceptance_meshes(compiled):
    c=probe_library(compiled[1]);assert c['max_kpoints']==4096
    for n in (6,12,16):
        assert check_mesh_capacity({'kmesh':[n]*3},c)['native_kpoints_upper_bound']==n**3
    with pytest.raises(ValueError,match='4913.*4096'):check_mesh_capacity({'kmesh':[17]*3},c)


@pytest.mark.parametrize('mesh',[6,12])
def test_old_limit_rejected_before_account_intent_image_or_reservation(prepared,tmp_path,compiled,mesh):
    remesh(prepared,mesh);pin_runtime(tmp_path,compiled,128)
    before=(tmp_path/'budget.json').read_bytes();transport=SyntheticTransport()
    with pytest.raises(ValueError,match='compiled device-SCF limit is 128'):
        execute_query(tmp_path,'execution-config.json','bundle','attempt',transport=transport)
    assert transport.info_calls==[] and transport.calls==[]
    assert not (tmp_path/'.research-execution').exists() and not (tmp_path/'attempt').exists()
    assert (tmp_path/'budget.json').read_bytes()==before


def test_missing_build_capability_fails_closed_without_spending(prepared,tmp_path):
    r=load(tmp_path/'runtime.json');r['environment']['REMAG_DEVICE_SCF']='1';save(tmp_path/'runtime.json',r)
    c=load(tmp_path/'execution-config.json');c['runtime_sha256']=sha256(tmp_path/'runtime.json');save(tmp_path/'execution-config.json',c)
    t=SyntheticTransport()
    with pytest.raises(ValueError,match='build-specific capabilities'):
        execute_query(tmp_path,'execution-config.json','bundle','attempt',transport=t)
    assert not t.calls and not t.info_calls


def test_build_binding_library_bytes_and_probe_are_checked(prepared,tmp_path,compiled):
    r=pin_runtime(tmp_path,compiled);verify_runtime_files(r)
    stale=copy.deepcopy(r);stale['capabilities']['executable_sha256']='0'*64
    with pytest.raises(ValueError,match='bound to this executable'):validate_runtime(stale)
    changed=copy.deepcopy(r);changed['capabilities']['device_scf']['max_kpoints']=128
    with pytest.raises(ValueError,match='bytes/capabilities differ'):verify_runtime_files(changed)
    changed=copy.deepcopy(r);changed['capabilities']['library_sha256']='0'*64
    with pytest.raises(ValueError,match='bytes/capabilities differ'):verify_runtime_files(changed)
    changed=copy.deepcopy(r);changed['environment']['LD_LIBRARY_PATH']='/unrelated'
    with pytest.raises(ValueError,match='pinned native library'):validate_runtime(changed)


def test_builder_preserves_runtime_mount_alias_and_still_verifies_bytes(tmp_path,compiled):
    # A volume mount may resolve to a provider-internal directory. Its public
    # alias is the portable runtime path and already leads LD_LIBRARY_PATH.
    alias=tmp_path/'results';alias.symlink_to(compiled[1].parent,target_is_directory=True)
    exe=alias/compiled[0].name;lib=alias/compiled[1].name
    runtime=build_runtime(exe,lib,{'REMAG_DEVICE_SCF':'1','LD_LIBRARY_PATH':str(alias)})
    assert runtime['executable']==str(exe)
    assert runtime['capabilities']['library']==str(lib)
    assert verify_runtime_files(runtime)['actual_files_verified']
    runtime['capabilities']['library_sha256']='0'*64
    with pytest.raises(ValueError,match='bytes/capabilities differ'):verify_runtime_files(runtime)


def test_prepared_mesh_and_input_cannot_disagree(prepared,tmp_path,compiled):
    r=pin_runtime(tmp_path,compiled);remesh(prepared,6)
    assert preflight_query(prepared,r)['native_kpoints_upper_bound']==216
    manifest=load(prepared/'scalar/manifest.json');manifest['kmesh']=[12]*3;save(prepared/'scalar/manifest.json',manifest)
    q=load(prepared/'experiment.json');q['files']['scalar/manifest.json']=sha256(prepared/'scalar/manifest.json');save(prepared/'experiment.json',q)
    with pytest.raises(ValueError,match='meshes differ'):preflight_query(prepared,r)
    q['kmesh']=[12]*3;save(prepared/'experiment.json',q)
    with pytest.raises(ValueError,match='Native input mesh differs'):preflight_query(prepared,r)


def test_low_level_modal_entry_rejects_before_any_image_or_app_creation(prepared,tmp_path,compiled,monkeypatch):
    remesh(prepared,6);pin_runtime(tmp_path,compiled,128)
    campaign=tmp_path/'campaign';campaign.mkdir();shutil.copytree(prepared,campaign/'query')
    save(campaign/'execution.json',dict(experiments=['query'],timeout_seconds=120,hourly_reservation_usd=3,budget_usd=.1,budget_stage='queries'))
    for k,v in dict(REMAG_RESEARCH_CAMPAIGN=campaign,REMAG_RESEARCH_RUNTIME=tmp_path/'runtime.json',REMAG_RESEARCH_IMAGE='im-fixture',REMAG_RESEARCH_BUDGET_LEDGER=tmp_path/'budget.json').items():monkeypatch.setenv(k,str(v))
    # Any attempted Image/App/Volume API access raises AttributeError, so the
    # expected capability rejection proves the ordering on the production entry.
    monkeypatch.setitem(sys.modules,'modal',types.SimpleNamespace(is_local=lambda:True))
    monkeypatch.setattr(sys,'path',list(sys.path))
    before=(tmp_path/'budget.json').read_bytes()
    with pytest.raises(ValueError,match='compiled device-SCF limit is 128'):
        runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/research_modal.py'))
    assert (tmp_path/'budget.json').read_bytes()==before


def test_offline_cli_accepts_new_build_without_authorization_or_ledger_mutation(prepared,tmp_path,compiled,capsys):
    from siesta_remag.research.__main__ import main
    remesh(prepared,12);pin_runtime(tmp_path,compiled)
    before=(tmp_path/'budget.json').read_bytes()
    assert main(['preflight-query','--runtime',str(tmp_path/'runtime.json'),'--bundle',str(prepared)])==0
    assert json.loads(capsys.readouterr().out)[0]['native_kpoints_upper_bound']==1728
    assert (tmp_path/'budget.json').read_bytes()==before


def held_runtime(tmp_path,compiled):
    runtime=pin_runtime(tmp_path,compiled)
    # Incident identities, independent of the production hold implementation.
    runtime['executable_sha256']='afeabb67ad88ed8ab5735970b3e43008b800755c5ba28f2d2db66db02d2f9604'
    runtime['capabilities']['executable_sha256']=runtime['executable_sha256']
    runtime['capabilities']['library_sha256']='8594a158fbacef1adb7a5a175ffad3f76ddd4f0c8cd1fce76e6b253d27956534'
    save(tmp_path/'runtime.json',runtime)
    config=load(tmp_path/'execution-config.json');config['runtime_sha256']=sha256(tmp_path/'runtime.json')
    save(tmp_path/'execution-config.json',config)
    return runtime


@pytest.mark.parametrize('mesh',[6,12])
def test_native_crash_hold_precedes_account_intent_and_reservation(prepared,tmp_path,compiled,mesh):
    remesh(prepared,mesh);runtime=held_runtime(tmp_path,compiled)
    # Capacity still passes, and historical metadata remains readable.
    validate_runtime(runtime)
    assert check_mesh_capacity({'kmesh':[mesh]*3},runtime['capabilities']['device_scf'])['mesh_capacity_passed']
    before=(tmp_path/'budget.json').read_bytes();transport=SyntheticTransport()
    with pytest.raises(ValueError,match='cudaGraphInstantiate remains unresolved; no spending'):
        execute_query(tmp_path,'execution-config.json','bundle','attempt',transport=transport)
    assert not transport.info_calls and not transport.calls and not transport.native
    assert not (tmp_path/'.research-execution').exists() and not (tmp_path/'attempt').exists()
    assert (tmp_path/'budget.json').read_bytes()==before


def test_native_crash_hold_precedes_low_level_image_creation(prepared,tmp_path,compiled,monkeypatch):
    held_runtime(tmp_path,compiled)
    campaign=tmp_path/'campaign';campaign.mkdir();shutil.copytree(prepared,campaign/'query')
    save(campaign/'execution.json',dict(experiments=['query'],timeout_seconds=120,hourly_reservation_usd=3,budget_usd=.1,budget_stage='queries'))
    for k,v in dict(REMAG_RESEARCH_CAMPAIGN=campaign,REMAG_RESEARCH_RUNTIME=tmp_path/'runtime.json',REMAG_RESEARCH_IMAGE='im-fixture',REMAG_RESEARCH_BUDGET_LEDGER=tmp_path/'budget.json').items():monkeypatch.setenv(k,str(v))
    monkeypatch.setitem(sys.modules,'modal',types.SimpleNamespace(is_local=lambda:True))
    monkeypatch.setattr(sys,'path',list(sys.path))
    before=(tmp_path/'budget.json').read_bytes()
    with pytest.raises(ValueError,match='Native execution held'):
        runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/research_modal.py'))
    assert (tmp_path/'budget.json').read_bytes()==before


def test_native_crash_hold_cannot_be_evaded_by_graph_settings(prepared,tmp_path,compiled):
    runtime=held_runtime(tmp_path,compiled)
    runtime['environment'].update(REMAG_SCF_WARM_START='0',REMAG_SCF_SOLVER_LANES='1')
    with pytest.raises(ValueError,match='Native execution held'):preflight_query(prepared,runtime)


def test_native_crash_hold_precedes_worker_library_loading(prepared,tmp_path,compiled,monkeypatch):
    import siesta_remag.research.preflight as preflight
    runtime=held_runtime(tmp_path,compiled)
    def unexpected(*args):raise AssertionError('Held runtime touched native files/library')
    monkeypatch.setattr(preflight,'sha256',unexpected)
    monkeypatch.setattr(preflight,'probe_library',unexpected)
    with pytest.raises(ValueError,match='Native execution held'):verify_runtime_files(runtime)


def test_native_crash_hold_does_not_relabel_unrelated_or_historical_paths(prepared,tmp_path,compiled):
    runtime=pin_runtime(tmp_path,compiled)
    assert preflight_query(prepared,runtime)['mesh_capacity_passed']
    runtime=held_runtime(tmp_path,compiled);runtime['environment']['REMAG_DEVICE_SCF']='0'
    result=preflight_query(prepared,runtime)
    assert result['mesh_capacity_passed'] is None and not result['device_scf_enabled']
    assert 'not numerical acceptance' in result['scope']


@pytest.mark.parametrize('proof',['valid','missing','wrong_library'])
def test_collected_build_proof_required_for_usable_result(prepared,tmp_path,compiled,monkeypatch,proof):
    import test_research_execution as fixture
    r=pin_runtime(tmp_path,compiled);original=fixture.pack
    def collect(remote,**kwargs):
        top=load(remote/'report.json')
        if proof!='missing':
            from siesta_remag.research import runtime_policy
            monkeypatch.setattr(runtime_policy,'observe',lambda:dict(cuda_runtime=12060,cuda_driver_api=13030,nvidia_driver_versions=['610.57.04']))
            top['runtime_verification']=verify_runtime_files(r,require_deployment=True)
            if proof=='wrong_library':top['runtime_verification']['capabilities']['library_sha256']='0'*64
        save(remote/'report.json',top)
        return original(remote,**kwargs)
    monkeypatch.setattr(fixture,'pack',collect)
    t=SyntheticTransport();result=execute_query(tmp_path,'execution-config.json','bundle','attempt',transport=t)
    assert result['allocation_stopped'] and len(t.calls)==1
    if proof=='valid':
        assert result['status']=='measured',result
        assert result['review']['runtime_verification']['actual_files_verified']
        assert result['usable_result_seconds']>0
    else:
        assert result['status']=='held_review' and result['usable_result_seconds'] is None
        assert ('capability verification' in result['review_error']['message'] or
                (proof=='missing' and 'Missing native runtime admission' in result['review_error']['message']))
        with pytest.raises(ValueError,match='Previous execution needs'):
            execute_query(tmp_path,'execution-config.json','bundle','next',transport=t)
        assert len(t.calls)==1


def test_export_and_history_policy_coverage_is_offline():
    from siesta_remag.research.preflight import preflight_input_audits
    env = {'REMAG_INTEGRAL_FORCE':'1', 'REMAG_GATE_H_EXPORT':'1', 'REMAG_GPU_HISTORY':'1'}
    text = 'Write.DM false\nWrite.DM.End.Of.Cycle true\nSaveHS true\nMD.TypeOfRun CG\n'
    r = preflight_input_audits(text, env)
    assert r['hamiltonian_exports'] == 4 and r['density_exports'] == 2
    assert r['density_history']['depth'] == 1
    assert preflight_input_audits(text.replace('CG', 'Broyden'), env) == r
    for bad in ('FIRE', 'FC', 'Verlet'):
        with pytest.raises(ValueError, match='MD mode'):
            preflight_input_audits(text.replace('CG', bad), env)
    with pytest.raises(ValueError, match='analysis consumer'):
        preflight_input_audits(text+'OpticalCalculation true\n', env)
    assert preflight_input_audits(text.replace('CG', 'Broyden'), {}) == {}


def test_unqualified_input_policy_cannot_reach_query_account(prepared, tmp_path):
    runtime = load(tmp_path/'runtime.json')
    runtime['environment'] = {'REMAG_INTEGRAL_FORCE':'1', 'REMAG_GATE_H_EXPORT':'1'}
    save(tmp_path/'runtime.json', runtime)
    config = load(tmp_path/'execution-config.json')
    config['runtime_sha256'] = sha256(tmp_path/'runtime.json')
    save(tmp_path/'execution-config.json', config)
    p = prepared/'scalar/input.fdf'; p.write_text(p.read_text()+'\nOpticalCalculation true\n')
    m = load(prepared/'scalar/manifest.json'); m['input_sha256'] = sha256(p); save(prepared/'scalar/manifest.json', m)
    q = load(prepared/'experiment.json')
    for name in q['files']: q['files'][name] = sha256(prepared/name)
    save(prepared/'experiment.json', q)
    before = (tmp_path/'budget.json').read_bytes(); t = SyntheticTransport()
    with pytest.raises(ValueError, match='analysis consumer'):
        execute_query(tmp_path, 'execution-config.json', 'bundle', 'attempt', transport=t)
    assert not t.calls and not t.info_calls
    assert not (tmp_path/'attempt').exists() and not (tmp_path/'.research-execution').exists()
    assert (tmp_path/'budget.json').read_bytes() == before


def test_general_material_native_incident_hold_preserves_saved_inspection():
    from siesta_remag.research.preflight import check_execution_hold
    r = dict(executable='/results/build/siesta',
        executable_sha256='68fbdc97c03e00d4088b1e4454cb42284eb0af28735e5eabd87e5b8d2e36c0fd',
        environment={'REMAG_DEVICE_SCF':'1','LD_LIBRARY_PATH':'/results/build'}, threads=1,
        capabilities=dict(format='REMAG-BOUND-CAPABILITIES/1',
            executable_sha256='68fbdc97c03e00d4088b1e4454cb42284eb0af28735e5eabd87e5b8d2e36c0fd',
            library='/results/build/libremag_gpu.so',
            library_sha256='e96e212fc6553632b4a66ff0bc5a3b5355bade9b4fee0f09b44b9e09b5774754',
            device_scf=dict(format='REMAG-DEVICE-SCF-CAPABILITIES/1', max_kpoints=4096,
                            max_spatial_orbitals=2048, max_spinor_dimension=4096)))
    assert validate_runtime(r) == r['capabilities']
    with pytest.raises(ValueError, match='general-material-loop-r1 SIGSEGV'):
        check_execution_hold(r)
    r['environment']['CUDA_DISABLE_PTX_JIT'] = '1'
    r['executable'] = '/another/alias/siesta'
    with pytest.raises(ValueError, match='no spending'):
        check_execution_hold(r)
