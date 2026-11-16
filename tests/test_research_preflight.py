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


@pytest.mark.parametrize('proof',['valid','missing','wrong_library'])
def test_collected_build_proof_required_for_usable_result(prepared,tmp_path,compiled,monkeypatch,proof):
    import test_research_execution as fixture
    r=pin_runtime(tmp_path,compiled);original=fixture.pack
    def collect(remote,**kwargs):
        top=load(remote/'report.json')
        if proof!='missing':
            top['runtime_verification']=verify_runtime_files(r)
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
        assert 'capability verification' in result['review_error']['message']
        with pytest.raises(ValueError,match='Previous execution needs'):
            execute_query(tmp_path,'execution-config.json','bundle','next',transport=t)
        assert len(t.calls)==1
