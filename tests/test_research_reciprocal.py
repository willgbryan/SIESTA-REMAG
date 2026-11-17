from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pytest

from test_research_inspection import spectrum_fixture
from test_research_mechanisms import native_fixture
from test_research import experiment
from test_discovery import backend
from siesta_remag.discovery.common import load,save
from siesta_remag.corpus import sha256
from siesta_remag.research.inspection import inspect_electronic,projected_dos,compare_electronic
from siesta_remag.research.reciprocal import region_indices,path_geometry,export_band_path
from siesta_remag.research.electronic import spectrum_from_run
from siesta_remag.research.tools import ResearchTools


def test_periodic_regions_keep_original_weights_and_empty_selections(tmp_path):
    ks=[[.49,0,0],[-.49,0,0],[.2,0,0],[1.49,0,0]]
    region=dict(center=[.5,0,0],half_width=[.02,.1,.1])
    assert region_indices(ks,region)==[0,1,3]
    assert region_indices(ks,dict(center=[1.5,0,0],half_width=[.02,.1,.1]))==[0,1,3]
    p=spectrum_fixture(tmp_path/'s')
    selection=dict(k_region=dict(center=[.5,0,0],half_width=[.01,.1,.1]),atom_indices=[1])
    result=inspect_electronic(p,selection)
    assert result['selected_k_weight']==.75
    assert {s['k_index'] for s in result['states']}=={1}
    assert any(s['selected_mulliken_weight']<0 for s in inspect_electronic(p,dict(k_indices=[0],atom_indices=[1]))['states'])
    dos=projected_dos(p,selection);equivalent=projected_dos(p,dict(k_indices=[1],atom_indices=[1]))
    assert dos['density_states_per_ev']==equivalent['density_states_per_ev']
    empty=dict(k_region=dict(center=[.25,0,0],half_width=[.01,.1,.1]))
    assert inspect_electronic(p,empty)['states']==[]
    assert projected_dos(p,empty)['density_states_per_ev']==[0.]*161
    with pytest.raises(ValueError,match='Choose'):inspect_electronic(p,selection|{'k_indices':[0]})


def test_region_comparison_rejects_boundary_membership_changes(tmp_path):
    paths=[spectrum_fixture(tmp_path/n) for n in ('a','b')]
    for path,x in zip(paths,[.9e-12,1.1e-12]):
        artifact=path.parent/'spectrum.npz'
        with np.load(artifact) as source:arrays={k:source[k].copy() for k in source.files}
        arrays['k_fractional'][0,0]=x;np.savez(artifact,**arrays)
        meta=load(path);meta['artifact_sha256']=sha256(artifact);save(path,meta)
    with pytest.raises(ValueError,match='membership differs'):
        compare_electronic(*paths,selection=dict(k_region=dict(center=[0,0,0],half_width=[0,.1,.1])))


@pytest.mark.parametrize('region',[{},dict(center=[True,0,0],half_width=[.1]*3),dict(center=[0]*3,half_width=[-.1]*3),
    dict(center=[0]*3,half_width=[.6]*3),dict(center=[float('nan'),0,0],half_width=[.1]*3)])
def test_region_rejects_ambiguous_or_invalid_coordinates(region):
    with pytest.raises(ValueError):region_indices([[0,0,0]],region)


def test_skew_cell_band_metric_breaks_and_repeated_endpoints():
    cell=np.array([[2,0,0],[1,3,0],[.2,.3,4]])
    ks=np.array([[0,0,0],[.5,0,0],[.5,.5,0],[0,.5,0]])
    distance,b=path_geometry(ks,cell,[0,1,2,0])
    assert np.allclose(b@cell.T,2*np.pi*np.eye(3))
    assert distance[-1]==pytest.approx(sum(np.linalg.norm(v) for v in np.diff(ks[[0,1,2,0]],axis=0)@b))
    distance,_=path_geometry(ks,cell,[0,1,3,2],breaks=[2])
    assert distance[1]==distance[2]
    for ids,breaks in [([True,1],None),([0,0],None),([0,1],[1]),([0,1],[2]),([0,1],[1,1])]:
        with pytest.raises(ValueError):path_geometry(ks,cell,ids,breaks)


def test_native_band_export_cli_agent_and_access_controls(tmp_path,capsys):
    report,native=native_fixture(tmp_path/'native');r=load(report);r['status']='measured';save(report,r)
    spectrum_from_run(native,tmp_path/'spectrum');spectrum=tmp_path/'spectrum/electronic.json'
    original={p:sha256(p) for p in native.iterdir()}
    args=dict(report='native/report.json',stage='soc-z',spectrum='spectrum/electronic.json',native_hsx='native/soc-z/remag.HSX',output='bands',k_indices=[0,1,0],selection={'atom_indices':[1]})
    result=ResearchTools(tmp_path).call('export_band_path',args)
    assert result['source_sha256']=='a'*64 and result['band_count']==4 and not result['quadrature_defined']
    with np.load(tmp_path/'bands/bands.npz') as bands,np.load(tmp_path/'spectrum/spectrum.npz') as spec:
        assert np.array_equal(bands['eigenvalues_ev'],spec['eigenvalues_ev'][[0,1,0]])
        assert np.array_equal(bands['selected_mulliken'],spec['orbital_mulliken'][[0,1,0],1,:])
        assert bands['distance_inv_angstrom'][-1]==pytest.approx(1/.529177210903)
    with pytest.raises(ValueError,match='scalar profile'):ResearchTools(tmp_path,information_profile='scalar').call('export_band_path',args)
    with pytest.raises(ValueError,match='held-out'):ResearchTools(tmp_path,withheld_sources=['a'*64]).call('export_band_path',args)
    from siesta_remag.research.__main__ import main
    main(['bands',str(report),'--stage','soc-z','--spectrum',str(spectrum),'--native-hsx',str(native/'remag.HSX'),'--k-indices','0','1','--output',str(tmp_path/'cli')])
    assert json.loads(capsys.readouterr().out)['format']=='REMAG-BAND-PATH/1'
    bad=tmp_path/'wrong.HSX';bad.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='Hamiltonian binding'):export_band_path(report,'soc-z',spectrum,bad,tmp_path/'wrong',k_indices=[0,1])
    assert not (tmp_path/'wrong').exists()
    archive=tmp_path/'spectrum/spectrum.npz';original_archive=archive.read_bytes();original_meta=spectrum.read_bytes()
    with np.load(archive) as data:changed={k:data[k].copy() for k in data.files}
    changed['k_fractional'][0,0]+=.01;np.savez(archive,**changed)
    altered=load(spectrum);altered['artifact_sha256']=sha256(archive);save(spectrum,altered)
    with pytest.raises(ValueError,match='bound native sampling'):export_band_path(report,'soc-z',spectrum,native/'remag.HSX',tmp_path/'changed',k_indices=[0,1])
    assert not (tmp_path/'changed').exists()
    archive.write_bytes(original_archive);spectrum.write_bytes(original_meta)
    r['stages'][0]['branch']['passed']=False;save(report,r)
    with pytest.raises(ValueError,match='not passed'):export_band_path(report,'soc-z',spectrum,native/'remag.HSX',tmp_path/'bad',k_indices=[0,1])
    assert not (tmp_path/'bad').exists()
    assert all(sha256(p)==h for p,h in original.items())


def test_large_browser_samples_do_not_silently_reduce_the_mesh():
    from siesta_remag.research.reciprocal import sampled_view
    result=sampled_view({'fermi_ev':0},{'eigenvalues_ev':np.zeros((4096,80))},2)
    assert result['available'] is False and 'k_fractional' not in result
    assert 'artifact tools' in result['reason']
    assert not sampled_view({'fermi_ev':0},{'eigenvalues_ev':np.zeros((4097,1))},1)['available']


def test_browser_transport_deduplicates_aliases_without_mutating_dataset(tmp_path):
    from siesta_remag.research.catalog import electronic_view
    from siesta_remag.research.workspace import browser_state
    p=spectrum_fixture(tmp_path/'s');view=electronic_view(p,'a'*64,2)
    source={'a':view,'b':[view]};wire=browser_state(source)
    assert len(wire['sampled_arrays'])==1
    assert wire['a']['sampled']==wire['b'][0]['sampled']=={'ref':'sample-0'}
    assert source['a']['sampled']['available'] is True
    assert wire['sampled_arrays']['sample-0']==source['a']['sampled']


@pytest.mark.parametrize('mode',['embedded','deferred'])
def test_shipped_reciprocal_client_updates_dos_and_state_selection(experiment,tmp_path,mode):
    if not shutil.which('node'):pytest.skip('Node required for client protocol test')
    from siesta_remag.research.experiments import run_experiment
    from siesta_remag.research.workspace import dataset,browser_state
    from siesta_remag.research.catalog import electronic_view
    binary=tmp_path/'binary';binary.write_text('fixture')
    run_experiment(experiment,tmp_path/'run',binary,executable_sha256=sha256(binary),runner=backend([]))
    save(tmp_path/'workspace.json',dict(format='REMAG-RESEARCH-WORKSPACE/1',title='Research',structures=[
        dict(id='base',label='Baseline',structure='bundle/structure.json',calculations=[dict(report='run/report.json')])]))
    state=dataset(tmp_path);p=spectrum_fixture(tmp_path/'spectrum');view=electronic_view(p,'a'*64,2)
    state['families'][0]['rows'][0]['calculations'][0]['spectra']={'soc-z':view};save(tmp_path/'state.json',browser_state(state))
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['node',str(root/'tests/fixtures/workspace_reciprocal_dom.cjs'),str(root/'src/siesta_remag/research/workspace.html'),str(tmp_path/'state.json'),mode],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
