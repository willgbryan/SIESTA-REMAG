import numpy as np
import pytest
from test_research_inspection import spectrum_fixture
from siesta_remag.discovery.common import load,save
from siesta_remag.corpus import sha256
from siesta_remag.research.inspection import inspect_electronic,projected_dos,compare_electronic
from siesta_remag.research.reciprocal import region_indices, path_geometry

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

def test_large_browser_samples_do_not_silently_reduce_the_mesh():
    from siesta_remag.research.reciprocal import sampled_view
    result=sampled_view({'fermi_ev':0},{'eigenvalues_ev':np.zeros((4096,80))},2)
    assert result['available'] is False and 'k_fractional' not in result
    assert 'artifact tools' in result['reason']
    assert not sampled_view({'fermi_ev':0},{'eigenvalues_ev':np.zeros((4097,1))},1)['available']
