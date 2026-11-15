import json

import numpy as np
import pytest

from test_research import experiment
from test_discovery import backend
from test_research_inspection import spectrum_fixture
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load,save
from siesta_remag.research.catalog import electronic_view
from siesta_remag.research.experiments import run_experiment
from siesta_remag.research.workspace import dataset,snapshot


def test_general_sources_and_source_bound_calculation(experiment,tmp_path):
    from gpac import ir,physical
    other=ir.author_program(dict(hall=1,gram=[[16,0,0],[0,16,0],[0,0,25]],
        orbits=[dict(z=z,wyckoff_index=0,parameters=q) for z,q in [(26,[0,0,0]),(26,['1/2','1/2','1/2']),(5,['1/4','1/4','1/4'])]]))
    save(tmp_path/'other.json',physical.export_geometry(other,length_unit='angstrom'))
    binary=tmp_path/'binary';binary.write_text('fixture')
    run_experiment(experiment,tmp_path/'run',binary,executable_sha256=sha256(binary),runner=backend([]))
    manifest=dict(format='REMAG-RESEARCH-WORKSPACE/1',title='Research',structures=[
        dict(id='base',label='Baseline',structure='bundle/structure.json',calculations=[dict(report='run/report.json')]),
        dict(id='fe2b',label='Fe₂B authored candidate',structure='other.json')])
    save(tmp_path/'workspace.json',manifest)
    state=dataset(tmp_path)
    assert state['mode']=='registered' and len(state['families'])==2
    assert state['families'][0]['rows'][0]['result']['status']=='measured'
    assert state['families'][1]['rows'][0]['structure']['geometry']['atomic_numbers'].count(5)==1
    assert state['families'][1]['rows'][0]['result'] is None
    assert state['element_symbols'][5]=='B'
    snapshot(tmp_path,tmp_path/'view.html')
    assert 'id="saved-state"' in (tmp_path/'view.html').read_text()
    manifest['structures'][1]['calculations']=[dict(report='run/report.json')];save(tmp_path/'workspace.json',manifest)
    with pytest.raises(ValueError,match='another registered source'):dataset(tmp_path)
    manifest['structures'][1].pop('calculations');manifest['structures'][1]['structure']='../escape.json';save(tmp_path/'workspace.json',manifest)
    with pytest.raises(ValueError,match='escapes'):dataset(tmp_path)


def test_preview_retains_signed_atom_state_weights_and_bound_hamiltonian(tmp_path):
    path=spectrum_fixture(tmp_path/'spectrum')
    view=electronic_view(path,'a'*64,2)
    state=next(s for s in view['states'] if s['k_index']==0 and s['band_index']==0)
    assert state['orbital_weights']==[1.1,-.1]
    assert state['spin_pauli']==[0,0,1]
    assert len(view['groups'])==2 and view['omitted_states']==0
    # Integral of the sum over all atomic projections has four sampled states.
    summed=np.sum([g['density'] for g in view['groups']],axis=0)
    assert np.trapezoid(summed,view['energy_ev'])==pytest.approx(2,abs=1e-4)
    with pytest.raises(ValueError,match='another displayed Hamiltonian'):electronic_view(path,'b'*64,2)
    with pytest.raises(ValueError,match='does not belong'):electronic_view(path,'a'*64,1)


def test_catalog_escapes_embedded_markup(experiment,tmp_path):
    save(tmp_path/'workspace.json',dict(format='REMAG-RESEARCH-WORKSPACE/1',title='Example',structures=[
        dict(id='base',label='</script><script>bad()</script>',structure='bundle/structure.json')]))
    snapshot(tmp_path,tmp_path/'view.html')
    assert '</script><script>bad()' not in (tmp_path/'view.html').read_text()
