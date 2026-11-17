"""Initial UI payload excludes complete arrays; explicit reads retain all data."""
import json
import threading
from urllib.parse import urlsplit, parse_qs
from urllib.request import urlopen
from urllib.error import HTTPError

import pytest
from gpac.errors import GPACError

from test_research import experiment
from test_discovery import backend
from test_research_inspection import spectrum_fixture
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save
from siesta_remag.research.experiments import run_experiment
from siesta_remag.research.workspace import dataset, make_server
from siesta_remag.research.catalog import registered_sampling


@pytest.fixture
def registered(experiment, tmp_path):
    binary=tmp_path/'binary';binary.write_text('fixture')
    run_experiment(experiment,tmp_path/'run',binary,executable_sha256=sha256(binary),runner=backend([]))
    hsx=tmp_path/'run/soc-z/remag.HSX';hsx.write_bytes(b'bound UI fixture only')
    report=load(tmp_path/'run/report.json');report['artifacts']['soc-z/remag.HSX']=sha256(hsx);save(tmp_path/'run/report.json',report)
    spectrum=spectrum_fixture(tmp_path/'spectrum');meta=load(spectrum);meta['hsx_sha256']=sha256(hsx);save(spectrum,meta)
    save(tmp_path/'workspace.json',dict(format='REMAG-RESEARCH-WORKSPACE/1',title='Sampling',structures=[
        dict(id='candidate',label='Saved candidate',structure='bundle/structure.json',
             calculations=[dict(report='run/report.json',spectra={'soc-z':'spectrum/electronic.json'})])]))
    return tmp_path


def deferred(root):
    state=dataset(root,include_sampling=False)
    view=state['families'][0]['rows'][0]['result']['spectra']['soc-z']
    return view,{k:v[0] for k,v in parse_qs(urlsplit(view['sampled']['url']).query).items()}


def test_initial_preview_and_on_demand_sampling_equal_the_full_saved_data(registered):
    full=dataset(registered)['families'][0]['rows'][0]['result']['spectra']['soc-z']
    view,query=deferred(registered)
    assert view['sampled']['available'] is False and view['sampled']['deferred']
    assert 'k_fractional' not in view['sampled']
    assert {k:v for k,v in view.items() if k!='sampled'}=={k:v for k,v in full.items() if k!='sampled'}
    assert registered_sampling(registered,query)==full['sampled']
    assert full['sampled']['k_weights']==[.25,.75]
    assert full['sampled']['atom_mulliken'][0][1][0]==-.1


@pytest.mark.parametrize('change',['source','report','metadata','array','stage','index','escape'])
def test_deferred_read_rechecks_bound_source_report_and_native_artifacts(registered,change):
    _,query=deferred(registered)
    if change=='source':
        p=registered/'bundle/structure.json';d=load(p);d['source']['sha256']='0'*64;save(p,d)
    elif change=='report':query['report_sha256']='0'*64
    elif change=='metadata':
        p=registered/'spectrum/electronic.json';d=load(p);d['fermi_ev']=1;save(p,d)
    elif change=='array':(registered/'spectrum/spectrum.npz').write_bytes(b'changed')
    elif change=='stage':query['stage']='not-registered'
    elif change=='index':query['calculation']='-1'
    else:query['structure']='../../outside'
    with pytest.raises(GPACError if change=='source' else ValueError):registered_sampling(registered,query)


def test_http_loads_only_requested_spectrum_and_rejects_duplicate_selection(registered):
    server=make_server(registered,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/api/state') as r:state=json.load(r)
        sample=state['families'][0]['rows'][0]['result']['spectra']['soc-z']['sampled']
        assert sample['deferred'] and 'sampled_arrays' not in state
        with urlopen(base+sample['url']) as r:loaded=json.load(r)
        assert loaded['available'] and loaded['complete_exported_sampling']
        with pytest.raises(HTTPError) as error:urlopen(base+sample['url']+'&stage=soc-x')
        assert error.value.code==400
    finally:
        server.shutdown();server.server_close();thread.join(5)
