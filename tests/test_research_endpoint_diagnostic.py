import numpy as np
import pytest
from ase import Atoms
from test_research_relaxation_execution import discovery, DiscoveryTransport, run
from test_research_execution import SyntheticTransport

from siesta_remag.discovery.common import load, save
from siesta_remag.discovery.config import Settings
from siesta_remag.research.endpoint_diagnostic import endpoint_order
from siesta_remag.research.execution import execute_query
from siesta_remag.research.experiments import prepare_experiment


def test_endpoint_site_and_frame_binding():
    original = Atoms('Fe2Co', scaled_positions=[[.2,.3,.4],[.1,.2,.4],[.5,.5,.5]],cell=[2,3,4],pbc=True)
    trial = original[[1,0,2]]
    trial.positions[0] += trial.cell[0]
    trial.rotate(15, 'y', rotate_cell=True)
    np.testing.assert_array_equal(endpoint_order(original,trial),[1,0,2])
    reflected = original.copy()
    reflected.set_cell(original.cell.array @ np.diag([-1,1,1]),scale_atoms=True)
    with pytest.raises(ValueError,match='terminal cell metric'):
        endpoint_order(reflected,original)
    for change in ('site','translation','strain','duplicate','identity'):
        bad = trial.copy()
        if change=='site': bad.positions[0,1] += .0001
        if change=='translation': bad.translate([0,.001,0])
        if change=='strain': bad.set_cell(1.001*bad.cell,scale_atoms=True)
        if change=='duplicate': bad.positions[0] = bad.positions[1]
        if change=='identity': bad.numbers[2] = 26
        with pytest.raises(ValueError,match='terminal cell metric|original endpoint sites'):
            endpoint_order(original,bad)


def test_stopped_step_limit_admits_only_the_matched_endpoint(discovery,tmp_path,monkeypatch):
    prev = DiscoveryTransport(monkeypatch,failure='step_limit')
    old = run(tmp_path,prev)
    assert old['status']=='held_native_review' and old['allocation_stopped']
    config = load(tmp_path/'config.json')
    config.update(operation='query', previous_app_id=old['app_id'], authorization_id='endpoint',
                  max_queries=1,relaxation_endpoint_report_sha256=old['review']['report_sha256'])
    save(tmp_path/'endpoint.json',config)
    dest = tmp_path/'scalar-query'
    saved = load(discovery/'request.json')
    export = load(tmp_path/'structure.json')
    prepare_experiment(export,discovery/'pseudos',dest,settings=Settings.from_dict(saved['settings']),
                       mesh=load(discovery/'prepared.json')['kmesh'],query={'kind':'scalar'})
    t = SyntheticTransport()
    # The relaxation fixture explicitly seeds 2 muB on both elements. A scalar
    # query uses the same default for Co and a different default for Ni.
    if export['geometry']['atomic_numbers'][0]==28:
        with pytest.raises(ValueError,match='input does not reproduce'):
            execute_query(tmp_path,'endpoint.json',dest,'diagnostic',transport=t)
        assert not t.calls
    else:
        result=execute_query(tmp_path,'endpoint.json',dest,'diagnostic',transport=t)
        assert len(t.calls)==1 and len(t.native)==1 and result['allocation_stopped']
        assert result['status']=='measured', result
        assert load(tmp_path/'attempt/execution.json')==old
