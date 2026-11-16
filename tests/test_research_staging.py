from copy import deepcopy
import json

import pytest

from test_research_convergence import report
from test_research_calibration import compact
from test_research_study import study_fixture
from siesta_remag.discovery.common import digest, load, save
from siesta_remag.research.staging import validate_scalar_data
from siesta_remag.research.study import register_study
from siesta_remag.research.toolhost import ToolHost
from siesta_remag.research.tools import ResearchTools, STANDARD_OBSERVATIONS


@pytest.fixture
def raw(tmp_path):
    r = report(6, .001311, .001285)
    r['artifacts'] = {}
    for row in r['observations'].values():
        row.update(eigenvalues={'values_ev': ['secret-band']}, fermi_energy_ev=-2.5,
                   soc_energy_ev=.2, atom_net_charges_e=[.3, -.3], future_diagnostic='secret-future',
                   total_moment_muB=[0., 0., 3.4], scf_iterations=15)
    # Inspection metadata is not safe to copy wholesale either.
    r['request']['future_diagnostic'] = 'secret-request'
    r['stages'][0]['future_diagnostic'] = 'secret-stage'
    r['anisotropy']['future_diagnostic'] = 'secret-anisotropy'
    save(tmp_path/'raw/report.json', r)
    return r


def stage(tmp_path, output='safe/packet.json', **options):
    t = ResearchTools(tmp_path, **options)
    return t.call('stage_scalar_data', dict(calculations={'baseline': {'report': 'raw/report.json'}}, output=output))


def test_operator_staging_regression_never_copies_electronic_or_unknown_fields(raw, tmp_path):
    for arm in ('scalar', 'physics'):
        host = ToolHost(tmp_path, information_profile=arm)
        reply = host.call(dict(tool='stage_scalar_data', args=dict(
            calculations={'baseline': {'report': 'raw/report.json'}}, output=arm+'/packet.json'),
            rationale='Stage shared initial data without raw archives'), 'calls/'+arm)
        assert reply['ok'] and reply['result']['raw_artifacts_copied'] == 0
        assert [p.name for p in (tmp_path/arm).iterdir()] == ['packet.json']
    assert (tmp_path/'scalar/packet.json').read_bytes() == (tmp_path/'physics/packet.json').read_bytes()
    packet = ResearchTools(tmp_path, information_profile='scalar').call('inspect_scalar_data', {'packet':'scalar/packet.json'})
    encoded = json.dumps(packet)
    for key in ('eigenvalues', 'fermi_energy_ev', 'soc_energy_ev', 'atom_net_charges_e', 'future_diagnostic', 'secret-'):
        assert key not in encoded
    row = packet['cases']['baseline']['observations']['soc-x']
    assert set(row) <= STANDARD_OBSERVATIONS
    assert row['total_energy_ev'] == .001311 and row['total_moment_muB'] == [0.,0.,3.4]
    assert load(tmp_path/'raw/report.json') == raw  # Preserve original evidence.
    with pytest.raises(ValueError, match='must be new'):stage(tmp_path, 'scalar/packet.json')


@pytest.mark.parametrize('mutation', ['extra_observable','nested_payload','extra_metadata','digest'])
def test_packet_preflight_rejects_extra_fields_and_payloads_even_when_rehashed(raw, tmp_path, mutation):
    stage(tmp_path); packet=load(tmp_path/'safe/packet.json'); row=packet['cases']['baseline']
    if mutation=='extra_observable':row['observations']['soc-x']['eigenvalues']=[99]
    elif mutation=='nested_payload':row['observations']['soc-x']['total_moment_muB']=[{'bands':[99]},0,0]
    elif mutation=='extra_metadata':row['request']={'bands':[99]}
    else:row['observations']['soc-x']['total_energy_ev']=123
    if mutation!='digest':packet['sha256']=digest({k:v for k,v in packet.items() if k!='sha256'})
    with pytest.raises(ValueError):validate_scalar_data(packet)


def test_verified_collection_and_holdouts_are_enforced_before_writing(raw,tmp_path):
    endpoint=compact([tmp_path/'raw/report.json'],tmp_path/'collected')[0]
    args=dict(calculations={'baseline':{k:str(v.relative_to(tmp_path)) for k,v in endpoint.items()}},output='safe/packet.json')
    t=ResearchTools(tmp_path)
    t.call('stage_scalar_data',args)
    held=ResearchTools(tmp_path,withheld_sources=['a'*64])
    with pytest.raises(ValueError,match='held-out'):held.call('stage_scalar_data',dict(args,output='blocked.json'))
    with pytest.raises(ValueError,match='held-out'):held.call('inspect_scalar_data',{'packet':'safe/packet.json'})
    assert not (tmp_path/'blocked.json').exists()
    (endpoint['report'].parent/'soc-x/stdout.log').write_text('corrupted')
    with pytest.raises(ValueError,match='artifact binding'):t.call('stage_scalar_data',dict(args,output='bad.json'))
    assert not (tmp_path/'bad.json').exists()


@pytest.mark.parametrize('kind',['raw_report','original_leak','bad_packet','heldout_packet','nested_heldout_packet','safe'])
def test_study_preflight_blocks_leaked_common_packets_before_context_start(raw,tmp_path,kind):
    d=study_fixture(tmp_path)
    stage(tmp_path);packet=load(tmp_path/'safe/packet.json')
    if kind=='raw_report':data=raw
    elif kind=='original_leak':data={'baseline-k6':dict(observations=raw['observations'],anisotropy_mev={'E':1.311})}
    elif kind=='bad_packet':
        data=deepcopy(packet);data['cases']['baseline']['observations']['soc-x']['fermi_energy_ev']=-2.5
    elif kind in ('heldout_packet','nested_heldout_packet'):
        data=deepcopy(packet);data['cases']['baseline']['source_sha256']='c'*64
        data['sha256']=digest({k:v for k,v in data.items() if k!='sha256'})
        if kind=='nested_heldout_packet':data={'wrapped':[data]}
    else:data=packet
    for arm in ('scalar','physics'):save(tmp_path/('arm-'+arm)/'common.json',data)
    if kind=='safe':assert register_study(tmp_path,d,'study')['native_budget_per_arm'] is None
    else:
        with pytest.raises(ValueError):register_study(tmp_path,d,'study')
        assert not (tmp_path/'study').exists()
    assert not (tmp_path/'arm-scalar/session').exists()


def test_nested_scalar_values_and_workspace_escapes_fail_before_output(raw,tmp_path):
    raw['observations']['soc-x']['total_energy_ev']={'secret':[1]}
    save(tmp_path/'raw/report.json',raw)
    with pytest.raises(ValueError,match='scalar observation'):stage(tmp_path)
    assert not (tmp_path/'safe/packet.json').exists()
    with pytest.raises(ValueError,match='escapes'):stage(tmp_path,'../escape.json')
