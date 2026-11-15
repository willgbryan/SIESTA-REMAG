"""Synthetic native handoff fixtures; these are engineering tests, not DFT data."""
import json
import shutil

import pytest

from test_discovery import bundle, backend, execute
from siesta_remag.discovery.common import load,save
from siesta_remag.corpus import sha256
from siesta_remag.research.relaxation import inspect_relaxation,prepare_relaxation
from siesta_remag.research.tools import ResearchTools


@pytest.fixture
def export():
    pytest.importorskip('gpac')
    from gpac import ir,physical
    p=ir.author_program(dict(hall=1,gram=[[9,0,0],[0,9,0],[0,0,9]],
        orbits=[dict(z=26,wyckoff_index=0,parameters=[x]*3) for x in [0,'1/2']]))
    return physical.export_geometry(p,length_unit='angstrom')


def test_saved_relaxation_tool_is_source_bound(bundle,tmp_path,export):
    calls=[];report=execute(bundle,tmp_path,calls)
    assert report['status']=='screened'
    save(tmp_path/'source.json',export)
    result=ResearchTools(tmp_path).call('inspect_relaxation',dict(structure='source.json',bundle='bundle',report='run/report.json'))
    assert len(calls)==4 and result['solver_executed'] is False
    assert result['feedback']['symmetry']['all_authored_operations_preserved']
    assert result['feedback']['observation']['max_nonaffine_displacement_angstrom']<1e-12
    assert 'retrospective' in result['source_binding']
    assert result['source_sha256']==export['source']['sha256']
    (tmp_path/'run/dft-relax/remag.STRUCT_OUT').write_text('changed')
    with pytest.raises(ValueError,match='artifact changed'):
        inspect_relaxation(export,bundle,tmp_path/'run/report.json')


def test_failed_relaxation_or_wrong_source_is_not_feedback(bundle,tmp_path,export):
    execute(bundle,tmp_path,[],fail='optimizer')
    with pytest.raises(ValueError,match='No accepted'):
        inspect_relaxation(export,bundle,tmp_path/'run/report.json')
    from gpac import physical
    wrong=physical.export_geometry(physical.fept_reference(),length_unit='angstrom')
    with pytest.raises(ValueError,match='source differs'):
        inspect_relaxation(wrong,bundle,tmp_path/'run/report.json')


def test_preparation_binds_source_before_cpu_work(bundle,tmp_path,export,monkeypatch):
    from siesta_remag.research import relaxation
    calls=[]
    def prepare(source,pseudos,weights,destination,settings,**kwargs):
        assert kwargs['orb_device']=='cpu'
        assert load(destination.parent/'source-binding.json')['input_sha256']==sha256(source)
        shutil.copytree(bundle,destination)
        request=load(destination/'request.json');request['input_sha256']=sha256(source)
        save(destination/'request.json',request)
        from siesta_remag.discovery.common import digest
        receipt=load(destination/'prepared.json');receipt['request_sha256']=digest(request)
        receipt['files']['request.json']=sha256(destination/'request.json');save(destination/'prepared.json',receipt)
        calls.append(source)
        return receipt
    monkeypatch.setattr(relaxation,'prepare_bundle',prepare)
    p=prepare_relaxation(export,bundle/'pseudos',tmp_path/'weights',tmp_path/'prepared')
    assert len(calls)==1 and p['native_submitted'] is False
    execute(tmp_path/'prepared/bundle',tmp_path,[])
    result=inspect_relaxation(export,tmp_path/'prepared/bundle',tmp_path/'run/report.json')
    assert result['source_binding']=='authored source bound before CPU preparation'
    with pytest.raises(FileExistsError):
        prepare_relaxation(export,bundle/'pseudos',tmp_path/'weights',tmp_path/'prepared')
    binding=load(tmp_path/'prepared/source-binding.json');binding['source_sha256']='0'*64
    save(tmp_path/'prepared/source-binding.json',binding)
    with pytest.raises(ValueError,match='source binding differs'):
        inspect_relaxation(export,tmp_path/'prepared/bundle',tmp_path/'run/report.json')
