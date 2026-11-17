"""Fresh binary research scope rejects incompatible writes before execution."""
import pytest

pytest.importorskip('gpac')
from siesta_remag.discovery.common import save,load
from siesta_remag.research.tools import ResearchTools
from siesta_remag.research.authoring import register_structure, workspace_source
from siesta_remag.research.material_scope import validate_scope

SCOPE=dict(format='REMAG-MATERIAL-SCOPE/1',allowed_atomic_numbers=[26,27],
           required_atomic_numbers=[26,27],max_atoms=8)


def recipe(elements):
    return dict(op='construct',recipe=dict(gram=[[9,0,0],[0,9,0],[0,0,9]],
        components=[dict(id='site'+str(i),z=z,shape=dict(kind='points',points=[[str(i)+'/2']*3]))
                    for i,z in enumerate(elements)]))


@pytest.mark.parametrize('elements',[[26],[27],[28],[26,78],[26,28]])
def test_nonbinary_candidate_rejected_before_files(tmp_path,elements):
    save(tmp_path/'research-material-scope.json',SCOPE)
    with pytest.raises(ValueError,match='material scope'):
        ResearchTools(tmp_path).call('author_structure',dict(request=recipe(elements),output='candidate'))
    assert not (tmp_path/'candidate').exists()


def test_binary_authoring_and_ide_source_guard(tmp_path):
    tools=ResearchTools(tmp_path)
    foreign=tools.call('author_structure',dict(request=recipe([26,78]),output='foreign'))
    save(tmp_path/'research-material-scope.json',SCOPE)
    tools=ResearchTools(tmp_path)
    good=tools.call('author_structure',dict(request=recipe([26,27]),output='binary'))
    register_structure(tmp_path,good['structure_path'],'reference','Known binary control')
    with pytest.raises(ValueError,match='material scope'):
        register_structure(tmp_path,foreign['structure_path'],'foreign','Foreign')
    structure=load(tmp_path/good['structure_path'])
    bad=load(tmp_path/foreign['structure_path'])
    for saving in (False,True):
        payload=dict(source_id='reference',expected_source_sha256=structure['source']['sha256'],program=bad['source']['program'])
        if saving:payload['label']='Out of scope'
        with pytest.raises(ValueError,match='material scope'):workspace_source(tmp_path,payload,save_source=saving)
    assert len(load(tmp_path/'workspace.json')['structures'])==1
    for tool,args in [('prepare_query',dict(structure=foreign['structure_path'],pseudos='absent',output='query',query=dict(kind='scalar'))),
                      ('prepare_relaxation',dict(structure=foreign['structure_path'],pseudos='absent',weights='absent',output='relax'))]:
        with pytest.raises(ValueError,match='material scope'):tools.call(tool,args)
    assert not (tmp_path/'query').exists() and not (tmp_path/'relax').exists()


@pytest.mark.parametrize('patch',[dict(allowed_atomic_numbers=[True,27]),dict(required_atomic_numbers=[28]),
    dict(allowed_atomic_numbers=[26,26]),dict(max_atoms=True),dict(max_atoms=1),dict(unknown=True)])
def test_invalid_scope_fails_closed(patch):
    with pytest.raises(ValueError):validate_scope(SCOPE|patch)


def test_new_workspace_has_no_old_memory_and_cannot_read_it(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    ResearchTools(old).call('record_hypothesis',dict(statement='Old Ni result',predictions=['Old'],alternatives=['Other']))
    old_bytes=(old/'research-memory.sqlite').read_bytes()
    assert ResearchTools(new).call('retrieve_research_history',{})==[]
    with pytest.raises(ValueError,match='escapes'):ResearchTools(new).path('../old/research-memory.sqlite')
    assert (old/'research-memory.sqlite').read_bytes()==old_bytes
