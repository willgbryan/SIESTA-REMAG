"""The advertised phase contract is also the executable action contract."""
import copy
import json

import pytest

from siesta_remag.research.codex_policy import decode
from siesta_remag.research.model_policy import bind_tool_arguments, validate_tool_arguments
from siesta_remag.research.tools import TOOLS


def action(tool, args):
    return json.dumps(dict(tool=tool, arguments_json=json.dumps(args), rationale='Use measured evidence.'))


def test_revision_exposes_and_enforces_the_actual_evidence_path():
    original = copy.deepcopy(TOOLS)
    fixed = dict(parent_id='experiment-1', evidence='scores/iteration-1.json')
    catalog = bind_tool_arguments(TOOLS, 'record_analysis', fixed)
    fixed['evidence'] = 'later-change.json'
    assert catalog['record_analysis']['fixed_arguments']['evidence'] == 'scores/iteration-1.json'
    args = dict(parent_id='experiment-1', evidence='scores/iteration-1.json', conclusion='Unresolved sign; compare a second structure.')
    assert decode(action('record_analysis', args), catalog)['args'] == args
    # The generic tool admits a measurement report; the phase must advertise
    # its stricter score-file binding before asking the model to choose.
    args['evidence'] = 'runs/iteration-1/research-observation/report.json'
    assert decode(action('record_analysis', args), TOOLS)['args'] == args
    with pytest.raises(ValueError, match='advertised binding: evidence'):
        decode(action('record_analysis', args), catalog)
    assert TOOLS == original


@pytest.mark.parametrize('key,bad', [('experiment_id', 'other'), ('observable', 'different'), ('unit', 'meV/cell')])
def test_prediction_identifiers_observable_and_unit_share_one_contract(key, bad):
    bound = dict(experiment_id='experiment-2', observable='anisotropy.total_energy_ev', unit='eV/cell')
    catalog = bind_tool_arguments(TOOLS, 'record_prediction', bound)
    args = dict(bound, value=0.001, uncertainty=0.002, interval_probability=0.8)
    assert decode(action('record_prediction', args), catalog)['args'] == args
    args[key] = bad
    with pytest.raises(ValueError, match='advertised binding: '+key):
        decode(action('record_prediction', args), catalog)


def test_bindings_do_not_leak_between_phases_or_admit_unknown_arguments():
    first = bind_tool_arguments(TOOLS, 'author_structure', {'output': 'candidates/iteration-2'})
    second = bind_tool_arguments(TOOLS, 'author_structure', {'output': 'candidates/iteration-3'})
    args = dict(request={'op':'construct', 'recipe':{}}, output='candidates/iteration-3')
    validate_tool_arguments(args, second['author_structure'])
    with pytest.raises(ValueError, match='advertised binding: output'):
        validate_tool_arguments(args, first['author_structure'])
    with pytest.raises(ValueError, match='belong to the tool catalog'):
        bind_tool_arguments(TOOLS, 'record_analysis', {'shell_command':'false'})
    with pytest.raises(ValueError, match='Invalid fixed argument contract'):
        validate_tool_arguments(dict(parent_id='p', conclusion='c', evidence='e'),
            dict(TOOLS['record_analysis'], fixed_arguments={'unknown':1}))
