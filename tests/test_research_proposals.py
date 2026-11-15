import json
import sys

import pytest

from test_research import experiment
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save
from siesta_remag.research.agent import run_session
from siesta_remag.research.proposals import verify_assessment
from siesta_remag.research.tools import ResearchTools


POLICY = dict(format='GPAC-PROPOSAL-POLICY/1', cutoff_angstrom='3',
              minimum_distance_angstrom='1', max_sites=1, max_orbits=1)


@pytest.mark.parametrize('profile',['scalar','physics'])
def test_saved_assessment_advisory_findings_preserve_query_access(experiment,tmp_path,profile):
    tools=ResearchTools(tmp_path,information_profile=profile)
    assert 'assess_structure' in tools.catalog()
    original=sha256(experiment/'structure.json')
    args=dict(structure='bundle/structure.json',policy=POLICY,output='assessment',reference='bundle/structure.json')
    result=tools.call('assess_structure',args)
    assessment=result['assessment']
    assert {f['code'] for f in assessment['findings']}=={'cell_complexity','orbit_complexity','translational_redundancy'}
    assert assessment['measurements']['reference_cell_strain']['max_absolute_stretch_deviation']==pytest.approx(0)
    assert verify_assessment(tmp_path/'assessment')==result['record']
    assert sha256(experiment/'structure.json')==original
    assert not result['record']['calculation_performed']
    q=tools.call('prepare_query',dict(structure='bundle/structure.json',pseudos='pseudos',output='query',query={'kind':'scalar'},mesh=[3,3,3]))
    assert q['source_sha256']==assessment['source_sha256'] and not (tmp_path/'query/report.json').exists()
    # Inspection can be cited in durable research memory without becoming a
    # measured property or an automatic mechanism-validation result.
    h=tools.call('record_hypothesis',dict(statement='A smaller cell is sufficient',predictions=['same geometry'],alternatives=['intentional magnetic supercell']))
    a=tools.call('record_analysis',dict(parent_id=h,conclusion='Inspect complexity preference before DFT',evidence=result['evidence_path']))
    history=tools.call('retrieve_research_history',dict(query='complexity'))
    assert any(e['id']==a and e['payload']['report_sha256']==sha256(tmp_path/result['evidence_path']) for e in history)
    with pytest.raises(FileExistsError):tools.call('assess_structure',args)
    save(tmp_path/'assessment/assessment.json',assessment|{'findings':[]})
    with pytest.raises(ValueError,match='artifact changed'):verify_assessment(tmp_path/'assessment')


@pytest.mark.parametrize('policy',[POLICY|{'cutoff_angstrom':3.0},POLICY|{'max_stretch_deviation':'1/20'},POLICY|{'max_sites':0}])
def test_bad_policy_creates_no_assessment(experiment,tmp_path,policy):
    with pytest.raises(Exception):ResearchTools(tmp_path).call('assess_structure',dict(structure='bundle/structure.json',policy=policy,output='bad'))
    assert not (tmp_path/'bad').exists()


def test_export_tampering_and_workspace_escape_fail_before_output(experiment,tmp_path):
    tools=ResearchTools(tmp_path)
    args=dict(structure='bundle/structure.json',policy=POLICY,output='bad')
    with pytest.raises(ValueError,match='escapes'):tools.call('assess_structure',args|{'output':'../escape'})
    s=load(experiment/'structure.json');s['geometry']['fractional'][0][0]+=.001;save(experiment/'structure.json',s)
    with pytest.raises(Exception):tools.call('assess_structure',args)
    assert not (tmp_path/'bad').exists()


def test_policy_driven_session_receives_assessment_before_deciding(experiment,tmp_path):
    policy=tmp_path/'policy.py'
    policy.write_text('''import json,sys
c=json.load(sys.stdin)
if not c['history']:
 print(json.dumps({'tool':'assess_structure','args':{'structure':'bundle/structure.json','output':'assessment','policy':POLICY},'rationale':'Inspect geometry before choosing a calculation'}))
else:
 print(json.dumps({'finish':'Review the declared complexity preferences; no calculation requested'}))
'''.replace('POLICY',repr(POLICY)))
    report=run_session(tmp_path,'session',objective='Engineering assessment',policy_command=[sys.executable,str(policy)],max_steps=3,max_seconds=30)
    assert report['status']=='completed'
    assert report['native_submissions']==0 and len(report['steps'])==2
    assert verify_assessment(tmp_path/'assessment')['advisory_only']
    assert not (tmp_path/'assessment/report.json').exists()
