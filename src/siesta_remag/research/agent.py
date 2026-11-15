"""Bounded local research-policy loop over the existing tool protocol.

The trusted operator supplies a policy bridge (for Astra or another agent).
This module does not choose a model, hold credentials, or grant compute funds.
Policy commands are trusted local programs, not a filesystem/network sandbox.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..corpus import sha256
from ..discovery.common import save
from .tools import TOOLS,ResearchTools

LOCAL_TOOLS=frozenset({'inspect_calculation','inspect_electronic_structure','projected_dos',
    'compare_electronic_states','structural_operation','prepare_query','record_hypothesis',
    'propose_experiment','record_prediction','record_calculation_outcome','evaluate_hypothesis',
    'record_analysis','retrieve_research_history','calibrate_exploratory_response',
    'inspect_relaxation','prepare_relaxation','export_electronic_matrices','compare_electronic_matrices','export_orbital_moments'})


def _process(command,payload,deadline,folder,label,*,limit):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Research session time limit reached')
    env=dict(os.environ)
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
    with (folder/(label+'.stdout')).open('xb') as stdout,(folder/(label+'.stderr')).open('xb') as stderr:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
        try:
            proc.communicate(json.dumps(payload,allow_nan=False).encode()+b'\n',timeout=remaining)
        except BaseException:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=1)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    if proc.returncode:raise RuntimeError(label+' process failed; streams retained')
    output=folder/(label+'.stdout')
    if output.stat().st_size>limit:raise ValueError(label+' response exceeds size bound; retained on disk')
    return json.loads(output.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite policy/result JSON')))


def run_session(workspace,output,*,objective,policy_command,max_steps=24,max_seconds=300,allow_cpu_relaxation=False):
    """One policy decision then one tool call; retain intent before side effects.

    Every output directory is new. A timed-out tool may have committed notebook
    events: inspect evidence, never replay the action automatically. Native/GPU
    submission is deliberately absent from the tool catalog for this session.
    """
    tools=ResearchTools(workspace)
    output=tools.path(str(output))
    if (not isinstance(objective,str) or not 1<=len(objective)<=4000 or
        not isinstance(policy_command,list) or not policy_command or
        any(not isinstance(x,str) or not x for x in policy_command)):
        raise ValueError('Provide an objective and a trusted policy command argument list')
    if type(max_steps) is not int or not 1<=max_steps<=128 or type(max_seconds) not in (int,float) or not 1<=max_seconds<=3600:
        raise ValueError('Session bounds are 1..128 decisions and 1..3600 seconds')
    if type(allow_cpu_relaxation) is not bool:raise ValueError('CPU relaxation permission must be boolean')
    output.mkdir(parents=True,exist_ok=False)
    allowed={k:v for k,v in TOOLS.items() if k in LOCAL_TOOLS and (allow_cpu_relaxation or k!='prepare_relaxation')}
    request=dict(format='REMAG-AGENT-SESSION/1',objective=objective,policy_command=policy_command,
        max_steps=max_steps,max_seconds=max_seconds,allow_cpu_relaxation=allow_cpu_relaxation,
        native_execution_enabled=False,tools=allowed,started_at=datetime.now(timezone.utc).isoformat())
    save(output/'request.json',request)
    started=time.monotonic();deadline=started+max_seconds
    report=dict(format='REMAG-AGENT-SESSION-RESULT/1',status='running',request_sha256=sha256(output/'request.json'),
        steps=[],native_submissions=0,policy_conclusion=None,scientific_success_assessed=False)
    history=[]
    try:
        for step in range(max_steps):
            folder=output/f'{step:03d}';folder.mkdir()
            context=dict(objective=objective,tools=allowed,history=history,
                decisions_remaining=max_steps-step,seconds_remaining=max(0,deadline-time.monotonic()),
                response_contract='Exactly one {tool,args,rationale} or {finish: conclusion} JSON object. No native submission tool is available.')
            if len(json.dumps(context,allow_nan=False).encode())>1024*1024:
                report['status']='held_context_limit';break
            before=time.monotonic()
            decision=_process(policy_command,context,deadline,folder,'policy',limit=1024*1024)
            save(folder/'decision.json',decision)
            entry=dict(index=step,decision_sha256=sha256(folder/'decision.json'),policy_seconds=time.monotonic()-before)
            report['steps'].append(entry)
            if isinstance(decision,dict) and set(decision)=={'finish'}:
                if not isinstance(decision['finish'],str) or not decision['finish']:raise ValueError('A policy conclusion must be text')
                report.update(status='completed',policy_conclusion=decision['finish']);break
            if (not isinstance(decision,dict) or set(decision)!={'tool','args','rationale'} or
                not isinstance(decision['tool'],str) or decision['tool'] not in allowed or
                not isinstance(decision['args'],dict) or not isinstance(decision['rationale'],str) or not decision['rationale']):
                raise ValueError('Policy returned an invalid or unavailable action')
            action=dict(op=decision['tool'],args=decision['args'])
            # Durable intent exists before launching the tool process.
            save(folder/'action.json',action);save(output/'report.json',report)
            before=time.monotonic()
            response=_process([sys.executable,'-m','siesta_remag.research','tools','--workspace',str(tools.root)],
                action,deadline,folder,'tool',limit=8*1024*1024)
            save(folder/'response.json',response)
            entry.update(tool=decision['tool'],tool_seconds=time.monotonic()-before,
                response_sha256=sha256(folder/'response.json'),ok=response.get('ok') is True)
            history.append(dict(decision=decision,response=response))
            if response.get('ok') is not True:
                report['status']='held_tool_error';break
        else:report['status']='held_step_limit'
    except Exception as exc:
        report.update(status='held_error',error=dict(type=type(exc).__name__,message=str(exc)),
                      retry_authorized=False)
    except BaseException:
        report.update(status='held_interrupted',retry_authorized=False)
        raise
    finally:
        report.update(wall_seconds=time.monotonic()-started,finished_at=datetime.now(timezone.utc).isoformat(),
            scope='Policy/tool execution record; conclusions are not automatic scientific verification; no native or GPU submission')
        save(output/'report.json',report)
    return report
