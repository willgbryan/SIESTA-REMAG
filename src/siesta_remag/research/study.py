"""Registered paired research sessions; no automatic queue or compute grant.

The same trusted policy bridge and authoring interface receive scalar versus
electronic observations. Protocol isolation is not an OS sandbox or proof of
blinding. An empirical research advantage is never inferred from this runner.
"""
from datetime import datetime,timezone
from pathlib import Path
import math
import re
import json
import importlib.util

from ..corpus import sha256
from ..discovery.common import load,save,digest
from ..discovery.pipeline import lock
from .agent import run_session
from .execution import read_execution_config,ModalExecution
from .experiments import experiment_code
from .memory import Notebook
from .evaluation import prediction_scores,response_observation,validate_target
from .tools import calculation_observation

ARMS=('scalar','physics')


def _implementation():
    gpac=importlib.util.find_spec('gpac')
    return dict(remag=experiment_code(),gpac_python=None if gpac is None else
        {str(p.relative_to(Path(gpac.origin).parent)):sha256(p) for p in sorted(Path(gpac.origin).parent.rglob('*.py'))})


def _path(root,value):
    if not isinstance(value,str) or not value:raise ValueError('Expected a workspace path')
    p=(root/value).resolve()
    if not p.is_relative_to(root):raise ValueError('Study path escapes workspace')
    return p


def _inventory(root,excluded=()):
    files={};total=0
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Study input inventories cannot contain symlinks')
        if not p.is_file() or str(p.relative_to(root)) in excluded:continue
        total+=p.stat().st_size
        if len(files)>=10000 or total>8*1024**3:raise ValueError('Study initial inventory exceeds its finite bound')
        files[str(p.relative_to(root))]=sha256(p)
    return files


def register_study(workspace,definition,output):
    root=Path(workspace).resolve();out=_path(root,str(output));p=json.loads(json.dumps(definition,allow_nan=False))
    fields={'format','objective','policy_command','policy_artifacts','common_inputs','limits','arms','evaluation'}
    if not isinstance(p,dict) or set(p)!=fields or p['format']!='REMAG-PAIRED-STUDY/1':raise ValueError('Expected a complete paired-study definition')
    if not isinstance(p['objective'],str) or not 1<=len(p['objective'])<=4000:raise ValueError('Study needs an objective')
    if not isinstance(p['policy_command'],list) or not p['policy_command'] or any(not isinstance(x,str) or not x for x in p['policy_command']):
        raise ValueError('Provide one trusted policy command for both arms')
    for name in ('policy_artifacts','common_inputs'):
        values=p[name]
        if not isinstance(values,list) or not 1<=len(values)<=256 or any(not isinstance(x,str) for x in values) or len(set(values))!=len(values):
            raise ValueError('Declare distinct policy artifacts and common initial inputs')
    limits=p['limits']
    if (not isinstance(limits,dict) or set(limits)!={'max_steps','max_seconds','allow_cpu_relaxation'}
            or type(limits['max_steps']) is not int or not 1<=limits['max_steps']<=128
            or type(limits['max_seconds']) not in (int,float) or not math.isfinite(limits['max_seconds']) or not 1<=limits['max_seconds']<=3600
            or type(limits['allow_cpu_relaxation']) is not bool):raise ValueError('Invalid common session limits')
    evaluation=p['evaluation']
    if (not isinstance(evaluation,dict) or not {'heldout_sources','observables'}<=set(evaluation)
            or set(evaluation)-{'heldout_sources','observables','discovery'}):raise ValueError('Preregister heldout sources and observables')
    if 'discovery' in evaluation:
        from .discovery_scores import validate_policy
        evaluation['discovery']=validate_policy(evaluation['discovery'])
    sources=evaluation['heldout_sources'];observables=evaluation['observables']
    if (not isinstance(sources,list) or not 1<=len(sources)<=128 or any(not isinstance(s,str) or not re.fullmatch('[0-9a-f]{64}',s) for s in sources)
            or len(set(sources))!=len(sources)):raise ValueError('Declare distinct heldout source hashes')
    if not isinstance(observables,dict) or not 1<=len(observables)<=32:raise ValueError('Declare 1..32 evaluation observables')
    for q,row in observables.items():
        if (not isinstance(q,str) or not q or not isinstance(row,dict) or set(row) not in ({'unit','resolution'},{'unit','resolution','target'})
                or not isinstance(row['unit'],str) or not row['unit'] or type(row['resolution']) not in (int,float)
                or not math.isfinite(row['resolution']) or row['resolution']<0):raise ValueError('Evaluation needs observable units and finite resolutions')
        if 'target' in row:
            row['target'],unit=validate_target(row['target'])
            if row['unit']!=unit:raise ValueError('Registered response units differ from the target')
    if not isinstance(p['arms'],dict) or set(p['arms'])!=set(ARMS):raise ValueError('Require scalar and physics arms')
    records={};budgets=[];paths=[]
    for arm in ARMS:
        row=p['arms'][arm]
        if not isinstance(row,dict) or set(row)!={'workspace','execution_config'}:raise ValueError('Each arm needs a separate workspace and optional execution configuration')
        folder=_path(root,row['workspace']);paths.append(folder)
        if not folder.is_dir() or any((folder/x).exists() for x in ('session','research-memory.sqlite','.research-execution')):
            raise ValueError('Use arm workspaces without previous sessions or notebook/execution history')
        excluded=[];execution=None;budget=None
        if row['execution_config'] is not None:
            execution=read_execution_config(folder,row['execution_config']);c=execution['config']
            capacity=min(c['max_reserved_cents'],c['max_queries']*execution['reserved_cents'])
            if execution['ledger'].remaining_cents(c['budget_stage'])<capacity:raise ValueError('Existing stage cannot fund its full declared arm ceiling')
            if limits['max_seconds']<execution['required_session_seconds']:raise ValueError('Session deadline cannot contain its configured execution ceiling')
            ledger=execution['ledger'].path;excluded=[str(ledger.relative_to(folder)),str(ledger.with_suffix(ledger.suffix+'.lock').relative_to(folder))]
            budget=dict(max_queries=c['max_queries'],max_reserved_cents=c['max_reserved_cents'],per_query_cents=execution['reserved_cents'],
                timeout_seconds=c['app_timeout_seconds'],hourly_reservation_usd=c['hourly_reservation_usd'],runtime=execution['runtime'],image=c['image'])
        files=_inventory(folder,excluded)
        for name in p['common_inputs']:
            path=_path(folder,name)
            if str(path.relative_to(folder)) not in files:raise ValueError('Common inputs must be immutable files in each arm')
        for name in files:
            path=folder/name
            if arm == 'scalar' and path.suffix == '.json' and path.stat().st_size > 16*1024**2:
                raise ValueError('Scalar initial JSON exceeds the staging audit bound')
            if path.suffix=='.json' and path.stat().st_size<=16*1024**2:
                data=load(path)
                if isinstance(data,dict) and data.get('format') in ('REMAG-EXPERIMENT-RESULT/1','REMAG-DISCOVERY-RESULT/1') and data.get('source_sha256') in sources:
                    raise ValueError('Do not seed an arm with heldout calculation results')
                if arm == 'scalar':
                    from .staging import check_scalar_input
                    check_scalar_input(data, sources)
                if isinstance(data,dict) and data.get('format') == 'REMAG-SCALAR-DATA/1':
                    from .staging import validate_scalar_data
                    validate_scalar_data(data)
                    if any(row['source_sha256'] in sources for row in data['cases'].values()):
                        raise ValueError('Do not seed an arm with heldout calculation results')
        records[arm]=dict(initial_files=files,mutable_budget_files=excluded,
            execution_config_sha256=execution['sha256'] if execution else None,
            initial_ledger=execution['ledger'].snapshot() if execution else None)
        budgets.append(budget)
    if paths[0].is_relative_to(paths[1]) or paths[1].is_relative_to(paths[0]):raise ValueError('Arm workspaces must be disjoint')
    if any(out.is_relative_to(x) or x.is_relative_to(out) for x in paths):raise ValueError('Study records must be separate from both arm workspaces')
    if budgets[0]!=budgets[1]:raise ValueError('Native execution ceilings, runtime and image must match between arms')
    for name in p['common_inputs']:
        if sha256(_path(paths[0],name))!=sha256(_path(paths[1],name)):raise ValueError('Common initial inputs differ')
    artifacts={name:sha256(_path(root,name)) for name in p['policy_artifacts']}
    if any(_path(root,name).is_relative_to(arm) for name in artifacts for arm in paths):raise ValueError('Keep the common immutable policy outside mutable arm workspaces')
    record=dict(format='REMAG-PAIRED-STUDY-REGISTRATION/1',definition=p,arms=records,policy_artifacts=artifacts,
        tool_implementation=_implementation(),
        native_budget_per_arm=budgets[0],registered_at=datetime.now(timezone.utc).isoformat(),
        budget_scope='Matched reservation/query ceilings; actual GPU-hours and invoices are separate measurements' if budgets[0] else 'Offline study; matched session bounds, no native execution grant',
        blinding_verified=False,scientific_advantage_demonstrated=False)
    record['sha256']=digest(record)
    out.mkdir(parents=True,exist_ok=False);save(out/'study.json',record)
    return record


def _registered(root,study):
    folder=_path(root,str(study));record=load(folder/'study.json')
    if record.get('format')!='REMAG-PAIRED-STUDY-REGISTRATION/1' or record.get('sha256')!=digest({k:v for k,v in record.items() if k!='sha256'}):
        raise ValueError('Study registration changed')
    if record['tool_implementation']!=_implementation():raise ValueError('Registered tool implementation changed')
    for path,expected in record['policy_artifacts'].items():
        if sha256(_path(root,path))!=expected:raise ValueError('Registered policy artifact changed')
    return folder,record


def _inputs_unchanged(folder,record,*,initial=False):
    files=_inventory(folder,record['mutable_budget_files'])
    if any(files.get(k)!=v for k,v in record['initial_files'].items()) or (initial and files!=record['initial_files']):
        raise ValueError('Registered initial knowledge changed')


def _released(root,definition,arm):
    row=definition['arms'][arm]
    if row['execution_config'] is None:return
    folder=_path(root,row['workspace']);session=folder/'session/report.json'
    if not session.exists():raise ValueError('Prior native-enabled arm lacks terminal session evidence')
    r=load(session)
    if r['native_execution_attempts']==0:return
    owner=load(folder/'.research-execution/owner.json')
    if owner['status']!='measured' or owner.get('allocation_stopped') is not True or not owner.get('app_id'):
        raise ValueError('Prior native execution needs explicit review; no following arm')
    info=ModalExecution().info(owner['app_id'])
    if info.get('app_id')!=owner['app_id'] or info.get('lifecycle',{}).get('state','').lower()!='stopped':
        raise ValueError('Prior owned app is not authoritatively STOPPED')


def _events(workspace):
    path=workspace/'research-memory.sqlite'
    if not path.exists():return {}
    notebook=Notebook(path)
    try:
        ids=notebook.db.execute('SELECT id FROM events ORDER BY rowid').fetchall()
        if len(ids)>10000:raise ValueError('Study notebook exceeds its finite event bound')
        return {r[0]:notebook.get(r[0]) for r in ids}
    finally:notebook.close()


def run_study_arm(workspace,study,arm):
    root=Path(workspace).resolve();folder,record=_registered(root,study);p=record['definition']
    if arm not in ARMS:raise ValueError('Select scalar or physics arm')
    with lock(folder/'study.lock'):
        receipt=folder/(arm+'.json');owner=folder/'owner.json'
        if receipt.exists():raise ValueError('Study arm already attempted; no automatic retry')
        if owner.exists() and load(owner)['status']!='terminal':raise ValueError('Study has unresolved process ownership')
        for other in ARMS:
            if (folder/(other+'.json')).exists():_released(root,p,other)
        row=p['arms'][arm];where=_path(root,row['workspace']);_inputs_unchanged(where,record['arms'][arm],initial=True)
        if row['execution_config'] is not None:
            execution=read_execution_config(where,row['execution_config'])
            if execution['sha256']!=record['arms'][arm]['execution_config_sha256'] or execution['ledger'].snapshot()!=record['arms'][arm]['initial_ledger']:
                raise ValueError('Registered execution grant or its initial ledger history changed')
        state=dict(format='REMAG-STUDY-ARM/1',arm=arm,registration_sha256=record['sha256'],status='running',
            started_at=datetime.now(timezone.utc).isoformat(),session='session',retry_authorized=False)
        save(receipt,state);save(owner,dict(arm=arm,status='running'))
        try:
            result=run_session(where,'session',objective=p['objective'],policy_command=p['policy_command'],
                **p['limits'],execution_config=row['execution_config'],information_profile=arm,withheld_sources=p['evaluation']['heldout_sources'])
            state.update(status=result['status'],session_report_sha256=sha256(where/'session/report.json'),
                notebook_snapshot={uid:digest(event) for uid,event in _events(where).items()})
            _inputs_unchanged(where,record['arms'][arm]);_registered(root,study)
        except Exception as exc:
            state.update(status='held_study_error',error=dict(type=type(exc).__name__,message=str(exc)))
        except BaseException:
            state['status']='held_interrupted';raise
        finally:
            state['finished_at']=datetime.now(timezone.utc).isoformat();save(receipt,state)
            save(owner,dict(arm=arm,status='terminal'))
        return state


def _audit_session(root,folder,record,arm):
    receipt=load(folder/(arm+'.json'));p=record['definition'];where=_path(root,p['arms'][arm]['workspace'])
    if receipt['registration_sha256']!=record['sha256']:raise ValueError('Arm registration binding differs')
    if 'session_report_sha256' not in receipt:
        return dict(status=receipt['status'],session_verified=False,completed_experiments=None,failed_experiments=None,native_execution_attempts=None,
                    reserved_cents=None,actual_gpu_hours=None,actual_billed_usd=None)
    session=where/'session';r=load(session/'report.json');request=load(session/'request.json')
    if sha256(session/'report.json')!=receipt['session_report_sha256'] or sha256(session/'request.json')!=r['request_sha256']:
        raise ValueError('Session artifact binding differs')
    expected=p['limits'] | dict(objective=p['objective'],policy_command=p['policy_command'],information_profile=arm,
        withheld_sources=sorted(p['evaluation']['heldout_sources']),execution_config_sha256=record['arms'][arm]['execution_config_sha256'])
    if any(request.get(k)!=v for k,v in expected.items()):raise ValueError('Session limits, policy or information profile differs')
    from .model_policy import audit_session
    model=audit_session(session)
    if model!=r.get('model_policy'):raise ValueError('Session model policy audit differs')
    native=[];native_windows={};candidates=[];failed_authoring=0;tool_seconds=policy_seconds=0.
    for step in r['steps']:
        path=session/f"{step['index']:03d}";decision=load(path/'decision.json')
        if sha256(path/'decision.json')!=step['decision_sha256']:raise ValueError('Session decision changed')
        policy_seconds+=step['policy_seconds']
        if 'response_sha256' not in step:continue
        if sha256(path/'response.json')!=step['response_sha256']:raise ValueError('Session response changed')
        if load(path/'action.json')!=dict(op=decision['tool'],args=decision['args']):raise ValueError('Session action differs from policy decision')
        if decision['tool'] not in request['tools']:raise ValueError('Session used an unavailable tool')
        response=load(path/'response.json');tool_seconds+=step['tool_seconds']
        if decision['tool']=='run_prepared_query':
            measured=response.get('result') if response.get('ok') else None;native.append(measured)
            review=(measured or {}).get('review',{})
            if review.get('report_sha256'):
                from .discovery_scores import report_interval
                file=_path(where,review['report'])
                if sha256(file)!=review['report_sha256']:raise ValueError('Native report changed after session review')
                native_windows[review['report_sha256']]=report_interval(load(file))
        if decision['tool']=='author_structure':
            if not response.get('ok'):failed_authoring+=1
            else:
                from .authoring import verify_candidate
                authored=response['result'];candidate_path=_path(where,authored['candidate_path'])
                candidate=verify_candidate(candidate_path)
                if (candidate!=authored['candidate'] or _path(where,authored['structure_path'])!=candidate_path.parent/'structure.json'):
                    raise ValueError('Session candidate differs from its recorded authoring result')
                candidates.append(dict(step=step['index'],source_sha256=candidate['source_sha256'],
                    candidate_path=authored['candidate_path'],candidate_sha256=sha256(candidate_path),structure_path=authored['structure_path']))
    _inputs_unchanged(where,record['arms'][arm])
    events=_events(where);snapshot=receipt.get('notebook_snapshot',{})
    if any(uid not in events or digest(events[uid])!=h for uid,h in snapshot.items()):raise ValueError('Session notebook evidence changed')
    if any(e['kind'] in ('experiment','prediction') and uid not in snapshot for uid,e in events.items()):
        raise ValueError('Experiment or prediction was added after the arm finished')
    graded={}
    if (folder/'grading.json').exists():
        grade=load(folder/'grading.json')
        if (grade['status']!='completed' or grade.get('sha256')!=digest({k:v for k,v in grade.items() if k!='sha256'})
                or len(grade['records'])!=len(grade['recorded_outcome_ids'])):raise ValueError('Grading is incomplete or changed')
        graded={uid:row for uid,row in zip(grade['recorded_outcome_ids'],grade['records']) if row['arm']==arm}
    for uid,event in events.items():
        if event['kind']=='outcome' and uid not in snapshot:
            row=graded.get(uid)
            if row is None or event['parent']!=row['experiment_id'] or event['payload']!=row['observation']:
                raise ValueError('Outcome was added outside the registered grading step')
    complete=[x for x in native if x and x.get('status')=='measured'];failed=[x for x in native if x and x.get('status')!='measured']
    costs=[x.get('reservation_recorded_cents') if x else None for x in native]
    times=[x.get('review',{}).get('native_wall_seconds') if x else None for x in native]
    return dict(status=r['status'],session_verified=True,session_report_sha256=sha256(session/'report.json'),
        finished_at=receipt['finished_at'],authored_candidates=candidates,failed_authoring_attempts=failed_authoring,
        native_app_ids=sorted({x['app_id'] for x in native if x and x.get('app_id')}),
        native_report_sha256s=sorted({x['review']['report_sha256'] for x in native if x and x.get('review',{}).get('report_sha256')}),
        native_report_intervals=native_windows,
        native_execution_attempts=r['native_execution_attempts'],native_submissions=r['native_submissions'],
        completed_experiments=len(complete),failed_experiments=len(failed),unresolved_attempts=r['native_execution_attempts']-len(complete)-len(failed),
        reserved_cents=sum(costs) if len(costs)==r['native_execution_attempts'] and all(x is not None for x in costs) else None,
        native_pipeline_seconds=sum(times) if len(times)==r['native_execution_attempts'] and all(x is not None for x in times) else None,
        policy_wall_seconds=policy_seconds,tool_wall_seconds=tool_seconds,session_wall_seconds=r['wall_seconds'],
        model_policy=model,
        actual_gpu_hours=None,actual_billed_usd=None,scope='Counts cover session execution actions, not cached inspections; native pipeline time excludes allocation overhead and is not GPU-hours')


def grade_study(workspace,study,outcomes):
    """Apply one shared held-out evidence manifest after both policies finish.

    All values are computed and validated before notebook writes. Durable grade
    intent prevents replay after an interruption; partial writes are retained.
    """
    root=Path(workspace).resolve();folder,record=_registered(root,study);p=record['definition']
    if not isinstance(outcomes,dict) or set(outcomes)!={'format','cases'} or outcomes['format']!='REMAG-STUDY-OUTCOMES/1' or not isinstance(outcomes['cases'],dict):
        raise ValueError('Expected a held-out outcome manifest')
    declared=p['evaluation']['observables'];sources=p['evaluation']['heldout_sources']
    if set(outcomes['cases'])-set(sources):raise ValueError('Outcome source is outside registered holdout')
    for row in outcomes['cases'].values():
        if not isinstance(row,dict) or set(row)-set(declared):raise ValueError('Outcome observable is outside registered evaluation')
    with lock(folder/'study.lock'):
        path=folder/'grading.json'
        if path.exists():raise ValueError('Study grading already attempted; inspect retained evidence, do not replay')
        if not all((folder/(arm+'.json')).exists() for arm in ARMS) or load(folder/'owner.json')['status']!='terminal':
            raise ValueError('Both arms must finish before exposing heldout outcomes')
        pending=[]
        for arm in ARMS:
            _released(root,p,arm);_audit_session(root,folder,record,arm)
            where=_path(root,p['arms'][arm]['workspace']);events=_events(where)
            sources_seen=set()
            for e in events.values():
                if e['kind']=='experiment' and e['payload']['structure_sha256'] in sources:
                    source=e['payload']['structure_sha256']
                    if source in sources_seen:raise ValueError('Multiple heldout experiments for one source')
                    sources_seen.add(source)
            for uid,event in events.items():
                if event['kind']!='prediction':continue
                experiment=events[event['parent']];source=experiment['payload']['structure_sha256'];pr=event['payload'];q=pr['observable']
                if source not in sources:continue
                if q not in declared or pr['unit']!=declared[q]['unit'] or pr.get('target')!=declared[q].get('target'):
                    raise ValueError('Prediction is outside registered observable/units/definition')
                if any(e['kind']=='outcome' and e['parent']==experiment['id'] and e['payload']['observable']==q for e in events.values()):
                    raise ValueError('A heldout outcome was already known before independent grading')
                observation=outcomes['cases'].get(source,{}).get(q)
                if observation is None:continue
                if not isinstance(observation,dict):raise ValueError('Expected a heldout observation definition')
                if 'target' in pr:
                    if set(observation)!={'before','after'}:raise ValueError('Response grading needs before and after endpoints')
                    endpoints=[{k:_path(root,v) for k,v in observation[key].items()} for key in ('before','after')]
                    result=response_observation(pr['target'],*endpoints,source_sha256=source)
                    result.update(observable=q,report_sha256=result['evidence']['after']['report_sha256'])
                else:
                    if set(observation)-{'report','collection','stage','observable'} or not {'report','stage','observable'}<=set(observation):
                        raise ValueError('Scalar grading needs a saved calculation observable')
                    result=calculation_observation(_path(root,observation['report']),source,observation['stage'],observation['observable'],
                        collection=_path(root,observation['collection']) if 'collection' in observation else None)
                    if result['observable']!=q:raise ValueError('Graded quantity differs from the prediction')
                if result['unit']!=pr['unit']:raise ValueError('Graded unit differs from the prediction')
                from .discovery_scores import report_interval
                paths=[e['report'] for e in endpoints] if 'target' in pr else [_path(root,observation['report'])]
                intervals={sha256(path):report_interval(load(path)) for path in paths}
                pending.append(dict(arm=arm,experiment_id=experiment['id'],prediction_id=uid,observation=result,report_intervals=intervals))
        receipt=dict(format='REMAG-STUDY-GRADING/1',registration_sha256=record['sha256'],outcomes_sha256=digest(outcomes),
            status='recording',records=pending,recorded_outcome_ids=[],blinding_verified=False)
        save(path,receipt)
        try:
            for row in pending:
                n=Notebook(_path(root,p['arms'][row['arm']]['workspace'])/'research-memory.sqlite')
                try:uid=n.record_outcome(row['experiment_id'],**row['observation'])
                finally:n.close()
                receipt['recorded_outcome_ids'].append(uid);save(path,receipt)
            receipt['status']='completed'
            receipt['sha256']=digest(receipt)
        except BaseException:
            receipt['status']='held_partial_grading';raise
        finally:save(path,receipt)
        return receipt


def evaluate_study(workspace,study,output,*,discovery_evidence=None,usage=None):
    root=Path(workspace).resolve();folder,record=_registered(root,study);p=record['definition'];out=_path(root,str(output))
    if out.exists():raise ValueError('Study score output must be new')
    if any(out.is_relative_to(x) for x in [folder]+[_path(root,p['arms'][arm]['workspace']) for arm in ARMS]):
        raise ValueError('Keep score outputs outside study ownership and arm workspaces')
    with lock(folder/'study.lock'):
        if not all((folder/(arm+'.json')).exists() for arm in ARMS) or load(folder/'owner.json')['status']!='terminal':
            raise ValueError('Both arms must be terminal before evaluation')
        arms={};cohorts={};declared=p['evaluation']['observables'];sources=p['evaluation']['heldout_sources']
        for arm in ARMS:
            _released(root,p,arm);arms[arm]=_audit_session(root,folder,record,arm)
            where=_path(root,p['arms'][arm]['workspace']);ids=[];found={};predicted=set()
            if (where/'research-memory.sqlite').exists():
                n=Notebook(where/'research-memory.sqlite')
                try:
                    for row in n.db.execute("SELECT id FROM events WHERE kind='experiment' ORDER BY rowid"):
                        event=n.get(row[0]);source=event['payload']['structure_sha256']
                        if source not in sources:continue
                        if source in found:raise ValueError('Multiple heldout experiments for one source; do not select the best retrospectively')
                        found[source]=event['id'];ids.append(event['id'])
                        for child in n.db.execute("SELECT id FROM events WHERE parent=? AND kind='prediction'",(event['id'],)):
                            pr=n.get(child[0])['payload'];q=pr['observable']
                            if q not in declared or pr['unit']!=declared[q]['unit'] or pr.get('target')!=declared[q].get('target'):
                                raise ValueError('Heldout prediction is outside registered observables/units/definition')
                            predicted.add(q)
                    score=prediction_scores(n,ids,{q:declared[q]['resolution'] for q in predicted}) if ids else None
                finally:n.close()
            else:score=None
            cohorts[arm]=dict(scores=score,missing_sources=[s for s in sources if s not in found],
                required_predictions=len(sources)*len(declared),recorded_predictions=len(score['rows']) if score else 0)
        paired={}
        for q in declared:
            rows={arm:{r['source_sha256']:r for r in (cohorts[arm]['scores'] or {}).get('rows',[])
                       if r['observable']==q and r['evaluation']['status']=='evaluated'} for arm in ARMS}
            shared=sorted(set(rows['scalar']) & set(rows['physics']))
            mean={arm:sum(rows[arm][s]['evaluation']['absolute_error'] for s in shared)/len(shared) if shared else None for arm in ARMS}
            paired[q]=dict(common_measured_sources=shared,paired_count=len(shared),mae=mean,
                physics_minus_scalar_mae=mean['physics']-mean['scalar'] if shared else None,
                scope='Paired subset only; missing predictions and failed outcomes remain in the full cohort inventories')
        result=dict(format='REMAG-PAIRED-STUDY-SCORES/1',registration_sha256=record['sha256'],arms=arms,heldout=cohorts,paired_prediction_errors=paired,
            registered_native_budget_per_arm=record['native_budget_per_arm'],matched_registered_limits=True,
            empirical_equal_compute_verified=False,blinding_verified=False,scientific_advantage_demonstrated=False,
            independent_candidate_validation=None,validated_candidates_per_gpu_hour=None,
            scope='Protocol comparison and recorded prediction scores; shared declared limits are not measured equal GPU consumption or a proof of independent/blinded science. Discovery validation and transfer remain separate evidence.')
        required_reports={a:set(arms[a].get('native_report_sha256s',[])) for a in ARMS}
        report_windows={h:v for a in ARMS for h,v in arms[a].get('native_report_intervals',{}).items()}
        if discovery_evidence is not None or 'discovery' in p['evaluation']:
            from .discovery_scores import score_candidates
            manifest=discovery_evidence if discovery_evidence is not None else dict(format='REMAG-STUDY-CANDIDATE-EVIDENCE/1',arms={a:{} for a in ARMS})
            candidate_scores=score_candidates(root,record,arms,manifest)
            result['independent_candidate_validation']=candidate_scores
            for a in ARMS:required_reports[a].update(candidate_scores['required_usage_reports'][a])
            report_windows.update(candidate_scores['required_usage_intervals'])
        if usage is not None:
            from .discovery_scores import score_usage
            # Acquisition and grading compute belong in an efficiency denominator
            # even when the corresponding files were merely inspected by an arm.
            for a in ARMS:
                where=_path(root,p['arms'][a]['workspace'])
                for name,h in record['arms'][a]['initial_files'].items():
                    file=where/name
                    if file.suffix=='.json' and file.stat().st_size<=16*1024**2:
                        initial=load(file)
                        if isinstance(initial,dict) and initial.get('format') in ('REMAG-EXPERIMENT-RESULT/1','REMAG-DISCOVERY-RESULT/1'):
                            required_reports[a].add(h)
                            from .discovery_scores import report_interval
                            report_windows[h]=report_interval(initial)
            if (folder/'grading.json').exists():
                def report_hashes(value):
                    if isinstance(value,dict):
                        for key,v in value.items():
                            if key=='report_sha256' and isinstance(v,str):yield v
                            else:yield from report_hashes(v)
                    elif isinstance(value,list):
                        for v in value:yield from report_hashes(v)
                for row in load(folder/'grading.json')['records']:
                    required_reports[row['arm']].update(report_hashes(row['observation']))
                    report_windows.update(row.get('report_intervals',{}))
            measurements=score_usage(root,arms,usage,required_reports=required_reports,report_windows=report_windows)
            result['measured_allocation_usage']=measurements
            for a in ARMS:arms[a]['actual_gpu_hours']=measurements['arms'][a]['actual_gpu_hours']
            if result['independent_candidate_validation'] is not None:
                result['validated_candidates_per_gpu_hour']={a:(
                    result['independent_candidate_validation']['arms'][a]['validated_exact_sources']/arms[a]['actual_gpu_hours']
                    if arms[a]['actual_gpu_hours'] is not None and arms[a]['actual_gpu_hours']>0 else None) for a in ARMS}
                result['candidate_rate_scope']='Exact-source candidates passing the registered finite controls per imported attributed allocation GPU-hour; not distinct crystals or physical stability validation'
            h=arms['physics']['actual_gpu_hours']
            result['paired_prediction_improvement_per_gpu_hour']={q:(-row['physics_minus_scalar_mae']/h
                if h is not None and h>0 and row['physics_minus_scalar_mae'] is not None else None) for q,row in paired.items()}
            result['prediction_rate_scope']='Scalar MAE minus physics MAE on their shared measured cohort, divided by attributed physics-arm GPU-hours; units are observable units per GPU-hour. Missing/failed predictions remain in the complete scorecard; this rate alone does not demonstrate scientific advantage.'
        save(out,result)
        return result
