"""Decision-only policies: the model receives data, never an ambient agent harness.

The operator owns this broker and its files. Only fixed ToolHost operations are
executed locally; model output cannot import Python, run a shell, enumerate
agents, select a transport, or enable compute. No desktop subagent is used.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re

from ..corpus import sha256
from ..discovery.common import load, save
from . import model_policy
from .staging import check_scalar_input, validate_scalar_data
from .toolhost import ToolHost  # Canonical import, owned and tested by the broker.

TOOLS = frozenset({'inspect_scalar_data', 'compare_electronic_states', 'author_structure',
    'record_hypothesis', 'propose_experiment', 'record_response_prediction', 'record_analysis'})
SEED = ('packet.json', 'parent.json')
ELECTRONIC = ('electronic/soc-z/electronic.json', 'electronic/soc-z/spectrum.npz',
              'electronic/soc-x/electronic.json', 'electronic/soc-x/spectrum.npz')


def preflight(workspace, *, information_profile, withheld_sources=()):
    """No model/service/solver. Validate exact staged surface and fixed import."""
    root = Path(workspace).resolve()
    host = ToolHost(root, information_profile=information_profile, withheld_sources=withheld_sources)
    expected = set(SEED) | (set(ELECTRONIC) if information_profile == 'physics' else set())
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if actual != expected or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Policy input directory must contain only the exact clean seed files; no symlinks')
    packet = validate_scalar_data(load(root/'packet.json'))
    check_scalar_input(packet, withheld_sources)
    parent = host.context.call('inspect_structure', {'structure': 'parent.json'})
    if information_profile == 'physics':
        from .inspection import load_spectrum
        for stage in ('soc-z', 'soc-x'):
            meta, _ = load_spectrum(root/f'electronic/{stage}/electronic.json')
            if meta['artifact'] != 'spectrum.npz':
                raise ValueError('Electronic input must use its local registered array')
    return dict(format='REMAG-ISOLATED-POLICY-PREFLIGHT/1', status='passed',
        information_profile=information_profile, withheld_sources=sorted(withheld_sources),
        canonical_import='siesta_remag.research.toolhost.ToolHost',
        files={p:sha256(root/p) for p in sorted(expected)},
        parent=parent, packet=packet, model_requests=0, GPU_launches=0,
        ambient_harness_enabled=False, native_execution_enabled=False)


class PolicyContext:
    """One durable arm; histories are built from its own receipts, never supplied by a policy."""
    @classmethod
    def create(cls, workspace, *, objective, information_profile, withheld_sources=()):
        root = Path(workspace).resolve()
        if len(set(withheld_sources)) != 2:
            raise ValueError('Register exactly two withheld sources')
        pf = preflight(root, information_profile=information_profile, withheld_sources=withheld_sources)
        if not isinstance(objective, str) or not 1 <= len(objective) <= 4000:
            raise ValueError('An explicit finite objective is required')
        private = root/'.policy'; private.mkdir(exist_ok=False)
        registered = dict(format='REMAG-ISOLATED-POLICY/1', objective=objective,
            information_profile=information_profile, withheld_sources=sorted(withheld_sources),
            files=pf['files'], initial_data={'parent':pf['parent'], 'scalar_data':pf['packet']},
            created_at=datetime.now(timezone.utc).isoformat(), max_calls=12,
            pre_outcome_calls=10, post_score_calls=2,
            boundary='Decision-only model connection; fixed local broker; no ambient agent/shell tools')
        save(private/'registration.json', registered)
        save(private/'state.json', dict(status='ready', calls=0, registration_sha256=sha256(private/'registration.json')))
        return cls(root)

    def __init__(self, workspace):
        self.root = Path(workspace).resolve(); self.private = self.root/'.policy'
        self.registration = load(self.private/'registration.json')
        self.host = ToolHost(self.root, information_profile=self.registration['information_profile'],
                            withheld_sources=self.registration['withheld_sources'])
        self.catalog = {k:v for k,v in self.host.context.catalog().items() if k in TOOLS}

    def _check(self):
        state = load(self.private/'state.json')
        if state['registration_sha256'] != sha256(self.private/'registration.json'):
            raise ValueError('Policy registration changed')
        for path, expected in self.registration['files'].items():
            if sha256(self.host.context.path(path)) != expected:
                raise ValueError('Policy seed changed')
        if state['status'] not in ('ready', 'revising'):
            raise ValueError('Policy phase is closed, unscored or held; no automatic retry')
        return state

    def _context(self, state, seconds):
        history = []
        for i in range(state['calls']):
            p=self.private/f'{i:03d}'
            binding=load(p/'binding.json')
            for name,h in binding.items():
                if sha256(p/name) != h:raise ValueError('Policy history changed')
            history.append(dict(decision=load(p/'decision.json'), response=load(p/'response.json')))
        context = {k:self.registration[k] for k in ('objective','information_profile','withheld_sources','initial_data')}
        context.update(tools=self.catalog, history=history, decisions_remaining=12-state['calls'],
                       seconds_remaining=seconds, phase=state['status'])
        if state['status']=='revising':
            if sha256(self.root/'scores.json') != state['scores_sha256']:raise ValueError('Released scores changed')
            context['released_scores']=load(self.root/'scores.json')
        return context

    def _authorize(self, decision, state):
        if (not isinstance(decision,dict) or set(decision)!={'tool','args','rationale'}
                or decision['tool'] not in self.catalog or not isinstance(decision['args'],dict)
                or not isinstance(decision['rationale'],str) or not decision['rationale'].strip()):
            raise ValueError('Policy may select only the fixed research catalog')
        name=decision['tool'];args=decision['args']
        if state['status']=='revising':
            if name!='record_analysis' or args.get('evidence')!='scores.json':
                raise ValueError('Scored phase permits only evidence-bound policy revisions')
        elif name=='record_analysis':raise ValueError('Scoring has not been released')
        if name=='inspect_scalar_data' and args.get('packet')!='packet.json':
            raise ValueError('Only the registered scalar packet is readable')
        if name=='author_structure':
            if args.get('structure')!='parent.json' or not re.fullmatch(r'candidates/[a-zA-Z0-9_-]{1,64}',str(args.get('output',''))):
                raise ValueError('Author only from the registered parent into a new candidate')
        if name=='compare_electronic_states':
            if (args.get('before')!='electronic/soc-z/electronic.json'
                    or args.get('after')!='electronic/soc-x/electronic.json'
                    or set(args)-{'before','after','selection','points','sigma_ev'}):
                raise ValueError('Only the registered electronic pair is readable')
        if name=='record_response_prediction' and (not isinstance(args.get('target'),dict) or args['target'].get('kind')!='anisotropy_response'):
            raise ValueError('This acceptance records E/F anisotropy responses only')

    def step(self, config, *, api_key=None, transport=None, seconds=600, backend=None):
        """A single model request and tool action. Any failure holds the whole arm."""
        with (self.private/'lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=self._check()
            backend_id=dict(kind=backend.kind if backend is not None else 'responses',
                            config_sha256=sha256(Path(config)))
            if state.get('policy_backend',backend_id)!=backend_id:
                raise ValueError('Policy backend changed; no context migration')
            state['policy_backend']=backend_id
            now=datetime.now(timezone.utc)
            if 'phase_started_at' not in state:state['phase_started_at']=now.isoformat()
            remaining=600-(now-datetime.fromisoformat(state['phase_started_at'])).total_seconds()
            if remaining<=2:
                state.update(status='held_deadline',retry_authorized=False);save(self.private/'state.json',state)
                raise ValueError('Policy phase deadline reached; no extension')
            seconds=min(seconds,remaining)
            folder=self.private/f"{state['calls']:03d}"
            folder.mkdir(exist_ok=False)
            state['status_before_call']=state['status'];state['status']='running'
            save(self.private/'state.json',state)  # Crash cannot silently reexecute.
            try:
                active=dict(state,status=state['status_before_call'])
                context=self._context(active,seconds)
                save(folder/'context.json',context)
                decision=(backend.decide(context,config,folder) if backend is not None else
                          model_policy.decide(context,config,folder,api_key=api_key,transport=transport))
                save(folder/'decision.json',decision); self._authorize(decision,active)
                response=self.host.call(decision,f"calls/{state['calls']:03d}")
                # Do not feed implementation paths/receipt metadata or exception text to a policy.
                if response['ok'] is not True:raise ValueError('Research tool failed; inspect private receipt')
                visible=dict(ok=True,result=response['result'])
                save(folder/'response.json',visible)
                save(folder/'binding.json',{n:sha256(folder/n) for n in ('decision.json','response.json')})
                state['calls']+=1
                state['status']='awaiting_scores' if state['calls']==10 else 'completed' if state['calls']==12 else active['status']
                if state['calls']==10:
                    self._verify_predictions()
                    with (self.private/'sealed-notebook.sqlite').open('xb') as frozen:
                        frozen.write((self.root/'research-memory.sqlite').read_bytes())
                    save(self.private/'seal.json',dict(status='sealed',calls=10,
                        notebook_sha256=sha256(self.root/'research-memory.sqlite'),
                        model_audit=backend.audit(self.private) if backend is not None else model_policy.audit_session(self.private),
                        bindings={str(p.relative_to(self.private)):sha256(p) for p in sorted(self.private.glob('[0-9][0-9][0-9]/binding.json'))}))
                return visible
            except BaseException as exc:
                state.update(status='held_error',error_type=type(exc).__name__,retry_authorized=False)
                raise
            finally:save(self.private/'state.json',state)

    def _verify_predictions(self):
        from .memory import Notebook
        book=Notebook(self.root/'research-memory.sqlite')
        try:
            events=[book.get(row[0]) for row in book.db.execute('SELECT id FROM events ORDER BY rowid')]
            if sorted(e['kind'] for e in events)!=sorted(['hypothesis']+['experiment']*2+['prediction']*4):
                raise ValueError('Seal requires one hypothesis, two experiments and four predictions')
            experiments=[e for e in events if e['kind']=='experiment']
            if {e['payload']['structure_sha256'] for e in experiments}!=set(self.registration['withheld_sources']):
                raise ValueError('Seal must cover exactly the two registered heldout sources')
            baseline=self.registration['initial_data']['parent']['structure']['source']['sha256']
            for experiment in experiments:
                predictions=[e for e in events if e['parent']==experiment['id']]
                if {e['payload']['observable'] for e in predictions}!={'E_response','F_response'}:
                    raise ValueError('Both E and F responses are required for each intervention')
                for e in predictions:
                    p=e['payload'];metric='total_energy_ev' if p['observable']=='E_response' else 'free_energy_ev'
                    if p.get('target')!={'kind':'anisotropy_response','baseline_source_sha256':baseline,'metric':metric} or p['unit']!='eV/cell':
                        raise ValueError('Prediction target differs from registered E/F response')
        finally:book.close()

    def release_scores(self, scores):
        """Operator-only: the model cannot invoke this method or supply its evidence."""
        with (self.private/'lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=load(self.private/'state.json')
            if state['status']!='awaiting_scores' or state['calls']!=10:
                raise ValueError('Both policy phases are finite; scores require an intact completed preregistration')
            seal=load(self.private/'seal.json')
            if any(sha256(p)!=seal['notebook_sha256'] for p in
                   (self.root/'research-memory.sqlite',self.private/'sealed-notebook.sqlite')):
                raise ValueError('Sealed notebook changed; ingest outcomes in separate evaluator copies')
            for path,h in seal['bindings'].items():
                if sha256(self.private/path)!=h:raise ValueError('Sealed predictions changed')
            if not isinstance(scores,list) or len(scores)!=4:raise ValueError('Release four scored E/F predictions')
            from .memory import Notebook
            import math
            notebook=Notebook(self.root/'research-memory.sqlite')
            try:
                for row in scores:
                    if (not isinstance(row,dict) or set(row)!={'prediction_id','outcome','absolute_error','resolution'}
                            or any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) for k in ('outcome','absolute_error','resolution'))
                            or row['absolute_error']<0 or row['resolution']<=0):raise ValueError('Only typed scalar scores may be released')
                    prediction=notebook.get(row['prediction_id'])
                    if prediction['kind']!='prediction':raise ValueError('Score must bind an existing prediction')
                    if abs(abs(prediction['payload']['value']-row['outcome'])-row['absolute_error'])>1e-15:
                        raise ValueError('Score does not reproduce prediction error')
                if len({r['prediction_id'] for r in scores})!=4:raise ValueError('Score each prediction exactly once')
            finally:notebook.close()
            target=self.root/'scores.json'
            with target.open('x') as stream:json.dump(scores,stream,allow_nan=False)
            state.pop('phase_started_at',None)
            state.update(status='revising',scores_sha256=sha256(target));save(self.private/'state.json',state)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','create','step'])
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--information-profile',choices=['scalar','physics'],default='scalar')
    parser.add_argument('--withheld-source',action='append',default=[])
    parser.add_argument('--objective')
    parser.add_argument('--config',type=Path)
    args=parser.parse_args(argv)
    if args.command=='preflight':
        result=preflight(args.workspace,information_profile=args.information_profile,withheld_sources=args.withheld_source)
        if args.config is not None:
            import os
            value=load(args.config)
            if value.get('format')=='REMAG-CODEX-POLICY/1':
                from .codex_policy import config_value
                config=config_value(value)
                result['model_readiness']=dict(model=config['model'],reasoning_effort=config['reasoning_effort'],
                    authentication='existing Codex-managed ChatGPT login',API_key_required=False,
                    binary_sha256=config['binary_sha256'],service_access_tested=False,model_calls=0)
            else:
                config=model_policy.config_value(value)
                result['model_readiness']=dict(model=config['model'],reasoning_effort=config.get('reasoning_effort'),
                    credential_present=bool(os.environ.get('OPENAI_API_KEY','').strip()),
                    service_access_tested=False,model_calls=0)
    elif args.command=='create':
        context=PolicyContext.create(args.workspace,objective=args.objective,information_profile=args.information_profile,withheld_sources=args.withheld_source)
        result=load(context.private/'state.json')
    else:
        if args.config is None:parser.error('step requires an explicit model configuration')
        context=PolicyContext(args.workspace)
        if load(args.config).get('format')=='REMAG-CODEX-POLICY/1':
            from .codex_policy import CodexPolicy
            calls=load(context.private/'state.json')['calls']
            backend=CodexPolicy(args.config,context.private/f'codex-process-{calls:03d}')
            try:result=context.step(args.config,backend=backend)
            finally:backend.close()
        else:result=context.step(args.config)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
