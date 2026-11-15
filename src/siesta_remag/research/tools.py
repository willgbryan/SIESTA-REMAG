"""Experimental local tool protocol for Astra or another research agent.

The agent supplies the scientific policy. This adapter supplies durable memory,
bounded data access and prepared physical questions. It neither calls a model
service nor submits paid work as a side effect of inspection or preparation.
"""
import json
from pathlib import Path
import sys

from ..corpus import sha256
from ..discovery.common import load
from .experiments import prepare_experiment
from .inspection import inspect_electronic, projected_dos, compare_electronic
from .memory import Notebook


TOOLS = {
    'run_prepared_query': {'required':['bundle','output'],'optional':['spectra'],
        'description':'Execute one prepared REMAG query through the existing single-GPU budget/account guards, collect, stop the owned app and review. Requires operator-enabled execution configuration.'},
    'export_orbital_moments': {'required':['report','stage','native_directory','output'], 'optional':['collection'],
        'description':'Inspect native-density orbital/atom moments using the pinned onsite approximation and final printed-table check; no solver run.'},
    'export_electronic_matrices': {'required':['report','stage','native_directory','output'],
        'optional':['k_indices','collinear_spin','collection'],
        'description':'Reconstruct eigenvectors, density and Lowdin orbital-coupling blocks at up to eight saved native k points; no solver run.'},
    'compare_electronic_matrices': {'required':['before','after','output'],
        'optional':['energy_window_ev','degeneracy_tolerance_ev','limit'],
        'description':'Associate same-basis endpoint states with overlap/subspace diagnostics and total Hamiltonian changes; does not isolate SOC or compute MAE.'},
    'prepare_relaxation': {'required':['structure','pseudos','weights','output'], 'optional':['settings','orb_threads'],
        'description':'Bind an exact source and run bounded CPU Orb preparation using supplied weights; no native/GPU submission.'},
    'inspect_relaxation': {'required':['structure','bundle','report'],
        'optional':['cutoff_angstrom','position_tolerance_angstrom','metric_relative_tolerance'],
        'description':'Verify a saved accepted native relaxation and report source-linked site/orbit, neighbor and authored-symmetry changes.'},
    'calibrate_exploratory_response': {'required': ['exploratory','reference','baseline_source_sha256','resolution_ev'], 'optional': [],
        'description': 'Compare response errors, signs and rankings from saved matched structures; selects no universal default.'},
    'inspect_calculation': {'required': ['report'], 'optional': ['collection'],
        'description': 'Verify a saved calculation and inspect measured stages, moments, forces, stress and anisotropy.'},
    'inspect_electronic_structure': {'required': ['spectrum'], 'optional': ['selection'],
        'description': 'Select near-Fermi states by atom, orbital, angular momentum and k-point indices.'},
    'projected_dos': {'required': ['spectrum'], 'optional': ['selection', 'sigma_ev', 'points'],
        'description': 'Inspect signed orbital projections on a bounded energy grid.'},
    'compare_electronic_states': {'required': ['before', 'after'], 'optional': ['selection', 'sigma_ev', 'points'],
        'description': 'Compare matched electronic samplings without assuming band tracking or SOC attribution.'},
    'structural_operation': {'required': ['request'], 'optional': [],
        'description': 'Use the existing GPAC typed JSON worker for exact inspection, authoring and modification.'},
    'prepare_query': {'required': ['structure', 'pseudos', 'output', 'query'],
        'optional': ['settings', 'mesh', 'basis_size', 'basis_energy_shift_ry', 'basis_spec', 'parameter'],
        'description': 'Prepare scalar, selected-direction or angular work; does not execute or reserve compute.'},
    'record_hypothesis': {'required': ['statement', 'predictions', 'alternatives'], 'optional': [],
        'description': 'Record a testable hypothesis and competing explanations in durable research memory.'},
    'propose_experiment': {'required': ['hypothesis_id', 'structure_sha256', 'action', 'rationale'], 'optional': [],
        'description': 'Record why an inspection or calculation distinguishes the hypothesis.'},
    'record_prediction': {'required': ['experiment_id', 'observable', 'value', 'unit'], 'optional': ['uncertainty'],
        'description': 'Record a prediction before an outcome exists; later revisions cannot overwrite it.'},
    'record_calculation_outcome': {'required': ['experiment_id', 'report', 'stage', 'observable'], 'optional': ['collection'],
        'description': 'Bind an outcome to an actual saved report; energy observables require accepted native stages.'},
    'evaluate_hypothesis': {'required': ['experiment_id'], 'optional': [],
        'description': 'Compare a recorded prediction with its measured outcome; agreement is not mechanism proof.'},
    'record_analysis': {'required': ['parent_id', 'conclusion', 'evidence'], 'optional': ['status'],
        'description': 'Append an evidence-bound interpretation without rewriting prior hypotheses or outcomes.'},
    'retrieve_research_history': {'required': [], 'optional': ['query', 'limit'],
        'description': 'Retrieve previous hypotheses, interventions, predictions, failures and interpretations.'},
}


def _verified_report(path, collection=None):
    root = path.parent
    report = load(path)
    if report.get('format') != 'REMAG-EXPERIMENT-RESULT/1':
        raise ValueError('Expected a research calculation report')
    if report['source_sha256'] != report['request']['source_sha256']:
        raise ValueError('Calculation source binding differs')
    collected = None
    if collection is not None:
        collection=Path(collection).resolve()
        manifest=load(collection)
        if collection.name!='archive-manifest.json' or manifest.get('schema')!=1:
            raise ValueError('Expected the original compact archive manifest')
        if not path.is_relative_to(collection.parent):raise ValueError('Report is outside the collection')
        collected=manifest['files']
        if 'report.json' not in collected:raise ValueError('Collection lacks its root report binding')
        key=str(path.relative_to(collection.parent))
        if collected.get(key)!=sha256(path):raise ValueError('Report is not bound by the collection')
        top=load(collection.parent/'report.json')
        if (top['status']=='running' or top['status']!=manifest['status'] or
            top['invocation_sha256']!=manifest['invocation_sha256']):
            raise ValueError('Collection invocation differs')
        for name,expected in collected.items():
            artifact=(collection.parent/name).resolve()
            if not artifact.is_relative_to(collection.parent) or sha256(artifact)!=expected:
                raise ValueError('Collected artifact binding differs: '+name)
        prefix=path.parent.relative_to(collection.parent)
        for stage in report['stages']:
            if stage['accepted']:
                for filename in ('result.json','stdout.log','input.fdf','review.json'):
                    if str(prefix/stage['stage']/filename) not in collected:
                        raise ValueError('Collection lacks accepted stage evidence: '+filename)
                stage_path=prefix/stage['stage']
                if not any(Path(name).parent==stage_path and Path(name).match('fdf*.log') for name in collected):
                    raise ValueError('Collection lacks effective FDF evidence')
    omitted=[]
    for name, expected in report['artifacts'].items():
        artifact = (root/name).resolve()
        if not artifact.is_relative_to(root):raise ValueError('Calculation artifact escapes result directory')
        if not artifact.exists() and collected is not None:
            if str(artifact.relative_to(collection.parent)) in collected:
                raise ValueError('Collected artifact is missing: '+name)
            omitted.append(name)
        elif sha256(artifact) != expected:
            raise ValueError('Calculation artifact binding differs: '+name)
    report=dict(report,verification=dict(mode='compact collection' if collected is not None else 'full local artifacts',
        omitted_artifacts=omitted,collection_manifest_sha256=sha256(collection) if collection else None,
        scope='All collected and locally present report artifacts verified; omitted remote bytes are not reverified'))
    return report


class ResearchTools:
    def __init__(self, workspace, *, execution_config=None):
        self.root = Path(workspace).resolve()
        if not self.root.is_dir():
            raise ValueError('Research workspace must already exist')
        self.execution_config = self.path(str(execution_config)) if execution_config is not None else None

    def catalog(self):
        return {k:v for k,v in TOOLS.items() if k != 'run_prepared_query' or self.execution_config is not None}

    def path(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError('Expected a workspace-relative artifact path')
        path = (self.root/value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Artifact path escapes the research workspace')
        return path

    def call(self, name, args):
        if name not in TOOLS or not isinstance(args, dict):
            raise ValueError('Unknown tool or invalid arguments')
        schema = TOOLS[name]
        if set(args)-set(schema['required']+schema['optional']) or set(schema['required'])-set(args):
            raise ValueError('Missing or unknown tool arguments')
        if name == 'run_prepared_query':
            if self.execution_config is None:
                raise ValueError('Native execution is not enabled by the operator')
            from .execution import execute_query
            return execute_query(self.root,self.execution_config,self.path(args['bundle']),self.path(args['output']),spectra=args.get('spectra'))
        if name == 'export_orbital_moments':
            from .orbital_moments import export_orbital_moments
            return export_orbital_moments(self.path(args['report']), args['stage'], self.path(args['native_directory']),
                self.path(args['output']), collection=self.path(args['collection']) if 'collection' in args else None)
        if name == 'export_electronic_matrices':
            from .mechanisms import export_matrices
            return export_matrices(self.path(args['report']), args['stage'], self.path(args['native_directory']),
                self.path(args['output']), collection=self.path(args['collection']) if 'collection' in args else None,
                **{k:v for k,v in args.items() if k in ('k_indices','collinear_spin')})
        if name == 'compare_electronic_matrices':
            from .mechanisms import compare_matrices
            return compare_matrices(self.path(args['before']),self.path(args['after']),self.path(args['output']),
                **{k:v for k,v in args.items() if k in ('energy_window_ev','degeneracy_tolerance_ev','limit')})
        if name == 'prepare_relaxation':
            from .relaxation import prepare_relaxation
            from ..discovery.config import Settings
            return prepare_relaxation(load(self.path(args['structure'])),self.path(args['pseudos']),
                self.path(args['weights']),self.path(args['output']),settings=Settings.from_dict(args.get('settings',{})),
                orb_threads=args.get('orb_threads',4))
        if name == 'inspect_relaxation':
            from .relaxation import inspect_relaxation
            return inspect_relaxation(load(self.path(args['structure'])),self.path(args['bundle']),self.path(args['report']),
                **{k:v for k,v in args.items() if k not in ('structure','bundle','report')})
        if name == 'calibrate_exploratory_response':
            from .calibration import calibrate_response
            return calibrate_response([self.path(p) for p in args['exploratory']], [self.path(p) for p in args['reference']],
                baseline_source_sha256=args['baseline_source_sha256'],resolution_ev=args['resolution_ev'])
        if name == 'inspect_calculation':
            path = self.path(args['report']); report = _verified_report(path,self.path(args['collection']) if 'collection' in args else None)
            observations = {stage: {k:v for k,v in row.items() if k != 'eigenvalues'}
                            for stage,row in report['observations'].items()}
            return dict(format='REMAG-CALCULATION-INSPECTION/1', report_sha256=sha256(path),
                status=report['status'], source_sha256=report['source_sha256'], request=report['request'],
                executable_sha256=report['executable_sha256'], stages=report['stages'], observations=observations,
                anisotropy=report.get('anisotropy'), angular_fit=report.get('angular_fit'),
                wall_seconds=report.get('wall_seconds'), latency_scope=report.get('latency_scope', 'Recorded experiment time only'),
                verification=report['verification'],physical_accuracy_qualified=False)
        if name in ('inspect_electronic_structure','projected_dos','compare_electronic_states'):
            options = {k:v for k,v in args.items() if k in ('selection','sigma_ev','points')}
            if name == 'compare_electronic_states':
                return compare_electronic(self.path(args['before']),self.path(args['after']),**options)
            function = inspect_electronic if name == 'inspect_electronic_structure' else projected_dos
            return function(self.path(args['spectrum']),**options)
        if name == 'structural_operation':
            from gpac.worker import handle_request
            return handle_request(args['request'])
        if name == 'prepare_query':
            from ..discovery.config import Settings
            options = {k:v for k,v in args.items() if k not in ('structure','pseudos','output','settings')}
            options['settings'] = Settings.from_dict(args.get('settings',{}))
            return prepare_experiment(load(self.path(args['structure'])), self.path(args['pseudos']),
                                      self.path(args['output']), **options)
        notebook = Notebook(self.path('research-memory.sqlite'))
        try:
            if name == 'record_calculation_outcome':
                return self._outcome(notebook,args)
            if name == 'record_analysis':
                return notebook.record_analysis(args['parent_id'], conclusion=args['conclusion'],
                    report_sha256=sha256(self.path(args['evidence'])), status=args.get('status','inconclusive'))
            method = {'evaluate_hypothesis':'evaluate','retrieve_research_history':'retrieve'}.get(name,name)
            return getattr(notebook,method)(**args)
        finally:
            notebook.close()

    def _outcome(self, notebook, args):
        path = self.path(args['report']); report = _verified_report(path,self.path(args['collection']) if 'collection' in args else None)
        experiment = notebook.get(args['experiment_id'])
        if experiment['kind'] != 'experiment' or experiment['payload']['structure_sha256'] != report['source_sha256']:
            raise ValueError('Outcome source differs from the recorded experiment')
        stage, observable = args['stage'], args['observable']
        if stage == 'anisotropy':
            mapping = {'total_energy_ev':'mae_ev_cell','free_energy_ev':'free_energy_ev_cell'}
            if observable not in mapping or report.get('anisotropy') is None:
                raise ValueError('Requested anisotropy was not measured')
            # Compute both observables directly from the stored directional energies.
            value = report['observations']['soc-x'][observable]-report['observations']['soc-z'][observable]
            accepted = all(s['accepted'] and s.get('branch',{}).get('passed',True) for s in report['stages'])
            unit = 'eV/cell'
        else:
            if observable not in ('total_energy_ev','free_energy_ev') or stage not in report['observations']:
                raise ValueError('Select a measured stage E/F observable')
            value = report['observations'][stage][observable]
            accepted = any(s['stage']==stage and s['accepted'] and s.get('branch',{}).get('passed',True) for s in report['stages'])
            unit = 'eV/cell'
        status = 'measured' if report['status']=='measured' and accepted else 'failed'
        return notebook.record_outcome(args['experiment_id'], observable=stage+'.'+observable, value=value if status=='measured' else None,
            unit=unit, report_sha256=sha256(path), status=status)


def serve(workspace, *, execution_config=None):
    """One request per line. Large arrays stay in artifacts, never in tool replies."""
    tools = ResearchTools(workspace,execution_config=execution_config)
    for line in sys.stdin:
        try:
            if len(line.encode()) > 1024*1024:
                raise ValueError('Request exceeds 1 MiB')
            request = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
            if not isinstance(request,dict): raise ValueError('Expected a tool request object')
            if request == {'op':'catalog'}:
                result = dict(format='REMAG-RESEARCH-TOOLS/1', tools=tools.catalog(),
                    selections=dict(atom_indices='zero-based atom indices', orbital_indices='zero-based basis indices',
                        l='angular momentum indices', k_indices='zero-based sampled k indices',
                        energy_window_ev='[low,high] relative to each spectrum Fermi level', limit='1..200 states'))
            else:
                if set(request) != {'op','args'}:raise ValueError('Expected op and args')
                result = tools.call(request['op'],request['args'])
            response = dict(ok=True,result=result)
        except Exception as exc:
            response = dict(ok=False,error=dict(type=type(exc).__name__,message=str(exc)))
        encoded = json.dumps(response,allow_nan=False)
        if len(encoded.encode()) > 8*1024*1024:
            encoded = json.dumps(dict(ok=False,error=dict(type='ResponseLimit',message='Refine the selection; response exceeds 8 MiB')))
        print(encoded,flush=True)
