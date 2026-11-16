"""Direct tools for an existing chat, with no embedded model or solver runner.

Extensions are explicitly imported trusted Python code, not a sandbox. Built-in
operations reuse ResearchTools and its source, artifact and notebook validation.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
import time

from ..corpus import sha256
from ..discovery.common import save
from .tools import ResearchTools


PREPARATION = frozenset({'author_structure', 'register_structure', 'assess_structure',
    'prepare_query', 'prepare_response_series', 'analyze_response_series',
    'export_band_path', 'export_electronic_matrices', 'compare_electronic_matrices',
    'export_orbital_moments', 'export_soc_matrices'})
MEMORY = frozenset({'record_hypothesis', 'propose_experiment', 'record_prediction',
    'record_response_prediction', 'record_calculation_outcome', 'record_response_outcome',
    'record_analysis', 'retrieve_research_history', 'evaluate_hypothesis', 'evaluate_predictions'})
INSPECTION = frozenset({'inspect_structure', 'inspect_calculation', 'inspect_electronic_structure',
    'projected_dos', 'compare_electronic_states', 'structural_operation',
    'inspect_relaxation', 'calibrate_exploratory_response'})
DISABLED = frozenset({'run_prepared_query', 'prepare_relaxation'})


@dataclass(frozen=True)
class ToolDefinition:
    """Register one trusted handler(context: ResearchTools, arguments: dict).

    Required/optional names form the transport contract. Semantic argument
    validation belongs to the handler, which must precede its side effects.
    Effects/profiles describe the reviewed handler; they do not sandbox Python.
    """
    name: str
    description: str
    required: tuple
    optional: tuple
    handler: object
    effect: str
    profiles: tuple = ('physics',)

    def metadata(self):
        return dict(name=self.name, description=self.description, effect=self.effect,
            inputSchema=dict(type='object', properties={key: {} for key in self.required+self.optional},
                required=list(self.required), additionalProperties=False),
            validation='Field presence at transport; value types, units and invariants in handler')


class ToolHost:
    def __init__(self, workspace, *, information_profile='physics', withheld_sources=()):
        # Deliberately no execution_config: direct-chat calls do not grant compute.
        self.context = ResearchTools(workspace, information_profile=information_profile,
                                     withheld_sources=withheld_sources)
        self._tools = {}
        for name, schema in self.context.catalog().items():
            if name in DISABLED:
                continue
            if name in PREPARATION:
                effect = 'prepare'
            elif name in MEMORY:
                effect = 'notebook'
            elif name in INSPECTION:
                effect = 'inspect'
            else:
                raise ValueError('Classify a new built-in tool before exposing it: '+name)
            self.register(ToolDefinition(name, schema['description'], tuple(schema['required']),
                tuple(schema['optional']), self._builtin(name), effect, ('scalar', 'physics')))

    @staticmethod
    def _builtin(name):
        return lambda context, arguments: context.call(name, arguments)

    def register(self, definition):
        if not isinstance(definition, ToolDefinition):
            raise ValueError('Register a ToolDefinition')
        d = definition
        if (not isinstance(d.name, str) or not re.fullmatch('[a-z][a-z0-9_]{0,63}', d.name)
                or d.name in self._tools or d.name in DISABLED
                or not isinstance(d.description, str) or not d.description.strip()
                or not callable(d.handler) or d.effect not in ('inspect', 'prepare', 'notebook')
                or not isinstance(d.required, tuple) or not isinstance(d.optional, tuple)
                or any(not isinstance(k, str) or not k for k in d.required+d.optional)
                or len(set(d.required+d.optional)) != len(d.required+d.optional)
                or not isinstance(d.profiles, tuple) or not d.profiles
                or any(p not in ('scalar', 'physics') for p in d.profiles)):
            raise ValueError('Invalid, duplicate or reserved tool definition')
        self._tools[d.name] = d

    def catalog(self):
        return dict(format='REMAG-DIRECT-TOOLS/1', information_profile=self.context.information_profile,
            native_execution_enabled=False, cpu_relaxation_enabled=False,
            tools={name: d.metadata() for name, d in self._tools.items()
                   if self.context.information_profile in d.profiles})

    def call(self, request, receipt):
        """Retain intent before dispatch; a used receipt cannot execute again.

        A request with no response means interrupted/unknown side effects, never
        an invitation to retry. Receipts are evidence, not a security boundary.
        """
        if (not isinstance(request, dict) or set(request) != {'tool', 'args', 'rationale'}
                or not isinstance(request['tool'], str) or not isinstance(request['args'], dict)
                or not isinstance(request['rationale'], str) or not request['rationale'].strip()):
            raise ValueError('Expected tool, args and nonempty rationale')
        encoded = json.dumps(request, allow_nan=False)
        if len(encoded.encode()) > 1024*1024:
            raise ValueError('Request exceeds 1 MiB')
        request = json.loads(encoded)  # Freeze caller-owned mutable arguments.
        folder = self.context.path(str(receipt))
        folder.mkdir(parents=True, exist_ok=False)
        definition = self._tools.get(request['tool'])
        implementation = {str(Path(__file__).resolve()): sha256(Path(__file__)),
                          str(Path(inspect.getfile(ResearchTools)).resolve()): sha256(Path(inspect.getfile(ResearchTools)))}
        if definition:
            try:
                source = inspect.getsourcefile(definition.handler)
            except TypeError:
                source = None
            if source and Path(source).is_file():
                implementation[str(Path(source).resolve())] = sha256(Path(source))
        intent = dict(format='REMAG-DIRECT-TOOL-CALL/1', request=request,
            information_profile=self.context.information_profile,
            withheld_sources=sorted(self.context.withheld_sources),
            definition=definition.metadata() if definition else None,
            implementation_sha256=implementation, started_at=datetime.now(timezone.utc).isoformat(),
            native_execution_enabled=False, cpu_relaxation_enabled=False)
        save(folder/'request.json', intent)
        started = time.monotonic()
        try:
            if definition is None or self.context.information_profile not in definition.profiles:
                raise ValueError('Unknown or unavailable tool')
            args = request['args']
            if set(args)-set(definition.required+definition.optional) or set(definition.required)-set(args):
                raise ValueError('Missing or unknown tool arguments')
            # Reserve stdout for machine-readable output, even for extensions.
            with redirect_stdout(sys.stderr):
                value = definition.handler(self.context, deepcopy(args))
            response = dict(ok=True, result=value)
            if len(json.dumps(response, allow_nan=False).encode()) > 8*1024*1024:
                raise ValueError('Response exceeds 8 MiB; inspect saved artifacts before any follow-up')
        except Exception as exc:
            response = dict(ok=False, error=dict(type=type(exc).__name__, message=str(exc)),
                            retry_safe=False)
        response.update(format='REMAG-DIRECT-TOOL-RESULT/1', request_sha256=sha256(folder/'request.json'),
            elapsed_seconds=time.monotonic()-started, finished_at=datetime.now(timezone.utc).isoformat(),
            receipt=str(folder.relative_to(self.context.root)))
        save(folder/'response.json', response)
        return response


def create_host(workspace, *, extensions=(), **options):
    host = ToolHost(workspace, **options)
    for name in extensions:
        # Opt-in program configuration only; request JSON cannot import code.
        if not re.fullmatch(r'[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*', name):
            raise ValueError('Expected a trusted extension module name')
        with redirect_stdout(sys.stderr):
            importlib.import_module(name).register_tools(host)
    return host


def read_request(path):
    if path == '-':
        text = sys.stdin.read(1024*1024+1)
    else:
        with Path(path).open() as stream:
            text = stream.read(1024*1024+1)
    if len(text.encode()) > 1024*1024:
        raise ValueError('Request exceeds 1 MiB')
    return json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
