"""Fixed synthetic graph-setup diagnostic; never a material acceptance gate."""
import json
import re
from pathlib import Path

CASES = [
    dict(name='n30-k132-grid4',dimension=30,kpoints=132,mesh=4),
    dict(name='n76-k132-grid4',dimension=76,kpoints=132,mesh=4),
    dict(name='n30-k948-grid4',dimension=30,kpoints=948,mesh=4),
    dict(name='n76-k948-grid4',dimension=76,kpoints=948,mesh=4),
    dict(name='n76-k132-grid40',dimension=76,kpoints=132,mesh=40),
]


def validate_policy(policy):
    if (policy.get('format')!='REMAG-SCF-SETUP-PROBE/1' or policy.get('cases')!=CASES
            or policy.get('timeout_seconds')!=420 or policy.get('reservation_cents')!=53
            or policy.get('retries')!=0 or policy.get('gpu_count')!=1
            or policy.get('cycle_launch') is not False):
        raise ValueError('Expected the fixed finite setup-only diagnostic')
    for key,prefix in [('image','im-'),('previous_app','ap-')]:
        if not re.fullmatch(prefix+r'[A-Za-z0-9]+',policy.get(key,'')):
            raise ValueError('Explicit image and prior app identities required')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',policy.get('volume','')):
        raise ValueError('Explicit existing result volume required')
    if (not isinstance(policy.get('files'),dict) or not policy['files']
            or not {'runtime.json','probe_scf_setup.cpp'}<=policy['files'].keys()):
        raise ValueError('Missing bound diagnostic inputs')
    for name,sha in policy['files'].items():
        if Path(name).name!=name or name in ('.','..') or not re.fullmatch(r'[0-9a-f]{64}',sha):
            raise ValueError('Unsafe or unbound diagnostic input')
    for key in ('module_sha256','helper_sha256','held_owner_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}',policy.get(key,'')):
            raise ValueError('Missing diagnostic/owner hash')
    return policy


def summarize_case(case, stdout, returncode, timed_out):
    """Retain malformed/signal/partial output without converting it to a pass."""
    if case not in CASES:raise ValueError('Unregistered diagnostic shape')
    messages=[];errors=[]
    for line in stdout.splitlines():
        if line.startswith('{'):
            try:messages.append(json.loads(line))
            except json.JSONDecodeError as e:errors.append(str(e))
    setup=next((r for r in messages if isinstance(r,dict) and r.get('stage')=='setup'),{})
    end=messages[-1] if messages and isinstance(messages[-1],dict) else {}
    matches=(setup.get('n')==case['dimension'] and setup.get('nk')==case['kpoints'] and setup.get('mesh')==case['mesh'])
    constructed=(returncode==0 and not timed_out and not errors and matches and end.get('stage')=='constructed'
        and end.get('forbidden_nodes')==0 and type(end.get('graph_nodes')) is int and end['graph_nodes']>0
        and end.get('jobs')==2*case['kpoints'] and end.get('lanes')==8)
    return dict(case,returncode=returncode,timed_out=timed_out,messages=messages,parse_errors=errors,
        constructed=constructed,native_runtime_repaired=False,scientific_acceptance=False)
