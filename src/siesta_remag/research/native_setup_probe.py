"""Original-input initialization diagnostic; never a native acceptance gate."""
import json
from pathlib import Path
import re

INPUTS = {'runtime.json', 'input.fdf', 'Ni.psml',
          'probe_cycle_barrier.cpp', 'probe_graph_interpose.cpp'}


def validate_policy(policy):
    if (policy.get('format') != 'REMAG-NATIVE-SETUP-PROBE/1'
            or policy.get('timeout_seconds') != 180 or policy.get('reservation_cents') != 23
            or policy.get('gpu_count') != 1 or policy.get('retries') != 0
            or policy.get('cycle_launch') is not False):
        raise ValueError('Expected fixed, finite native initialization diagnostic')
    for key, prefix in [('image', 'im-'), ('previous_app', 'ap-'), ('held_app', 'ap-')]:
        if not re.fullmatch(prefix + r'[A-Za-z0-9]+', policy.get(key, '')):
            raise ValueError('Missing explicit prior allocation identity')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', policy.get('volume', '')):
        raise ValueError('Missing existing volume')
    if set(policy.get('files', {})) != INPUTS:
        raise ValueError('Unexpected diagnostic inputs')
    for name, value in policy['files'].items():
        if Path(name).name != name or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('Unbound input')
    for key in ('helper_sha256', 'module_sha256', 'held_owner_sha256', 'driver_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', policy.get(key, '')):
            raise ValueError('Unbound diagnostic code or owner')
    return policy


def barrier_reached(returncode, marker, *, timed_out=False):
    return (not timed_out and returncode == 90 and isinstance(marker, dict)
            and marker.get('format') == 'REMAG-CYCLE-BARRIER/1'
            and type(marker.get('geometry')) is int
            and marker.get('SCF_cycle_launched') is False and marker.get('intentional_exit') == 90)


def summarize(directory, returncode, timed_out):
    directory = Path(directory)
    errors, events = [], []
    marker = None
    try:
        if (directory / 'cycle-barrier.json').exists():
            marker = json.loads((directory / 'cycle-barrier.json').read_text())
        for line in (directory / 'graph-interpose.log').read_text().splitlines():
            events.append(json.loads(line))
    except (ValueError, OSError) as error:
        errors.append(str(error))
    return {'returncode': returncode, 'timed_out': timed_out, 'barrier': marker,
            'barrier_reached': barrier_reached(returncode, marker, timed_out=timed_out),
            'events': events, 'parse_errors': errors,
            'native_runtime_repaired': False, 'scientific_acceptance': False}
