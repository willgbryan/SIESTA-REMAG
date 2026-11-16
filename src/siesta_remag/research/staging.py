"""Scalar-only initial data, without raw archives or arbitrary dictionaries.

This is a staging contract for trusted policy contexts, not an OS sandbox.
"""
import json
import math
import re

from ..corpus import sha256
from ..discovery.common import digest

FORMAT = 'REMAG-SCALAR-DATA/1'
STAGES = frozenset({'scalar', 'soc-z', 'soc-x', 'soc-y'})
HASH_FIELDS = frozenset({'source_sha256', 'report_sha256', 'executable_sha256',
                         'collection_sha256'})


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _vector(value):
    return isinstance(value, list) and len(value) == 3 and all(_number(x) for x in value)


def _observations(values):
    from .tools import STANDARD_OBSERVATIONS
    if not isinstance(values, dict) or not values or set(values)-STAGES:
        raise ValueError('Scalar packet requires known observation stages')
    for row in values.values():
        if not isinstance(row, dict) or set(row)-STANDARD_OBSERVATIONS:
            raise ValueError('Scalar packet contains nonstandard observations')
        for key, value in row.items():
            if key in ('total_energy_ev', 'free_energy_ev'):
                valid = _number(value)
            elif key in ('scf_iterations', 'spin_components'):
                valid = type(value) is int and value >= 0
            elif key == 'total_moment_muB':
                valid = _vector(value)
            else:
                valid = (isinstance(value, list) and 1 <= len(value) <= 200
                         and all(_vector(v) for v in value)
                         and (key != 'stress_ev_ang3' or len(value) == 3))
            if not valid:
                raise ValueError('Invalid scalar observation value: '+key)


def validate_scalar_data(packet):
    """Reject extra fields, nested payloads and changed bytes before exposure."""
    if (not isinstance(packet, dict) or set(packet) != {'format', 'cases', 'sha256'}
            or packet['format'] != FORMAT):
        raise ValueError('Expected a sealed scalar data packet')
    cases = packet['cases']
    if not isinstance(cases, dict) or not 1 <= len(cases) <= 100:
        raise ValueError('Scalar packet needs 1..100 cases')
    for name, row in cases.items():
        if not isinstance(name, str) or not re.fullmatch('[a-zA-Z0-9_-]{1,64}', name):
            raise ValueError('Invalid scalar case identifier')
        required = HASH_FIELDS | {'observations'}
        if not isinstance(row, dict) or not required <= set(row) or set(row)-required-{'report_interval'}:
            raise ValueError('Unexpected scalar case fields')
        for key in HASH_FIELDS:
            value = row[key]
            if value is None and key == 'collection_sha256':
                continue
            if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
                raise ValueError('Invalid scalar provenance hash')
        _observations(row['observations'])
        interval = row.get('report_interval')
        if interval is not None:
            from .discovery_scores import report_interval
            if not isinstance(interval, dict) or set(interval) != {'started_at', 'wall_seconds'}:
                raise ValueError('Invalid scalar acquisition interval')
            if report_interval(interval) is None:
                raise ValueError('Scalar acquisition interval cannot have missing values')
    if packet['sha256'] != digest({k: v for k, v in packet.items() if k != 'sha256'}):
        raise ValueError('Scalar packet digest differs')
    return packet


def stage_scalar_data(context, calculations, output):
    """Verify reports, project standard numeric values, write one new packet.

    Keep reports outside the policy workspace. Both arms get this same packet;
    only the physics arm gets separate electronic inputs. No archive is copied.
    """
    from .tools import STANDARD_OBSERVATIONS, _verified_report
    if not isinstance(calculations, dict) or not 1 <= len(calculations) <= 100:
        raise ValueError('Stage 1..100 named calculation endpoints')
    target = context.path(output)
    if target.exists():
        raise ValueError('Scalar packet output must be new')
    cases = {}
    for name, endpoint in calculations.items():
        if (not isinstance(endpoint, dict) or 'report' not in endpoint
                or set(endpoint)-{'report', 'collection'}):
            raise ValueError('Expected report and optional collection endpoint')
        path = context.path(endpoint['report'])
        collection = context.path(endpoint['collection']) if 'collection' in endpoint else None
        report = _verified_report(path, collection)
        context._source_access(report['source_sha256'])
        if report['status'] != 'measured' or not all(
                s['accepted'] and s.get('branch', {}).get('passed', True) for s in report['stages']):
            raise ValueError('Stage only measured calculations with accepted branches')
        observations = {stage: {key: value for key, value in row.items() if key in STANDARD_OBSERVATIONS}
                        for stage, row in report['observations'].items()}
        from .discovery_scores import report_interval
        cases[name] = dict(source_sha256=report['source_sha256'], report_sha256=sha256(path),
            executable_sha256=report['executable_sha256'],
            collection_sha256=sha256(collection) if collection else None, observations=observations,
            report_interval=report_interval(report))
    packet = dict(format=FORMAT, cases=cases)
    packet['sha256'] = digest(packet)
    validate_scalar_data(packet)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        stream.write(json.dumps(packet, indent=2, allow_nan=False)+'\n')
    return dict(packet=output, sha256=sha256(target), cases=list(cases), raw_artifacts_copied=0)


def scalar_data_provenance(data):
    """Keep original acquisition costs in study accounting without raw reports.

    Legacy packets without timing still require their report's cost coverage;
    missing time bindings remain unknown rather than silently implying zero.
    """
    if isinstance(data, dict):
        if data.get('format') == FORMAT:
            validate_scalar_data(data)
            for row in data['cases'].values():
                yield row['report_sha256'], row.get('report_interval')
        else:
            for value in data.values():
                yield from scalar_data_provenance(value)
    elif isinstance(data, list):
        for value in data:
            yield from scalar_data_provenance(value)


def check_scalar_input(data, withheld_sources=()):
    """Catch raw reports and the original nested packet leak at study preflight.

    Policy code, prose and external context still require operator review.
    This does not certify blinding or filesystem isolation.
    """
    if isinstance(data, dict):
        if data.get('format') == FORMAT:
            validate_scalar_data(data)
            if any(row['source_sha256'] in withheld_sources for row in data['cases'].values()):
                raise ValueError('Do not seed an arm with heldout calculation results')
            return
        if data.get('format') in ('REMAG-EXPERIMENT-RESULT/1', 'REMAG-DISCOVERY-RESULT/1',
                                 'REMAG-CALCULATION-INSPECTION/1') or 'observations' in data:
            raise ValueError('Scalar initial data must use a verified scalar packet; raw observations are forbidden')
        for value in data.values():
            check_scalar_input(value, withheld_sources)
    elif isinstance(data, list):
        for value in data:
            check_scalar_input(value, withheld_sources)
