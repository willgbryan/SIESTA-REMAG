#!/usr/bin/env python3
"""Verify retained isolated-component fixtures, outputs and oracle provenance.

This verifies recorded content hashes, not numerical correctness or binary
reproducibility. Source archives have a separate verify_evidence.py check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root):
    scripts={}
    with tarfile.open(root/'implementation.tar.gz') as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.startswith('scripts/'):
                with archive.extractfile(member) as stream:
                    scripts[member.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    checks=[];components=[]
    def record(path,expected):
        actual=digest(path) if path.is_file() else None
        checks.append(dict(path=str(path.relative_to(root)),expected_sha256=expected,actual_sha256=actual,passed=actual==expected))
    for path in sorted(root.glob('*-check/check.json')):
        report=json.loads(path.read_text())
        if not isinstance(report,dict):continue
        fixtures=report.get('fixtures')
        if fixtures is None and path.parent.name in ('scf-cycle-check','scf-cycle-memory-check'):
            manifest=path.parent/'fixtures.json'
            if manifest.is_file():
                fixtures=json.loads(manifest.read_text())
        if path.parent.name in ('orbital-owner-check','atomic-owner-check','grid-owner-check','two-center-owner-check','nonlocal-owner-check','pair-geometry-check'):
            components.append(path.parent.name)
            expected=report['checker_sha256'];matches=[name for name,value in scripts.items() if value==expected]
            checks.append(dict(component=path.parent.name,archive_sources=matches,expected_sha256=expected,passed=bool(matches)))
            for row in report['checks']:
                for key in ('report','owner','log',*(['residency'] if path.parent.name in ('two-center-owner-check','nonlocal-owner-check','pair-geometry-check') else [])):record(path.parent/row[key+'_file'],row[key+'_sha256'])
            continue
        if path.parent.name in ('population-owner-check','population-observables-check'):
            components.append(path.parent.name)
            expected=report['checker_sha256'];matches=[name for name,value in scripts.items() if value==expected]
            checks.append(dict(component=path.parent.name,archive_sources=matches,expected_sha256=expected,passed=bool(matches)))
            for row in (report['checks'] if path.parent.name=='population-owner-check' else [report]):
                for name,expected in row['evidence_sha256'].items():record(path.parent/name,expected)
            continue
        if path.parent.name in ('pair-handoff-check','integral-publication-check'):
            components.append(path.parent.name)
            expected=report['checker_sha256'];matches=[name for name,value in scripts.items() if value==expected]
            checks.append(dict(component=path.parent.name,archive_sources=matches,expected_sha256=expected,passed=bool(matches)))
            for row in report['checks']:
                for key in ('report','log',*(['residency'] if path.parent.name=='integral-publication-check' else [])):record(path.parent/row[key+'_file'],row[key+'_sha256'])
            continue
        if fixtures is None and path.parent.name in ('grid-force-check','atomic-grid-force-check','final-force-check','orbital-basis-check','orbital-comparison-check','atomic-basis-check','atomic-comparison-check','atomic-masked-force-check','two-center-check','two-center-comparison-check','pair-force-check','population-check','soc-projector-check','nonlocal-force-check','nonlocal-compact-check'):
            components.append(path.parent.name)
            expected=report.get('checker_sha256')
            matches=[name for name,value in scripts.items() if value==expected]
            checks.append(dict(component=path.parent.name,archive_sources=matches,expected_sha256=expected,passed=bool(matches)))
            record(path.parent/report['output_file'],report['output_sha256'])
            if path.parent.name in ('two-center-check','pair-force-check','population-check','soc-projector-check','nonlocal-force-check','nonlocal-compact-check'):record(path.parent/report['inputs_file'],report['inputs_sha256'])
            if path.parent.name in ('final-force-check','orbital-comparison-check','atomic-comparison-check','two-center-comparison-check'):continue
            if path.parent.name in ('orbital-basis-check','atomic-basis-check'):record(path.parent/report['geometry_file'],report['geometry_sha256'])
            oracle=path.parent/report['oracle_file'];record(oracle,report['oracle_sha256'])
            if oracle.is_file():
                reference=json.loads(oracle.read_text())
                record(path.parent/reference['arrays_file'],report['oracle_arrays_sha256'])
                record(path.parent/reference['wrapper_file'],reference['wrapper_sha256'])
            continue
        if fixtures is None:continue
        if 'fixtures_sha256' in report:record(path.parent/'fixtures.json',report['fixtures_sha256'])
        components.append(path.parent.name)
        expected=report.get('fixture_script_sha256')
        matches=[name for name,value in scripts.items() if value==expected]
        checks.append(dict(component=path.parent.name,archive_sources=matches,expected_sha256=expected,passed=bool(matches)))
        for fixture in fixtures:record(path.parent/fixture['input_file'],fixture['input_sha256'])
        for result in report.get('checks',[]):
            if 'output_sha256' in result:record(path.parent/result['output_file'],result['output_sha256'])
        if 'oracle_file' in report:
            oracle=path.parent/report['oracle_file'];record(oracle,report['oracle_sha256'])
            if oracle.exists():
                reference=json.loads(oracle.read_text())
                record(path.parent/reference['arrays_file'],report['oracle_arrays_sha256'])
                if 'wrapper_file' in reference:record(path.parent/reference['wrapper_file'],reference['wrapper_sha256'])
        for name,expected_hash in report.get('wrappers_sha256',{}).items():record(path.parent/name,expected_hash)
        if 'mixer_wrapper_sha256' in report:record(path.parent/'mixing-reference.f90',report['mixer_wrapper_sha256'])
    return dict(scope=__doc__,components=components,passed=bool(checks) and all(c['passed'] for c in checks),
                recorded_hash_count=len(checks),checks=checks)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();result=verify(args.run)
    (args.run/'component-evidence-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"Component evidence: {result['recorded_hash_count']} hashes, passed={result['passed']}")
    if not result['passed']:raise SystemExit(1)
