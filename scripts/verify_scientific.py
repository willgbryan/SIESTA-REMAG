#!/usr/bin/env python3
"""Reproduce scientific comparisons from raw stdout, EIG, XV, RHO and DRHO.

Geometry is independently reparsed from the final XV file. Recorded results
and gates are never rewritten. Every repeated and cache-control MAE pair is required.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile

from siesta_remag.compare import compare, compare_mae
from siesta_remag.output import parse_output, read_eigenvalues, read_geometry, grid_artifact
from siesta_remag.energy_density_output import read_tsde


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root):
    ledger = json.loads((root/'ledger.json').read_text())
    metadata = json.loads((root/'metadata.json').read_text())
    checks, hashes, records = [], {}, {}
    with tarfile.open(root/'implementation.tar.gz') as archive:
        for fn in (compare, parse_output, read_tsde):
            source = Path(inspect.getsourcefile(fn))
            if hashlib.sha256(archive.extractfile('src/siesta_remag/'+source.name).read()).hexdigest() != sha(source):
                raise ValueError('Loaded scientific parser/comparator differs from executed archive')

    def read(folder):
        if folder in records:
            return records[folder]
        p = root/folder
        result = json.loads((p/'result.json').read_text())
        manifest = json.loads((p/'manifest.json').read_text())
        if manifest != result['benchmark'] or sha(p/'input.fdf') != manifest.get('input_sha256'):
            raise ValueError('Raw input provenance differs: '+folder)
        parsed = parse_output((p/'stdout.log').read_text())
        parsed['eigenvalues'] = read_eigenvalues(p/'remag.EIG')
        parsed['geometry'] = read_geometry(p/'remag.XV')
        if any(result['observables'].get(name) != value for name, value in parsed.items()):
            raise ValueError('Raw observable differs: '+folder)
        for name in ('result.json', 'stdout.log', 'remag.EIG', 'remag.XV', 'remag.RHO', 'remag.DRHO', 'input.fdf', 'manifest.json'):
            artifact=grid_artifact(p/name) if name in ('remag.RHO','remag.DRHO') else p/name
            hashes[str(artifact.relative_to(root))] = sha(artifact)
        for name in ('remag.TSDE', 'remag.nc'):
            if (p/name).is_file():hashes[str((p/name).relative_to(root))] = sha(p/name)
        records[folder] = result
        return result

    for row in ledger:
        a = row.get('reference_path', row['benchmark']+'/reference')
        b = row.get('candidate_path', row['benchmark']+'/candidate')
        result = compare(read(a), read(b), reference_dir=root/a, candidate_dir=root/b)
        checks.append(dict(name=row['benchmark'], passed=result['passed'] and all(row.get(k) == v for k, v in result.items())))
    for kind in ('grid-force', 'atomic-force', 'final-grid', 'final-density', 'orbital-basis', 'atomic-basis', 'vector-cache'):
        for path in sorted(root.glob('ablation-'+kind+'-*.json')):
            case = path.stem.removeprefix('ablation-'+kind+'-')
            a, b = case+'/candidate-without-'+kind, case+'/candidate'
            result = compare(read(a), read(b), reference_dir=root/a, candidate_dir=root/b)
            checks.append(dict(name=path.name, passed=result['passed'] and result == json.loads(path.read_text())))
    for family, label in (('fept2_soc', 'fept2'), ('fept2_soc_gamma', 'fept2-gamma'), ('fept16_soc', 'fept16')):
        if not {family+'_x', family+'_z'} <= set(metadata['cases']):
            continue
        roles = {Path(row.get('candidate_path', row['benchmark']+'/candidate')).name for row in ledger
                 if Path(row.get('candidate_path', row['benchmark']+'/candidate')).parts[0] in (family+'_x', family+'_z')}
        for role in sorted(roles):
            path = root/('mae-'+label+('' if role == 'candidate' else '-'+role)+'.json')
            folders = [family+'_x/reference', family+'_z/reference', family+'_x/'+role, family+'_z/'+role]
            result = compare_mae(*[read(p) for p in folders], directories=[root/p for p in folders])
            result['variant'] = role
            checks.append(dict(name=path.name, passed=path.is_file() and result['passed'] and result == json.loads(path.read_text())))
    return dict(scope=__doc__, passed=bool(checks) and all(c['passed'] for c in checks), checks=checks,
                evidence_sha256=hashes, comparator_sha256=sha(Path(inspect.getsourcefile(compare))), verifier_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run/'scientific-recheck.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'scientific-recheck.json').write_text(json.dumps(result, indent=2)+'\n')
    print(len(result['checks']), 'scientific comparisons reproduced; passed=', result['passed'])
    if not result['passed']:
        print([c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
