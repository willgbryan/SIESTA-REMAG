#!/usr/bin/env python3
"""Qualify omitted energy-density copies and requested TSDE/NetCDF outputs.

Require all nine export variants (three spin layouts x three output choices),
the full material/MAE audit, terminal copy accounting and a NetCDF-enabled
build. Both-format cases additionally require exact agreement between the
two serializations of each run's own DM/EDM, sparsity and Fermi value.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile

import numpy as np

from siesta_remag.corpus import ENERGY_EXPORT_CASES
from siesta_remag.energy_density_output import compare_outputs, read_tsde, read_netcdf
from siesta_remag.terminal_exports import requested_energy_exports


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(root):
    errors, checks, hashes = [], [], {}
    for name in ('source-verification.json', 'scientific-recheck.json', 'terminal-export-verification.json'):
        p = root/name
        if not p.is_file() or read(p).get('passed') is not True:
            errors.append('Missing/failed prerequisite: '+name)
        if p.is_file():
            hashes[name] = sha(p)
    meta, ledger = read(root/'metadata.json'), read(root/'ledger.json')
    if not set(ENERGY_EXPORT_CASES) <= set(meta['cases']):
        errors.append('Requested export coverage incomplete')
    for role in ('reference', 'candidate'):
        cache = meta.get('build_cache', {}).get(role, '')
        for feature in ('NETCDF', 'NCDF'):
            if 'SIESTA_WITH_'+feature+':BOOL=ON' not in cache:
                errors.append(role+' build does not establish '+feature+' support')
    with tarfile.open(root/'implementation.tar.gz') as archive:
        for filename, fn in (('energy_density_output.py', read_tsde),
                             ('terminal_exports.py', requested_energy_exports)):
            if hashlib.sha256(archive.extractfile('src/siesta_remag/'+filename).read()).hexdigest() != sha(Path(inspect.getsourcefile(fn))):
                raise ValueError('Output reader/validator differs from executed archive')
    seen = set()
    masks = set()
    removed = 0
    for row in ledger:
        case = row['benchmark']
        candidate = root/row.get('candidate_path', case+'/candidate')
        reference = root/row.get('reference_path', case+'/reference')
        case = candidate.parent.name
        result = read(candidate/'result.json')
        requests = result['benchmark']['energy_density_outputs']
        expected = ENERGY_EXPORT_CASES.get(case, (case, False, False))[1:]
        issues = []
        if requests != dict(tsde=expected[0], netcdf=expected[1]):
            issues.append('Input output request differs from registered variant')
        reason = requested_energy_exports((candidate/'input.fdf').read_text(), True)
        if reason != 2*expected[0]+4*expected[1]:
            issues.append('Raw FDF differs from requested export policy')
        matrices = compare_outputs(reference, candidate, requests)
        if not all(c['passed'] for c in matrices.values()):
            issues.append('Raw requested matrix comparison failed')
        if any(row['checks'].get(k) != v for k, v in matrices.items()):
            issues.append('Stored output comparison not reproduced')
        cross = []
        evidence = []
        for folder in (reference, candidate):
            evidence.extend(folder/name for name in ('input.fdf', 'manifest.json', 'result.json'))
            evidence.extend(p for p in (folder/'remag.TSDE', folder/'remag.nc') if p.is_file())
            if all(expected):
                a, b = read_tsde(folder/'remag.TSDE'), read_netcdf(folder/'remag.nc')
                for key in ('n_col', 'list_col', 'DM', 'EDM', 'Ef'):
                    equal = np.array_equal(np.asarray(a[key]).reshape(-1), np.asarray(b[key]).reshape(-1))
                    cross.append(dict(role=folder.name, variable=key, passed=bool(equal)))
                    if not equal:
                        issues.append('TSDE and NetCDF serializations differ: '+key)
        cycles = read(candidate/'remag-device-scf.json')['cycles']
        evidence.append(candidate/'remag-device-scf.json')
        for cycle in cycles:
            terminal = cycle['terminal']
            if terminal['energy_export_reasons'] != reason:
                issues.append('Native export reason differs')
            e = 8*terminal['capacity']*terminal['energy_spins']
            if terminal['matrix_bytes'][4] != (e if reason else 0):
                issues.append('Energy-density copy differs from actual request')
            if not reason:
                removed += e
        masks.add(reason)
        seen.add(case)
        checks.append(dict(case=case, role=candidate.name, passed=not issues, errors=issues,
                           reason_mask=reason, matrices=matrices, cross_format=cross,
                           evidence_sha256={str(p.relative_to(root)): sha(p) for p in evidence}))
    if seen != set(meta['cases']) or masks != {0, 2, 4, 6}:
        errors.append('Output/copy policy coverage incomplete')
    return dict(scope=__doc__, passed=bool(checks) and not errors and all(c['passed'] for c in checks),
                errors=errors, checks=checks, removed_unrequested_energy_density_bytes=removed,
                prerequisite_sha256=hashes, verifier_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run/'energy-density-output-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'energy-density-output-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Requested energy-density outputs:', len(result['checks']), 'passed=', result['passed'])
    if not result['passed']:
        print(result['errors'], [c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
