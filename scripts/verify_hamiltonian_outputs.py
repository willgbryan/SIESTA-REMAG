#!/usr/bin/env python3
"""Compare requested HSX outputs with pristine upstream, including every matrix entry.

Additional fixed artifact gates: H <= 1e-6 Ry, S <= 1e-6, Fermi <= 1e-6 Ry,
coordinates <= 1e-3 Bohr and cell <= 1e-8 Bohr. Original scientific gates
remain mandatory. These are regression tolerances, not physical convergence.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile

import numpy as np

from siesta_remag.hsx import read_hsx
from siesta_remag.density_history import settings, boolean


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(root):
    errors, checks, hashes = [], [], {}
    for name in ('source-verification.json', 'scientific-recheck.json'):
        p = root/name
        if not p.is_file() or read(p).get('passed') is not True:
            errors.append('Missing/failed prerequisite: '+name)
        if p.is_file():
            hashes[name] = sha(p)
    reader = Path(inspect.getsourcefile(read_hsx))
    policy = Path(inspect.getsourcefile(settings))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        archived = archive.extractfile('src/siesta_remag/hsx.py').read()
        archived_policy = archive.extractfile('src/siesta_remag/density_history.py').read()
    if hashlib.sha256(archived).hexdigest() != sha(reader):
        raise ValueError('HSX reader differs from executed archive')
    if hashlib.sha256(archived_policy).hexdigest() != sha(policy):
        raise ValueError('HSX request reader differs from executed archive')
    ledger = read(root/'ledger.json')
    for row in ledger:
        case = row['benchmark']
        files = [root/row.get('reference_path', case+'/reference')/'remag.HSX',
                 root/row.get('candidate_path', case+'/candidate')/'remag.HSX']
        issues, observed = [], {}
        try:
            requests=[]
            for path in files:
                values=settings((path.parent/'input.fdf').read_text())
                wanted=boolean(values,'savehs',True) or boolean(values,'coopwrite',False) or boolean(values,'writehshistory',False)
                manifest=read(path.parent/'result.json')['benchmark']
                if manifest.get('hamiltonian_output',wanted) is not wanted:
                    issues.append('HSX request differs from input provenance')
                requests.append(wanted)
            if requests[0]!=requests[1]:
                issues.append('Reference and candidate HSX requests differ')
            if not any(requests):
                if any(list(path.parent.glob('*.HSX')) for path in files):
                    issues.append('Unrequested HSX artifact was written')
                if row.get('passed') is not True:
                    issues.append('Original scientific comparison failed')
                evidence=[path.parent/name for path in files for name in ('input.fdf','result.json','remag-device-scf.json')
                          if (path.parent/name).is_file()]
                checks.append(dict(case=case,requested=False,passed=not issues,errors=issues,comparisons={},
                    evidence_sha256={str(p.relative_to(root)):sha(p) for p in evidence}))
                continue
            a, b = map(read_hsx, files)
            for key in ('header', 'offsets', 'labels', 'counts', 'columns'):
                if not np.array_equal(a[key], b[key]):
                    issues.append('HSX geometry/basis/sparsity metadata differs: '+key)
            for key in ('precision', 'basis', 'kpoints', 'charge', 'temperature_ry'):
                if a[key] != b[key]:
                    issues.append('HSX physical settings differ: '+key)
            for key, tolerance in (('h', 1e-6), ('overlap', 1e-6), ('fermi_ry', 1e-6),
                                   ('positions', 1e-3), ('cell', 1e-8)):
                left, right = np.asarray(a[key]), np.asarray(b[key])
                if left.shape != right.shape:
                    issues.append('HSX array shape differs: '+key)
                    continue
                error = float(np.max(np.abs(left-right), initial=0))
                observed[key] = dict(max_absolute_error=error, tolerance=tolerance, values=int(left.size))
                if not np.isfinite(error) or error > tolerance:
                    issues.append('HSX numerical comparison failed: '+key)
            # Bind the matrix Fermi value and spin/orbital bank to the actual
            # record dimensions as well as to the pristine requested output.
            terminal = read(files[1].parent/'remag-device-scf.json')['cycles'][-1]['terminal']
            if int(b['header'][2]) != terminal['spins'] or b['h'].shape[1] > terminal['capacity']:
                issues.append('HSX output does not fit the exported matrix state')
            if row.get('passed') is not True:
                issues.append('Original scientific comparison failed')
        except (OSError, ValueError, KeyError, IndexError) as exc:
            issues.append('HSX evidence could not be validated: '+str(exc))
        evidence=files+[path.parent/name for path in files for name in ('input.fdf','result.json')]
        checks.append(dict(case=case, requested=True, passed=not issues, errors=issues, comparisons=observed,
                           evidence_sha256={str(p.relative_to(root)): sha(p) for p in evidence if p.is_file()}))
    return dict(scope=__doc__, passed=bool(checks) and not errors and all(c['passed'] for c in checks),
                errors=errors, checks=checks, prerequisite_sha256=hashes,
                reader_sha256=sha(reader), policy_sha256=sha(policy), verifier_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run/'hamiltonian-output-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'hamiltonian-output-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Requested HSX outputs:', len(result['checks']), 'passed=', result['passed'])
    if not result['passed']:
        print(result['errors'], [c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
