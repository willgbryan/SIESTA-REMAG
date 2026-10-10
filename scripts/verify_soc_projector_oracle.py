#!/usr/bin/env python3
"""Audit pristine SOC fixtures with an independent angular-operator contraction."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

FIELDS = ('epsilon', 'left', 'right', 'left_gradient', 'right_gradient')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def angular_operators(l):
    """Build both j projectors as matrices in the real harmonic basis."""
    width = 2*l+1
    transform = np.zeros((width, width), complex)
    transform[l, l] = 1
    for m in range(1, l+1):
        transform[l-m, l+m] = 1/np.sqrt(2.)
        transform[l-m, l-m] = 1j/np.sqrt(2.)
        transform[l+m, l+m] = (-1.)**m/np.sqrt(2.)
        transform[l+m, l-m] = -1j*(-1.)**m/np.sqrt(2.)
    operators = []
    for branch, twice_j in enumerate((2*l-1, 2*l+1)):
        coefficients = np.zeros((twice_j+1, 2, width), complex)
        for state, twice_mj in enumerate(range(-twice_j, twice_j+1, 2)):
            mj = twice_mj/2
            plus = np.sqrt((l+.5+mj)/width)
            minus = np.sqrt((l+.5-mj)/width)
            cg = (-minus, plus) if branch == 0 else (plus, minus)
            for spin, m in enumerate(((twice_mj-1)//2, (twice_mj+1)//2)):
                if abs(m) <= l:
                    coefficients[state, spin] = cg[spin]*transform[l+m]
        operators.append(np.einsum('saM,sbK->abMK', coefficients, coefficients.conj()))
    return np.array(operators)


def predict(l, split, data):
    left, right, gradient = (data[k] for k in ('left', 'right', 'left_gradient'))
    epsilon = data['epsilon']
    operators = angular_operators(l)
    n = len(left)
    v = np.zeros((n, 2, 2), complex)
    f = np.zeros((n, 2, 2, 3), complex)
    for j in range(2):
        v += epsilon[j]*np.einsum('nM,abMK,nK->nab', left[:, j], operators[j], right[:, j], optimize=True)
        f += epsilon[j]*np.einsum('nMc,abMK,nK->nabc', gradient[:, j], operators[j], right[:, j], optimize=True)
    ion = np.zeros(n)
    if split:
        product = epsilon[0]*epsilon[1]
        mixed = np.sqrt(abs(product))*(np.sign(epsilon[0]) if product > 0 else -1)
        radial = np.array([[l*l*epsilon[0], l*(l+1)*mixed],
                           [l*(l+1)*mixed, (l+1)**2*epsilon[1]]])/(2*l+1)**2
        ion = np.einsum('njM,jk,nkM->n', left, radial, right, optimize=True)
        v[:, 0, 0] -= ion
        v[:, 1, 1] -= ion
    vf = v.transpose(0, 2, 1).reshape(n, 4)
    ff = f.transpose(0, 2, 1, 3).reshape(n, 12)
    return np.column_stack((ion, vf.real, vf.imag, ff.real, ff.imag))


def verify(path, upstream):
    report = json.loads(path.read_text())
    folder = path.parent
    archive = folder/report['arrays_file']
    errors, checks, seen = [], [], set()
    source = upstream/'Src/nlefsm.f'
    original = source.read_text()
    begin = original.index('      subroutine calc_Vj_offsiteSO(')
    end = original.index('      end subroutine calc_Vj_offsiteSO', begin)+len('      end subroutine calc_Vj_offsiteSO')
    literal = original[begin:end]+'\n'
    wrapper = folder/report['wrapper_file']
    if sha(source) != report['source_sha256'] or hashlib.sha256(literal.encode()).hexdigest() != report['literal_block_sha256'] or literal not in wrapper.read_text():
        errors.append('Literal source differs from pinned upstream')
    if sha(archive) != report['arrays_sha256'] or sha(wrapper) != report['wrapper_sha256']:
        errors.append('Retained content hash differs')
    signs = ((1.3, .8), (-.4, -.9), (.7, -1.1), (-.6, .9), (0., .7), (0., 0.))
    with np.load(archive) as arrays:
        for row in report['checks']:
            name = row['fixture']
            config = arrays[name+'_config']
            l, split, n = map(int, config)
            data = {k: arrays[name+'_'+k] for k in FIELDS}
            actual = arrays[name+'_expected']
            issues = []
            signature = (l, tuple(data['epsilon']), split, n)
            if signature in seen:
                issues.append('Duplicate fixture')
            seen.add(signature)
            raw = config.tobytes()+b''.join(data[k].tobytes() for k in FIELDS)
            if hashlib.sha256(raw).hexdigest() != row['input_sha256']:
                issues.append('Input bytes do not reproduce')
            maximum = 0.
            for start in range(0, n, 16384):
                end = min(n, start+16384)
                sliced = {k: v if k == 'epsilon' else v[start:end] for k, v in data.items()}
                wanted = predict(l, split, sliced)
                delta = np.abs(actual[start:end]-wanted)
                maximum = max(maximum, float(delta.max()))
                if not np.isfinite(actual[start:end]).all() or np.any(delta > 1e-12+1e-12*np.abs(wanted)):
                    issues.append('Independent angular contraction differs')
            # F is the derivative of the full nonlocal operator, including the
            # scalar piece even when the energies are reported separately.
            sample = {k: v if k == 'epsilon' else v[:8] for k, v in data.items()}
            full = predict(l, False, sample)
            split_value = predict(l, True, sample)
            reconstructed = split_value[:, 1:9].copy()
            reconstructed[:, 0] += split_value[:, 0]
            reconstructed[:, 3] += split_value[:, 0]
            if not np.allclose(full[:, 1:9], reconstructed, rtol=1e-13, atol=1e-13) or not np.array_equal(full[:, 9:], split_value[:, 9:]):
                issues.append('Scalar/SOC split changed full operator or force')
            fd_max = 0.
            h = 1e-5
            for axis in range(3):
                plus = {**sample, 'left': sample['left']+h*sample['left_gradient'][..., axis]}
                minus = {**sample, 'left': sample['left']-h*sample['left_gradient'][..., axis]}
                derivative = (predict(l, False, plus)[:, 1:9]-predict(l, False, minus)[:, 1:9])/(2*h)
                expected = np.column_stack((actual[:8, 9:21][:, axis::3], actual[:8, 21:33][:, axis::3]))
                fd_max = max(fd_max, float(np.abs(derivative-expected).max()))
            if fd_max > 2e-9:
                issues.append('Full-operator finite difference differs from pristine force')
            checks.append(dict(fixture=name, passed=not issues, errors=issues,
                               max_absolute_error=maximum, finite_difference_max_error=fd_max))
    expected = {(l, s, split, 17) for l in range(1, 6) for s in signs for split in (0, 1)}
    expected.update((l, signs[0], split, n) for l, n in ((2, 257), (1, 131073)) for split in (0, 1))
    if seen != expected or len(checks) != 64:
        errors.append('Fixture coverage differs')
    return dict(scope=__doc__, passed=not errors and all(c['passed'] for c in checks), errors=errors,
                checks=checks, report_sha256=sha(path), verifier_sha256=sha(Path(__file__)), gpu_validation=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--upstream', type=Path, default=Path('upstream/siesta'))
    args = parser.parse_args()
    result = verify(args.reference, args.upstream)
    args.reference.with_name('soc-projector-oracle-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('SOC projector oracle audit:', len(result['checks']), 'passed=', result['passed'])
    if not result['passed']:
        raise SystemExit(1)
