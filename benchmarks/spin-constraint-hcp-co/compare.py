#!/usr/bin/env python3
"""Compare constrained E(theta) of hcp Co from REMAG-SIESTA with pw.x and experiment.

Energies are per two-atom cell, relative to the moment along the c axis
(theta = 0) at the same k mesh and settings. Usage:

    python compare.py [--siesta siesta.json] [--figure mae.png] [--check]

Without --siesta the retained REMAG-SIESTA reference (reference/siesta.json)
is used. --check applies the acceptance tests of README.md and exits nonzero
if any fails.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RY_EV = 13.605693122994
MICRO = 1.e6
# Settings that define one E(theta) curve. Runs that differ only in theta share a curve.
KEYS = ('kmesh', 'basis', 'mesh_cutoff_ry', 'temperature_k', 'energy_shift_ry', 'ecutwfc_ry', 'lambda_ry', 'xc', 'path')


def load(path): return json.loads(Path(path).read_text())['runs']


def free_energy_ev(run):
    return run['free_energy_ev'] if 'free_energy_ev' in run else run['free_energy_ry']*RY_EV


def curves(runs):
    """[{settings, k_points_full, delta_uev: {theta: E(theta)-E(0)}}] for every setting with theta = 0."""
    groups = defaultdict(dict)
    for r in runs:
        if not r.get('converged'): continue
        key = tuple((k, tuple(r[k]) if isinstance(r[k], list) else r[k]) for k in KEYS if k in r)
        groups[key][r['theta_deg']] = free_energy_ev(r)
    out = []
    for key, energies in groups.items():
        if 0. not in energies: continue
        s = dict(key)
        out.append(dict(settings=s, k_points_full=s['kmesh'][0]*s['kmesh'][1]*s['kmesh'][2],
                        delta_uev={f'{t:g}': (e-energies[0.])*MICRO for t, e in sorted(energies.items()) if t != 0.}))
    return sorted(out, key=lambda c: (c['settings'].get('temperature_k', 0), c['k_points_full'],
                                      json.dumps(c['settings'], sort_keys=True)))


def series(result, code, temperature, **fixed):
    """{mesh N1: curve} for one code at one smearing temperature and fixed settings."""
    defaults = dict(siesta=dict(basis='TZP', mesh_cutoff_ry=600, energy_shift_ry=0.02),
                    qe=dict(ecutwfc_ry=100., lambda_ry=25.))[code]
    want = dict(defaults, temperature_k=float(temperature), **fixed)
    return {c['settings']['kmesh'][0]: c for c in result[code]
            if all(c['settings'].get(k) == v for k, v in want.items()) and 'path' not in c['settings']}


def summary(siesta_runs, qe_runs):
    return dict(siesta=curves(siesta_runs), qe=curves(qe_runs),
                experiment=json.loads((HERE/'reference/experiment.json').read_text()))


def basis_variants(result, temperature=2000., mesh=20):
    """SIESTA E(90)-E(0) at one k mesh for every basis and real-space mesh tried."""
    return sorted(((c['settings']['basis'], c['settings']['energy_shift_ry'], c['settings']['mesh_cutoff_ry']),
                   c['delta_uev']['90']) for c in result['siesta']
                  if c['settings']['kmesh'][0] == mesh and c['settings']['temperature_k'] == temperature
                  and '90' in c['delta_uev'] and 'path' not in c['settings'])


def checks(result, siesta_runs, retained_runs):
    """Acceptance checks of README.md, as (name, passed, detail). Thresholds were set after the retained
    runs were seen; they guard against regressions, they were not predictions."""
    out = []
    bad = [r['directory'] for r in siesta_runs if not (r.get('converged') and r.get('deviation_deg', 1.) <= 0.01)]
    out.append(('every constraint met within 0.01 degrees', not bad, bad[:5]))
    # A rerun of a retained input must give the retained energy: same code, same input bytes.
    retained = {r['input_sha256']: r for r in retained_runs}
    pairs = [(r, retained[r['input_sha256']]) for r in siesta_runs if r['input_sha256'] in retained]
    worst = max((abs(r['free_energy_ev']-q['free_energy_ev']) for r, q in pairs), default=None)
    out.append(('reruns reproduce retained free energies within 1e-5 eV', worst is not None and worst <= 1.e-5,
                dict(pairs=len(pairs), worst_ev=worst)))
    # Only at 2000 K are both codes converged in k within the 20x20x11 limit.
    s, q = series(result, 'siesta', 2000.).get(20), series(result, 'qe', 2000.).get(20)
    if s and q and '90' in s['delta_uev']:
        a, b = s['delta_uev']['90'], q['delta_uev']['90']
        out.append(('2000 K, 20x20x11: same easy direction as QE', a*b > 0, dict(siesta=a, qe=b)))
    if s and q and '45' in s['delta_uev']:
        rs, rq = (c['delta_uev']['45']/c['delta_uev']['90'] for c in (s, q))
        out.append(('2000 K, 20x20x11: E(45)/E(90) within 0.05 of QE', abs(rs-rq) <= 0.05, dict(siesta=rs, qe=rq)))
    for code in ('siesta', 'qe'):
        v = [c['delta_uev']['90'] for n, c in series(result, code, 2000.).items() if n in (16, 18, 20)]
        if len(v) == 3:
            out.append((f'2000 K: {code} E(90)-E(0) spread over meshes 16-20 within 15 micro-eV',
                        max(v)-min(v) <= 15., dict(values=v)))
    return out


def figure(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    temperatures = sorted({c['settings']['temperature_k'] for c in result['siesta'] if 'path' not in c['settings']})
    fig, axes = plt.subplots(1, len(temperatures)+1, figsize=(3.2*(len(temperatures)+1), 4.), sharey=True)
    e = result['experiment']['low_temperature']['per_cell_uev']
    for ax, t in zip(axes, temperatures):
        for code, style, label in (('siesta', 'o-', 'REMAG-SIESTA, TZP, 0.02 Ry, 600 Ry grid'), ('qe', 's--', 'Quantum ESPRESSO')):
            s = series(result, code, t)
            pts = sorted((n, c['delta_uev']['90']) for n, c in s.items() if '90' in c['delta_uev'] and n >= 12)
            if pts: ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=label)
        ax.axhline(e, color='0.5', lw=1, ls=':', label='experiment, low temperature')
        ax.axhline(0, color='0.8', lw=.5)
        ax.set_title(f'smearing {t:g} K', fontsize=9); ax.set_xlabel('k mesh N x N x M (M = 7, 8, 9, 10, 11)')
        ax.set_xticks([12, 14, 16, 18, 20])
    axes[0].set_ylabel('E(90 deg) - E(0 deg), micro-eV per 2-atom cell')
    axes[0].legend(frameon=False, fontsize=7, loc='lower left')
    # Last panel: SIESTA basis and real-space mesh choices at 20x20x11, 2000 K, against QE there.
    ax = axes[-1]
    qe = series(result, 'qe', 2000.).get(20)
    if qe: ax.axhline(qe['delta_uev']['90'], color='C1', ls='--', lw=1.5, label='Quantum ESPRESSO')
    markers = {600: 'o', 900: 's', 1200: '^'}
    for (basis, shift, cutoff), v in basis_variants(result):
        tzp = basis == 'TZP'
        ax.plot(shift*(1. if tzp else 1.25), v, markers.get(cutoff, 'x') if tzp else '*', color='C0' if tzp else 'C2', ms=6 if tzp else 11,
                mfc='C0' if tzp else 'none', label=f'REMAG-SIESTA {basis}, {cutoff} Ry grid')
    handles, labels = ax.get_legend_handles_labels(); unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=6, loc='upper right')
    ax.set_xscale('log'); ax.set_xlim(.03, .0014); ax.minorticks_off()
    ax.set_xticks([.02, .01, .005, .002]); ax.set_xticklabels(['0.02', '0.01', '0.005', '0.002'])
    ax.set_title('basis and grid at 20x20x11, smearing 2000 K', fontsize=9)
    ax.set_xlabel('orbital confinement energy (Ry)\nlonger orbitals to the right', fontsize=8)
    ax.axhline(0, color='0.8', lw=.5)
    fig.tight_layout(); fig.savefig(path, dpi=150)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--siesta', type=Path, default=HERE/'reference/siesta.json')
    p.add_argument('--qe', type=Path, default=HERE/'reference/qe.json')
    p.add_argument('--figure', type=Path); p.add_argument('--output', type=Path, help='write the curves and checks as JSON')
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    result = summary(load(a.siesta), load(a.qe))
    for code in ('siesta', 'qe'):
        for c in result[code]:
            s = c['settings']
            print(f"{code:6s} T={s.get('temperature_k'):6g} K mesh={s['kmesh']} "
                  + ' '.join(f'{k}={v}' for k, v in s.items() if k not in ('kmesh', 'temperature_k'))
                  + '  ' + ' '.join(f"{t}:{v:7.1f}" for t, v in c['delta_uev'].items()))
    if a.figure: figure(result, a.figure)
    if a.check:
        # A fresh run is checked together with the retained series, so convergence and
        # QE comparisons cover the meshes it did not repeat.
        retained = load(HERE/'reference/siesta.json')
        fresh = load(a.siesta)
        merged = summary(fresh+[r for r in retained if r['input_sha256'] not in {f['input_sha256'] for f in fresh}],
                         load(a.qe))
        result['checks'] = [dict(check=n, passed=bool(p), detail=d) for n, p, d in checks(merged, fresh, retained)]
        for c in result['checks']: print(('PASS ' if c['passed'] else 'FAIL ')+c['check'], c['detail'])
    if a.output: a.output.write_text(json.dumps(result, indent=1)+'\n')
    if a.check and not all(c['passed'] for c in result['checks']): raise SystemExit(1)
