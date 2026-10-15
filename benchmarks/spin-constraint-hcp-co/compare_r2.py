#!/usr/bin/env python3
"""Second comparison of hcp Co between REMAG-SIESTA and pw.x: off-site spin-orbit
coupling, the local-frame noncollinear GGA, torques, total energies and tetrahedra.

Reads reference/siesta-r2.json and reference/qe-r2.json (this round) and
reference/qe.json (the earlier pw.x runs at 2000 K). Usage:

    python compare_r2.py --check          # acceptance checks
    python compare_r2.py --figure r2.png  # the comparison figure
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RY_EV = 13.605693122994
UEV = 1e6


def load(name): return json.loads((HERE/'reference'/name).read_text())['runs']


def siesta(runs, **want):
    """Completed REMAG-SIESTA runs whose settings match want (missing keys take their defaults)."""
    defaults = dict(soc='onsite', occupation='FD', xc='PBE', nc_gga=None, pao_block=None, basis='TZP',
                    torque_tolerance_muB=1.e-3)
    out = []
    for r in runs:
        s = {k: r.get(k, defaults.get(k)) for k in want}
        if s == want and r.get('completed', True): out.append(r)
    return out


def qe(runs, **want):
    defaults = dict(occupation='smearing', dft=None, lambda_ry=25., ecutwfc_ry=100.)
    return [r for r in runs if all(r.get(k, defaults.get(k)) == v for k, v in want.items()) and r['converged']]


def curve(rows, energy):
    """E(theta) - E(0) in micro-eV per cell, as sorted (theta, value) arrays; energy(row) in eV."""
    e = {r['theta_deg']: energy(r) for r in rows}
    if 0. not in e: return None
    th = np.array(sorted(e)); return th, np.array([(e[t]-e[0.])*UEV for t in th])


def fit(th, E):
    """E = K1 sin^2 + K2 sin^4; returns (K1, K2), the rms residual and dE/dtheta as a function."""
    t = np.radians(th); A = np.stack([np.sin(t)**2, np.sin(t)**4], 1)
    c, *_ = np.linalg.lstsq(A, E, rcond=None)
    slope = lambda x: c[0]*np.sin(2*np.radians(x)) + 4*c[1]*np.sin(np.radians(x))**3*np.cos(np.radians(x))
    return c, float(np.sqrt(np.mean((E-A@c)**2))), slope


def torque(r):
    """Holding torque -(b.e_theta)|M| in micro-eV/rad, which equals dF/dtheta when the SCF is variational."""
    th = math.radians(r['theta_deg']); e = (math.cos(th), 0., -math.sin(th))
    return -sum(b*x for b, x in zip(r['field_ev_per_muB'], e))*r['moment_magnitude_muB']*UEV


def summary():
    s, q, q1 = load('siesta-r2.json'), load('qe-r2.json'), load('qe.json')
    best = dict(soc='offsite', nc_gga='local-frame', energy_shift_ry=0.005, mesh_cutoff_ry=1200, occupation='FD',
                temperature_k=2000., kmesh=[20, 20, 11])
    out = dict(curves={}, torques={}, basis=[], tetrahedra={})
    for xc, dft in (('PBE', None), ('PW92', 'pw')):
        rows = siesta(s, xc=xc, **best)
        qrows = qe(q, dft=dft, temperature_k=2000., kmesh=[20, 20, 11], occupation='smearing')
        if xc == 'PBE':
            qrows += [r for r in q1 if r['temperature_k'] == 2000. and r['kmesh'] == [20, 20, 11]
                      and r['lambda_ry'] == 25.]
        # Runs held to 1e-6 muB (the torque runs) and to 1e-3 muB give the same energies; prefer the former.
        rows_e = sorted(rows, key=lambda r: -r['torque_tolerance_muB'])
        cs = curve(rows_e, lambda r: r['free_energy_ev'])
        cq = curve(qrows, lambda r: r['free_energy_ry']*RY_EV)
        out['curves'][xc] = dict(siesta=cs, qe=cq, siesta_fit=fit(*cs)[:2], qe_fit=fit(*cq)[:2])
        slope_q = fit(*cq)[2]
        tight = sorted((r for r in rows if r['torque_tolerance_muB'] == 1.e-6), key=lambda r: r['theta_deg'])
        out['torques'][xc] = [(r['theta_deg'], torque(r), float(slope_q(r['theta_deg']))) for r in tight]
    # libgridxc's own noncollinear GGA (legacy): at 20x20x11 only with the default basis (drawn),
    # and at 16x16x9 with exactly the local-frame run's settings (checked).
    leg = siesta(s, soc='offsite', nc_gga=None, energy_shift_ry=0.02, mesh_cutoff_ry=600, temperature_k=2000.,
                 kmesh=[20, 20, 11], xc='PBE', occupation='FD')
    out['legacy'] = curve(leg, lambda r: r['free_energy_ev'])
    pair = dict(soc='offsite', energy_shift_ry=0.005, mesh_cutoff_ry=1200, temperature_k=2000., kmesh=[16, 16, 9],
                xc='PBE', occupation='FD', basis='TZP', pao_block=None)
    out['legacy_pair'] = [curve(siesta(s, nc_gga=g, **pair), lambda r: r['free_energy_ev'])[1][-1]
                          for g in (None, 'local-frame')]
    # Basis ladder at 16x16x9: anisotropy against the absolute free energy at theta = 0.
    qref = [r for r in q1 if r['temperature_k'] == 2000. and r['kmesh'] == [16, 16, 9] and r['lambda_ry'] == 25.]
    e0q = {r['theta_deg']: r['free_energy_ry']*RY_EV for r in qref}
    out['qe_k16'] = dict(mae=(e0q[90.]-e0q[0.])*UEV, f0=e0q[0.])
    for label, sel in (('TZP 0.005 Ry', dict(basis='TZP', energy_shift_ry=0.005, pao_block=None)),
                       ('TZDP 0.005 Ry', dict(basis='TZDP', energy_shift_ry=0.005, pao_block=None)),
                       ('TZP 0.002 Ry', dict(basis='TZP', energy_shift_ry=0.002, pao_block=None)),
                       ('semicore DZ 0.002 Ry', dict(pao_block='semicore-dz', energy_shift_ry=0.002)),
                       ('rich 0.002 Ry', dict(pao_block='rich', energy_shift_ry=0.002))):
        rows = siesta(s, soc='offsite', nc_gga='local-frame', mesh_cutoff_ry=1200, temperature_k=2000.,
                      kmesh=[16, 16, 9], xc='PBE', occupation='FD', **sel)
        e = {r['theta_deg']: r['free_energy_ev'] for r in rows}
        if 0. in e and 90. in e:
            out['basis'].append(dict(label=label, mae=(e[90.]-e[0.])*UEV, gap_ev=e[0.]-e0q[0.]))
    for xc, dft in (('PBE', None), ('PW92', 'pw')):
        series = {}
        for r in qe(q, dft=dft, occupation='tetrahedra_opt'):
            series.setdefault(('qe', r['kmesh'][0]), {})[r['theta_deg']] = r['free_energy_ry']*RY_EV
        for r in siesta(s, xc=xc, soc='offsite', nc_gga='local-frame', occupation='Tetrahedra.Optimized',
                        energy_shift_ry=0.005, mesh_cutoff_ry=1200):
            series.setdefault(('siesta', r['kmesh'][0]), {})[r['theta_deg']] = r['free_energy_ev']
        out['tetrahedra'][xc] = sorted((code, n, (e[90.]-e[0.])*UEV) for (code, n), e in series.items()
                                       if 0. in e and 90. in e)
    return out


def checks(result):
    """Acceptance checks, chosen after the runs were seen; they guard against regressions."""
    s = load('siesta-r2.json')
    out = []
    for xc in ('PBE', 'PW92'):
        k_s, k_q = result['curves'][xc]['siesta_fit'][0][0], result['curves'][xc]['qe_fit'][0][0]
        out.append((f'{xc}: SIESTA K1 within 15% of QE at 2000 K, 20x20x11', 0.85 < k_s/k_q < 1.15,
                    f'{k_s:.1f} vs {k_q:.1f} micro-eV'))
        ratios = [t/qq for _, t, qq in result['torques'][xc]]
        out.append((f'{xc}: field torques within 15% of QE dE/dtheta at all angles', bool(ratios) and
                    all(0.85 < x < 1.15 for x in ratios), ' '.join(f'{x:.3f}' for x in ratios)))
    leg, lf = result['legacy_pair']
    out.append(('legacy GGA anisotropy at least 1.8 times local-frame, same settings at 16x16x9', leg/lf > 1.8,
                f'{leg:.1f} vs {lf:.1f} micro-eV'))
    gaps = [b['gap_ev'] for b in result['basis']]
    out.append(('basis ladder lies above QE in energy', all(g > 0 for g in gaps), ' '.join(f'{g:.3f}' for g in gaps)))
    tq = siesta(s, soc='offsite', nc_gga='local-frame', occupation='FD', temperature_k=500., kmesh=[6, 6, 3],
                torque_tolerance_muB=1.e-6)
    by = {r['theta_deg']: r for r in tq if r.get('h_tolerance_ev') == 1.e-5}
    if {40., 45., 50.} <= set(by):
        slope = (by[50.]['free_energy_ev']-by[40.]['free_energy_ev'])/math.radians(10.)*UEV
        ratio = torque(by[45.])/slope
        out.append(('local-frame PBE torque identity within 0.5% of the finite-difference factor 1.005',
                    abs(ratio-1.0051) < 0.005, f'{ratio:.4f}'))
    return out


def figure(result, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for xc, colour in (('PBE', 'C0'), ('PW92', 'C1')):
        name = 'PBE' if xc == 'PBE' else 'LDA'
        c = result['curves'][xc]
        ax[0].plot(*c['qe'], 'o', color=colour, mfc='none', ms=8, label=f'QE, {name}')
        ax[0].plot(*c['siesta'], 's-', color=colour, ms=4, label=f'REMAG-SIESTA, {name}')
        th = np.array([t for t, _, _ in result['torques'][xc]])
        ax[1].plot(th, [qq for _, _, qq in result['torques'][xc]], 'o', color=colour, mfc='none', ms=8,
                   label=f'QE dE/dθ, {name}')
        ax[1].plot(th, [t for _, t, _ in result['torques'][xc]], 's-', color=colour, ms=4,
                   label=f'REMAG-SIESTA holding torque, {name}')
    leg = result['legacy']
    ax[0].plot(*leg, 'x', color='0.4', ms=8, label='REMAG-SIESTA, PBE, upstream GGA potential, default basis')
    ax[0].set_xlabel('angle of the moment from the c axis (degrees)')
    ax[0].set_ylabel('E(θ) − E(0) per two-atom cell (μeV)')
    ax[0].set_title('Energy of turning the moment, 2000 K, 20×20×11', fontsize=10)
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel('angle of the moment from the c axis (degrees)')
    ax[1].set_ylabel('dE/dθ per cell (μeV per radian)')
    ax[1].set_title('Torque: QE slope against SIESTA field', fontsize=10)
    ax[1].legend(fontsize=7)
    b = result['basis']
    ax[2].plot([x['gap_ev'] for x in b], [x['mae'] for x in b], 's', color='C0', label='REMAG-SIESTA, PBE')
    for x in b: ax[2].annotate(x['label'], (x['gap_ev'], x['mae']), fontsize=7, xytext=(4, 4), textcoords='offset points')
    ax[2].axhline(result['qe_k16']['mae'], color='C0', ls=':', label='QE, PBE')
    ax[2].plot([0.], [result['qe_k16']['mae']], 'o', color='C0', mfc='none', ms=8)
    ax[2].set_xlabel('SIESTA free energy above QE at θ = 0 (eV per cell)')
    ax[2].set_ylabel('E(90°) − E(0) per cell (μeV)')
    ax[2].set_title('Basis sets, PBE, 2000 K, 16×16×9', fontsize=10)
    ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true'); p.add_argument('--figure', type=Path)
    a = p.parse_args()
    result = summary()
    for xc in ('PBE', 'PW92'):
        c = result['curves'][xc]
        print(xc, 'SIESTA', np.round(c['siesta'][1], 2), 'QE', np.round(c['qe'][1], 2),
              'K1,K2 SIESTA', np.round(c['siesta_fit'][0], 2), 'QE', np.round(c['qe_fit'][0], 2))
        print('   torques (theta, SIESTA, QE):', [(t, round(x, 2), round(y, 2)) for t, x, y in result['torques'][xc]])
        print('   tetrahedra (code, n, E90-E0):', [(c_, n, round(v, 1)) for c_, n, v in result['tetrahedra'][xc]])
    print('basis:', [(b['label'], round(b['mae'], 2), round(b['gap_ev'], 3)) for b in result['basis']], 'QE k16', result['qe_k16'])
    if a.figure: figure(result, a.figure)
    if a.check:
        ok = True
        for name, passed, detail in checks(result):
            print(('PASS' if passed else 'FAIL'), name, '|', detail); ok &= passed
        raise SystemExit(0 if ok else 1)
