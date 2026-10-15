#!/usr/bin/env python3
"""Collect pw.x results for the hcp Co E(theta) reference into one JSON file.

Each run directory holds pw.in, pw.out and manifest.json (written by
make_qe.py). The record keeps only results: the Mermin free energy F = E - TS
printed after "!" (pw.x leaves the constraint energy out of it), the last
constraint energy, the sphere moments and their polar angles, the reduced
k-point count, convergence and wall time. Usage:

    python extract_qe.py <root> [<root> ...] --output reference/qe.json
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

RY_EV = 13.605693122994


def last(pattern, text, cast=float):
    found = re.findall(pattern, text, re.M)
    return cast(found[-1]) if found else None


def wall(out):
    """PWSCF wall time; pw.x prints 16.44s, 1m23.45s or 1h 2m."""
    line = re.findall(r'PWSCF\s+:.*CPU\s+(.*) WALL', out)
    if not line: return None
    units = dict(h=3600., m=60., s=1.)
    return sum(float(v)*units[u] for v, u in re.findall(r'([\d.]+)\s*([hms])', line[-1]))


def parse(directory):
    out = (directory/'pw.out').read_text(errors='replace')
    manifest = json.loads((directory/'manifest.json').read_text())
    sites = re.findall(r'polar coord\.: r, theta, phi \[deg\] :\s+(\S+)\s+(\S+)\s+(\S+)', out)
    nat = 2
    record = dict(manifest)
    record.update(
        directory=str(directory),
        input_sha256=hashlib.sha256((directory/'pw.in').read_bytes()).hexdigest(),
        converged='convergence has been achieved' in out,
        iterations=last(r'convergence has been achieved in\s+(\d+) iterations', out, int),
        k_points=last(r'number of k points=\s+(\d+)', out, int),
        free_energy_ry=last(r'^!\s+total energy\s+=\s+(\S+) Ry', out),
        smearing_ts_ry=last(r'smearing contrib\. \(-TS\)\s+=\s+(\S+) Ry', out),
        constraint_energy_ry=last(r'constraint energy \(Ry\) =\s+(\S+)', out),
        fermi_energy_ev=last(r'the Fermi energy is\s+(\S+) ev', out),
        total_magnetization=[float(x) for x in re.findall(r'total magnetization\s+=\s+(\S+)\s+(\S+)\s+(\S+)', out)[-1]],
        site_moments=[dict(r=float(r), theta_deg=float(t), phi_deg=float(p)) for r, t, p in sites[-nat:]],
        wall_seconds=wall(out))
    record['max_angle_error_deg'] = max(abs(s['theta_deg']-manifest['theta_deg']) for s in record['site_moments']) \
        if record['site_moments'] else None
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('roots', nargs='+', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    runs = [parse(d.parent) for root in a.roots for d in sorted(root.rglob('pw.out'))]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(code='Quantum ESPRESSO pw.x', runs=runs), indent=1)+'\n')
    print(f'{len(runs)} runs -> {a.output}')
