#!/usr/bin/env python3
"""Collect REMAG-SIESTA constrained runs for the hcp Co E(theta) test into JSON.

Each run directory holds input.fdf, output.out, manifest.json (written by
make_siesta.py) and run.json. From the last constraint step the record keeps
the free energy F (which excludes the constraint terms), the site moment and
its angle from the target, the constraining field, the number of constraint
steps and SCF iterations, and the wall time. Usage:

    python extract_siesta.py <root> [<root> ...] --output siesta.json
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

NUMBER = r'[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?'


def parse(directory):
    out = (directory/'output.out').read_text(errors='replace')
    record = json.loads((directory/'manifest.json').read_text())
    steps = re.findall(rf'REMAG SPIN CONSTRAINT STEP\s+(\d+) line\s+(\d+) F\(eV\)=\s*({NUMBER}) L\(eV\)=\s*({NUMBER})'
                       rf' max_angle\(deg\)=\s*({NUMBER}) max_torque\(muB\)=\s*({NUMBER}) max_field\(eV\)=\s*({NUMBER})', out)
    sites = re.findall(rf'REMAG SPIN CONSTRAINT SITE\s+1\s+((?:\s*{NUMBER}){{10}})', out)
    run = directory/'run.json'
    record.update(directory=str(directory), input_sha256=hashlib.sha256((directory/'input.fdf').read_bytes()).hexdigest(),
                  converged=bool(re.search(r'REMAG SPIN CONSTRAINT CONVERGED', out)),
                  constraint_steps=len(steps), scf_iterations=len(re.findall(r'^\s+scf:\s+\d+', out, re.M)),
                  wall_seconds=json.loads(run.read_text())['seconds'] if run.exists() else None)
    m = re.search(r'siesta: k-grid: Number of k-points\s*=\s*(\d+)', out)
    record['k_points'] = int(m.group(1)) if m else None
    fdf = (directory/'input.fdf').read_text()
    t = re.search(r'REMAG.SpinConstraint.TorqueTolerance\s+(\S+)', fdf)
    record['torque_tolerance_muB'] = float(t.group(1)) if t else None
    t = re.search(r'SCF.H.Tolerance\s+(\S+)\s+eV', fdf)
    record['h_tolerance_ev'] = float(t.group(1)) if t else None
    record['completed'] = 'Job completed' in out
    for key, label in (('total_energy_ev', 'Etot'), ('final_free_energy_ev', 'FreeEng')):
        found = re.findall(rf'siesta: {label}\s+=\s+({NUMBER})', out)
        record[key] = float(found[-1]) if found and record['completed'] else None
    if steps:
        s = steps[-1]
        record.update(free_energy_ev=float(s[2]), lagrangian_ev=float(s[3]), max_angle_deg=float(s[4]),
                      max_torque_muB=float(s[5]), max_field_ev=float(s[6]))
    if sites:
        v = [float(x) for x in sites[-1].split()]
        record.update(moment_muB=v[2:5], moment_magnitude_muB=v[5], deviation_deg=v[6], field_ev_per_muB=v[7:10])
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('roots', nargs='+', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    runs = [parse(d.parent) for root in a.roots for d in sorted(root.rglob('output.out'))
            if (d.parent/'manifest.json').is_file()]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(code='REMAG-SIESTA', runs=runs), indent=1)+'\n')
    print(f'{len(runs)} runs -> {a.output}')
