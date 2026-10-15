#!/usr/bin/env python3
"""Collect finished tetrahedron runs of pw.x into one JSON record per run.

Reads tmp/run.save/data-file-schema.xml, which holds every k point of the
full grid with its eigenvalues and occupations at full precision, the Fermi
energy, the electron count and the cell. Energies stay in Hartree as pw.x
writes them; k points are converted to Cartesian 1/Bohr. pw.out supplies the
total energy and convergence. Usage:

    python extract_qe.py <root> --output reference/qe-tetrahedra.json
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def numbers(element): return [float(x) for x in element.text.split()]


def parse(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    out = (directory/'pw.out').read_text(errors='replace')
    root = ET.parse(directory/'tmp/run.save/data-file-schema.xml').getroot()
    o = root.find('output')
    structure = o.find('atomic_structure'); alat = float(structure.get('alat'))
    cell = [numbers(structure.find(f'cell/a{i}')) for i in (1, 2, 3)]
    bands = o.find('band_structure')
    flag = lambda name: (bands.findtext(name) or 'false').strip() == 'true'
    lsda, noncolin = flag('lsda'), flag('noncolin')
    k, weights, energies, occupations = [], [], [], []
    for item in bands.findall('ks_energies'):
        point = item.find('k_point')
        k.append([2*np.pi/alat*x for x in numbers(point)]); weights.append(float(point.get('weight')))
        energies.append(numbers(item.find('eigenvalues'))); occupations.append(numbers(item.find('occupations')))
    energy = re.findall(r'^!\s+total energy\s+=\s+(\S+) Ry', out, re.M)
    record = dict(manifest,
        directory=str(directory),
        input_sha256=hashlib.sha256((directory/'pw.in').read_bytes()).hexdigest(),
        occupations_kind=bands.findtext('occupations_kind', '').strip() or None,
        converged='convergence has been achieved' in out,
        total_energy_ry=float(energy[-1]) if energy else None,
        lsda=lsda, noncolin=noncolin, spinorbit=flag('spinorbit'),
        nelec=float(bands.findtext('nelec')),
        fermi_energy_ha=float(bands.findtext('fermi_energy')),
        cell_bohr=cell, k_cartesian_per_bohr=k, k_weights=weights,
        eigenvalues_ha=energies, occupations=occupations)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('roots', nargs='+', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    runs = [parse(m.parent) for root in a.roots for m in sorted(root.rglob('manifest.json'))
            if (m.parent/'tmp/run.save/data-file-schema.xml').is_file()]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(
        source='pw.x 7.3.1 (NVIDIA HPC container nvcr.io/hpc/quantum_espresso:qe-7.3.1), one H100',
        energy_unit='Hartree', runs=runs), separators=(',', ':'))+'\n')
    print(len(runs), 'runs ->', a.output)
