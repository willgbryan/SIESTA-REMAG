#!/usr/bin/env python3
"""Qualify a pristine local relaxation as a reference for GPU density history."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from siesta_remag.output import parse_output, read_geometry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder):
    root = Path(__file__).resolve().parents[1]
    record = json.loads((folder/'preflight.json').read_text())
    manifest = json.loads((folder/'manifest.json').read_text())
    pin = json.loads((root/'upstream.lock.json').read_text())
    provenance = record['provenance']
    assert provenance['source_commit'] == pin['commit']
    assert provenance['binary_sha256'] == sha(root/'build/upstream/Src/siesta')
    assert provenance['input_sha256'] == manifest['input_sha256'] == sha(folder/'input.fdf')
    for name, digest in record['evidence_sha256'].items():
        assert sha(folder/name) == digest, name
    text = (folder/'stdout.txt').read_text()
    parsed = parse_output(text)
    assert record['exit_code'] == 0 and parsed['completed'] and parsed['converged']
    steps = [int(v) for v in re.findall(r'^new_DM -- step:\s*(\d+)', text, re.M)]
    completed = [int(v) for v in re.findall(r'^SCF cycle converged after (\d+) iterations', text, re.M)]
    assert len(steps) >= 4 and len(steps) == len(completed) and len(set(steps)) == len(steps)
    assert sum(completed) == parsed['scf_iterations']
    counts = [int(v) for v in re.findall(r'Number of DMs in history:\s*(\d+)', text)]
    depth = manifest['density_history_depth']
    assert depth > 1 and counts == [min(i, depth) for i in range(1, len(steps))]
    blocks = re.findall(r'DM extrapolation coefficients:[ \t]*\n((?:[ \t]*\d+[ \t]+[-+\d.]+[ \t]*\n)+)', text)
    coefficients = [[float(line.split()[1]) for line in block.splitlines()] for block in blocks]
    assert [len(c) for c in coefficients] == counts
    assert all(np.isfinite(c).all() for c in coefficients)
    nontrivial = sum(len(c) == depth and sum(abs(v) > 1e-4 for v in c) > 1 for c in coefficients)
    assert nontrivial > 0
    forces = np.asarray(parsed['forces_ev_ang'])
    assert forces.shape == (manifest['atoms'], 3) and np.isfinite(forces).all()
    maximum = float(np.max(np.linalg.norm(forces, axis=1)))
    tolerance = manifest['relaxation_force_tolerance_ev_ang']
    assert tolerance == .01 and maximum <= tolerance
    geometry = read_geometry(folder/'remag.XV')
    assert geometry is not None
    return dict(passed=True,scope=__doc__,case=folder.name,source_commit=pin['commit'],
        geometries=len(steps),converged_scf_cycles=len(completed),scf_iterations=sum(completed),
        history_counts=counts,printed_coefficients=coefficients,nontrivial_full_depth_uses=nontrivial,
        maximum_final_force_ev_ang=maximum,relaxation_force_tolerance_ev_ang=tolerance,
        gpu_qualified=False,speedup_qualified=False,verifier_sha256=sha(Path(__file__)),
        evidence_sha256={n:sha(folder/n) for n in ('preflight.json','provenance.json','manifest.json','input.fdf','stdout.txt','remag.XV')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path)
    args = parser.parse_args()
    report = verify(args.reference)
    (args.reference/'relaxation-reference-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.reference/'relaxation-reference-verifier.py').write_bytes(Path(__file__).read_bytes())
    print('History relaxation reference:',report['geometries'],'geometries,',
          report['nontrivial_full_depth_uses'],'nontrivial full-depth uses; passed')
