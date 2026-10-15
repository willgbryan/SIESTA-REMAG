#!/usr/bin/env python3
"""GPU tetrahedron occupations (native/tetrahedra_device.cu) against the host code (native/tetrahedra.cpp).

The bands are check_tetrahedra.py's model: a hexagonal tight-binding band
structure on full and time-reversal-reduced grids, a half-shifted grid, two
spin channels and spinless states, for the linear, Bloechl and optimized
methods, run directly and as a captured CUDA graph. The host runs with one
thread, whose accumulation order the GPU gather follows; the Fermi level is a
bisection whose sums differ only in order, so the comparison allows 1e-12 Ry
in the Fermi level and 1e-12 in the occupations. Refusals must match too.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_tetrahedra as host  # noqa: E402

P = ct.POINTER(ct.c_double)


def device(lib, method, grid, cell, k, w, energies, charge, nspinor=2, trs=False, displacement=(0., 0., 0.), graph=0):
    energies = np.ascontiguousarray(energies, dtype=np.float64)
    nk, maxspn, ne = energies.shape
    out = np.zeros_like(energies); ef = ct.c_double(); status = ct.c_int()
    keep = [np.ascontiguousarray(x, dtype=np.float64) for x in (displacement, cell, k, w)]
    code = lib.remag_check_tetra_device(method, (ct.c_int*3)(*grid), host.d(keep[0]), host.d(keep[1]), nk, host.d(keep[2]),
                                        host.d(keep[3]), int(trs), nspinor, maxspn, ne, ne, host.d(energies),
                                        ct.c_double(charge), out.ctypes.data_as(P), ct.byref(ef), graph, ct.byref(status))
    assert code == 0, code
    return status.value, out, ef.value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True, help='libremag_tetra_check.so')
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    lib = ct.CDLL(str(a.library.resolve())); lib.remag_check_tetra_device.restype = ct.c_int
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        ref = host.build(Path(tmp))
        grid = (8, 8, 5)
        k, f = host.mesh(grid); w = np.full(len(k), 1./len(k)); e = host.model_bands(k)
        kept, wr = host.reduce_trs(k, f, np.array(grid))
        ks, _ = host.mesh(grid, (.5, .5, .5)); es = host.model_bands(ks, spin_split=.3)
        layouts = [('full_spinor', dict(k=k, w=w, energies=e, charge=1.3)),
                   ('trs_reduced', dict(k=k[kept], w=wr, energies=e[kept], charge=1.3, trs=True)),
                   ('shifted_two_spin', dict(k=ks, w=w, energies=es, charge=2.1, displacement=(.5, .5, .5))),
                   ('spinless', dict(k=k, w=w, energies=e, charge=2.6, nspinor=1)),
                   ('too_few_states', dict(k=k, w=w, energies=e[:, :, :1], charge=1.5)),
                   ('top_band_occupied', dict(k=k, w=w, energies=e[:, :, :1], charge=.99))]
        for method in (1, 2, 3):
            for label, kw in layouts:
                status_h, occ_h, ef_h = host.occupations(ref, method, grid, host.CELL, **kw)
                for graph in (0, 1):
                    status_d, occ_d, ef_d = device(lib, method, grid, host.CELL, graph=graph, **kw)
                    row = dict(method=method, layout=label, graph=bool(graph), host_status=status_h, device_status=status_d)
                    if status_h == 0 and status_d == 0:
                        row.update(fermi_difference=abs(ef_d-ef_h), occupation_difference=float(np.abs(occ_d-occ_h).max()),
                                   charge=float(occ_d.sum()))
                        row['passed'] = row['fermi_difference'] < 1.e-12 and row['occupation_difference'] < 1.e-12
                    else:
                        row['passed'] = status_h == status_d
                    cases.append(row)
    report = dict(cases=cases, passed=all(c['passed'] for c in cases))
    text = json.dumps(report, indent=1)
    if a.output: a.output.mkdir(parents=True, exist_ok=True); (a.output/'check.json').write_text(text+'\n')
    print(text)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
