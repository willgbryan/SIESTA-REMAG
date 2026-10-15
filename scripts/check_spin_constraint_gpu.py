#!/usr/bin/env python3
"""Qualify the device spin-constraint kernels against the shared host header.

Both sides compile spin_constraint_math.hpp without contraction, so the
device field and moment kernels must agree bitwise with the host for every
fixture, uncaptured and as a replayed CUDA graph. Exact dyadic fixtures also
match the operator oracle of check_spin_constraint_math.py."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import tempfile

import numpy as np
from check_spin_constraint_math import (NATIVE, Sparse, build, dyadic, load, pack_density, ptr, random_pattern,
                                        sha, spinor_blocks)

I32 = np.int32


class Info(ct.Structure):
    _fields_ = [(k, ct.c_size_t) for k in ('allocated', 'staged', 'exported')]+[(k, ct.c_int) for k in ('graph_nodes', 'forbidden_nodes', 'stage')]


def pairs(sp, folded):
    """Pairs in hamiltonian.cu order: key col*n+row, stable sort, grouped."""
    entries = []
    for k, (io, _, _) in enumerate(sp.entries):
        orb = int(sp.cols[k]); row = (orb-1) % sp.n if folded else int(sp.mapping[orb-1])-1
        entries.append((io*sp.n+row, k))
    entries.sort(key=lambda e: e[0])
    out, indices, i = [], [], 0
    while i < len(entries):
        begin = i; key = entries[i][0]
        while i < len(entries) and entries[i][0] == key: indices.append(entries[i][1]); i += 1
        out.append((key % sp.n, key//sp.n, begin, i))
    return np.ascontiguousarray(np.array(out, I32)), np.ascontiguousarray(np.array(indices, I32))


def fixture(rng, spins, images, exact):
    n = int(rng.integers(7, 40)); sp = random_pattern(rng, n, images)
    sites = rng.integers(-1, 3, size=n).astype(I32)
    def value(shape): return dyadic(rng, shape) if exact else rng.normal(size=shape)
    s = np.zeros(sp.capacity)
    for k, m in enumerate(sp.mirror):
        if k <= m: s[k] = s[m] = value(())
    blocks = spinor_blocks(rng, sp, spins == 4) if exact else None
    if blocks is None:
        blocks = np.zeros((sp.capacity, 2, 2), complex)
        for k, m in enumerate(sp.mirror):
            if k > m: continue
            x = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
            if k == m or spins == 4: x = np.triu(x)+np.triu(x, 1).conj().T; x[0, 0] = x[0, 0].real; x[1, 1] = x[1, 1].real
            blocks[k] = x; blocks[m] = x.conj().T
    d = np.ascontiguousarray(pack_density(blocks, spins))
    b = value((3, 3)); field = np.ascontiguousarray(np.array([b[t] if t >= 0 else np.zeros(3) for t in sites]).ravel())
    h0 = np.ascontiguousarray(value((spins, sp.capacity)))
    return sp, s, d, field, h0


def run(library, output):
    output = output.resolve(); output.mkdir(parents=True, exist_ok=True)
    fn = ct.CDLL(str(library.resolve())).remag_check_spin_constraint
    fn.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]+[ct.c_void_p]*7+[ct.c_double]+[ct.c_void_p]*6+[ct.POINTER(Info)]
    fn.restype = ct.c_int
    report = dict(script_sha256=sha(__file__), library_sha256=sha(library), integrated_into_siesta=False,
                  header_sha256=sha(NATIVE/'spin_constraint_math.hpp'), kernel_sha256=sha(NATIVE/'spin_constraint.cu'), checks=[])
    with tempfile.TemporaryDirectory(prefix='remag-spin-constraint-gpu-') as tmp:
        host = load(build(tmp)); rng = np.random.default_rng(20261014)
        for spins in (8, 4):
            for images in (False, True):
                for exact in (True, False):
                    sp, s, d, field, h0 = fixture(rng, spins, images, exact)
                    p, indices = pairs(sp, not images)
                    site = np.array([i % 4-1 for i in range(sp.n)], I32)      # sites 0..2 and unconstrained rows
                    site_fields = np.ascontiguousarray(rng.normal(size=9)); targets = np.ascontiguousarray(
                        np.array([[0., 0., 1.], [np.sin(1.), 0., np.cos(1.)], [.6, .64, .48]]).ravel())
                    for penalty in (0., .37):
                      for captured in (0, 1):
                        for cycles in (1, 3):
                            expected = h0.copy(); moments = np.zeros(3*sp.n)
                            host.moments(spins, sp.n, sp.capacity, ptr(sp.num), ptr(sp.ptr), ptr(s), ptr(d), ptr(moments))
                            applied = field
                            if penalty:
                                applied = np.zeros(3*sp.n)
                                host.penalize(sp.n, ptr(site), ptr(moments), ptr(site_fields), ptr(targets), penalty, ptr(applied))
                            for _ in range(cycles):
                                host.pack(spins, sp.n, sp.capacity, ptr(sp.num), ptr(sp.ptr), ptr(sp.cols),
                                          ptr(None if not images else sp.mapping), ptr(s), ptr(applied), ptr(expected))
                            h = h0.copy(); device = np.full(3*sp.n, np.nan); penalized = np.full(3*sp.n, np.nan); info = Info()
                            config = np.array([3 if images else 2, sp.n, sp.capacity, spins, len(p), 3], I32)
                            code = fn(ptr(config), captured, cycles, ptr(p), ptr(indices), ptr(s), ptr(field),
                                      ptr(sp.num), ptr(sp.ptr), ptr(d), penalty, ptr(site), ptr(site_fields), ptr(targets),
                                      ptr(h), ptr(device), ptr(penalized), ct.byref(info))
                            same_field = not penalty or np.array_equal(penalized, applied)
                            ok = code == 0 and info.stage == 4 and np.array_equal(h, expected) and np.array_equal(device, moments) and \
                                same_field and (not captured or (info.graph_nodes == (3 if penalty else 1) and info.forbidden_nodes == 0))
                            report['checks'].append(dict(spins=spins, images=images, exact=bool(exact), penalty=penalty, captured=captured,
                                cycles=cycles, entries=sp.capacity, pairs=len(p), status=code, stage=info.stage,
                                graph_nodes=info.graph_nodes, forbidden_nodes=info.forbidden_nodes,
                                max_h_error=float(np.max(np.abs(h-expected))), max_moment_error=float(np.max(np.abs(device-moments))),
                                bitwise=bool(np.array_equal(h, expected) and np.array_equal(device, moments)), passed=bool(ok)))
    report['passed'] = all(c['passed'] for c in report['checks'])
    (output/'check.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.library, args.output)
    print(json.dumps(dict(passed=result['passed'], checks=len(result['checks']))))
    raise SystemExit(0 if result['passed'] else 1)
