"""Bounded reader for the pinned native, double-precision SIESTA DM format.

m_iodm.F90 writes (no_u,nspin,nsc), io_sparse.F90 writes row counts,
one column record per row, then one row per spin component. This is the
native real-space density, already integrated over the native k sampling.
No occupations, overlap inverse, or Fourier reconstruction are applied here.
"""
from pathlib import Path
import struct

import numpy as np


def read_dm(path):
    path = Path(path)
    if path.stat().st_size > 512*1024*1024:
        raise ValueError('Native DM exceeds the 512 MiB input bound')
    with path.open('rb') as stream:
        first = stream.read(4)
        if first not in (struct.pack('<i', 20), struct.pack('>i', 20)):
            raise ValueError('Expected pinned native DM header with supercell dimensions')
        endian = '<' if first == struct.pack('<i', 20) else '>'
        stream.seek(0)

        def values(kind, count):
            lead = stream.read(4)
            size = count*np.dtype(kind).itemsize
            if len(lead) != 4 or struct.unpack(endian+'i', lead)[0] != size:
                raise ValueError('Native DM record dimensions differ')
            raw = stream.read(size)
            if len(raw) != size or stream.read(4) != lead:
                raise ValueError('Truncated or corrupt native DM record')
            return np.frombuffer(raw, dtype=endian+kind).copy()

        header = values('i4', 5)
        n, spins, nx, ny, nz = map(int, header)
        if not 1 <= n <= 2048 or spins not in (1, 2, 4, 8) or min(nx, ny, nz) < 1 or nx*ny*nz > 1000000:
            raise ValueError('Invalid or unsupported native DM dimensions')
        counts = values('i4', n)
        total = int(counts.sum())
        if np.any(counts < 0) or total > 8000000 or total*(4+8*spins) > 512*1024*1024:
            raise ValueError('Invalid or oversized native DM sparsity')
        columns = np.concatenate([values('i4', int(count)) for count in counts])
        if np.any(columns < 1) or np.any(columns > n*nx*ny*nz):
            raise ValueError('Native DM column lies outside its supercell')
        density = np.array([np.concatenate([values('f8', int(count)) for count in counts]) for _ in range(spins)])
        if not np.isfinite(density).all() or stream.read(1):
            raise ValueError('Nonfinite or trailing native DM data')
    return dict(header=header, counts=counts, columns=columns, density=density, endian=endian)
