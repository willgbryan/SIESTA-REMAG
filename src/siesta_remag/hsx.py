"""Read pinned SIESTA HSX v2: io_hsx.F90 and io_sparse.F90 row records."""
from pathlib import Path
import struct

import numpy as np


def read_hsx(path):
    with Path(path).open('rb') as stream:
        first = stream.read(4)
        if first not in (struct.pack('<i', 4), struct.pack('>i', 4)):
            raise ValueError('Unsupported HSX record markers')
        endian = '<' if first == struct.pack('<i', 4) else '>'
        stream.seek(0)
        def record():
            lead = stream.read(4)
            if len(lead) != 4:
                raise ValueError('Truncated HSX record')
            size = struct.unpack(endian+'i', lead)[0]
            if size < 0 or size > 2**29:
                raise ValueError('Unsupported HSX record length')
            data = stream.read(size)
            if len(data) != size or stream.read(4) != lead:
                raise ValueError('Corrupt HSX record')
            return data
        def values(kind, size):
            data = np.frombuffer(record(), dtype=endian+kind)
            if len(data) != size:
                raise ValueError('HSX record dimensions differ')
            return data.copy()
        if values('i4', 1)[0] != 2:
            raise ValueError('Only pinned HSX version 2 is supported')
        precision = values('i4', 1)[0]
        if precision not in (0, 1):
            raise ValueError('Invalid HSX precision')
        header = values('i4', 7)
        atoms, rows, spins, species, nx, ny, nz = map(int, header)
        if (min(atoms, rows, species, nx, ny, nz) < 1 or atoms > rows or
                rows > 1000000 or nx*ny*nz > 10000000 or spins not in (1, 2, 4, 8)):
            raise ValueError('Invalid HSX dimensions')
        scalars = values('f8', 12)
        if not np.isfinite(scalars).all():
            raise ValueError('Nonfinite HSX header')
        data = record()
        noff = 3*nx*ny*nz
        if len(data) != 4*noff+32*atoms:
            raise ValueError('Invalid HSX geometry dimensions')
        offsets = np.frombuffer(data[:4*noff], dtype=endian+'i4').copy()
        positions = np.frombuffer(data[4*noff:4*noff+24*atoms], dtype=endian+'f8').copy()
        labels = np.frombuffer(data[4*noff+24*atoms:], dtype=endian+'i4').copy()
        if not np.isfinite(positions).all() or np.any(labels[:atoms] < 1) or np.any(labels[:atoms] > species) or labels[-1] != rows:
            raise ValueError('Invalid HSX coordinates/species/orbital ranges')
        basis = [record() for _ in range(species+1)]
        if len(basis[0]) != 32*species:
            raise ValueError('Invalid HSX species header')
        for i, shell in enumerate(basis[1:]):
            n = struct.unpack(endian+'i', basis[0][32*i+28:32*i+32])[0]
            if n < 1 or len(shell) != 12*n:
                raise ValueError('Invalid HSX orbital header')
        kpoints = record()
        if len(kpoints) != 60:
            raise ValueError('Invalid HSX k-point header')
        counts = values('i4', rows)
        if np.any(counts < 0) or int(counts.sum()) > 100000000:
            raise ValueError('Invalid HSX sparse row counts')
        columns = np.concatenate([values('i4', int(n)) for n in counts])
        if np.any(columns < 1) or np.any(columns > rows*nx*ny*nz):
            raise ValueError('Invalid HSX sparse columns')
        kind = 'f8' if precision else 'f4'
        h = np.array([np.concatenate([values(kind, int(n)) for n in counts]) for _ in range(spins)])
        overlap = np.concatenate([values(kind, int(n)) for n in counts])
        if not np.isfinite(h).all() or not np.isfinite(overlap).all() or stream.read(1):
            raise ValueError('Nonfinite or trailing HSX matrix data')
    return dict(header=header, precision=int(precision), cell=scalars[:9], fermi_ry=float(scalars[9]),
                charge=float(scalars[10]), temperature_ry=float(scalars[11]), offsets=offsets,
                positions=positions, labels=labels, basis=basis, kpoints=kpoints, counts=counts,
                columns=columns, h=h, overlap=overlap)
