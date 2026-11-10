"""Rotate scalar SIESTA DM to an SOC guess, preserving charge and sparse indices.

Little-endian record layout and spin convention checked against sisl 0.20.
The result is a guess only: each direction still requires self-consistent SOC.
"""
from pathlib import Path
import struct

import numpy as np


def records(path):
    raw = Path(path).read_bytes()
    parts, offset = [], 0
    while offset < len(raw):
        if len(raw)-offset < 8:
            raise ValueError('Truncated density record')
        size = struct.unpack_from('<i', raw, offset)[0]
        if not 0 <= size <= len(raw)-offset-8 or struct.unpack_from('<i', raw, offset+4+size)[0] != size:
            raise ValueError('Invalid density record markers')
        parts.append(raw[offset+4:offset+4+size])
        offset += size+8
    if not parts:
        raise ValueError('Empty density matrix')
    return parts


def collinear_to_soc(source, destination, direction):
    parts = records(source)
    header = np.frombuffer(parts[0], dtype='<i4').copy()
    if len(header) not in (2, 5) or header[0] <= 0 or header[1] != 2:
        raise ValueError('Expected a scalar two-spin density matrix')
    no = int(header[0])
    if len(parts) != 2+3*no:
        raise ValueError('Wrong density record count')
    ncol = np.frombuffer(parts[1], dtype='<i4')
    if len(ncol) != no or np.any(ncol < 0) or not ncol.sum():
        raise ValueError('Invalid density sparse row counts')
    for i, count in enumerate(ncol):
        if len(parts[2+i]) != count*4 or any(len(parts[2+no+s*no+i]) != count*8 for s in range(2)):
            raise ValueError('Density sparse row length mismatch')
    values = np.array([np.concatenate([np.frombuffer(p, dtype='<f8') for p in parts[2+no+s*no:2+no+(s+1)*no]]) for s in range(2)]).T
    axis = np.asarray(direction, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0 or not np.isfinite(values).all():
        raise ValueError('Nonfinite density or invalid rotation axis')
    axis = axis/np.linalg.norm(axis)
    charge, polarization = values.sum(axis=1), values[:, 0]-values[:, 1]
    dm = np.zeros((len(values), 8), dtype='<f8')
    dm[:, 0] = (charge+polarization*axis[2])/2
    dm[:, 1] = (charge-polarization*axis[2])/2
    dm[:, 2] = dm[:, 6] = polarization*axis[0]/2
    dm[:, 3] = dm[:, 7] = polarization*axis[1]/2
    error = float(np.max(abs(dm[:, 0]+dm[:, 1]-charge)))
    if error > 1e-14:
        raise ValueError('Density rotation changed charge')
    header[1] = 8
    out = [header.tobytes(), *parts[1:2+no]]
    offsets = np.r_[0, np.cumsum(ncol)]
    for spin in range(8):
        out.extend(dm[offsets[i]:offsets[i+1], spin].tobytes() for i in range(no))
    with Path(destination).open('xb') as stream:
        for record in out:
            size = struct.pack('<i', len(record))
            stream.write(size+record+size)
    return dict(spatial_orbitals=no, spin_components=8, sparse_entries=len(values),
                direction=axis.tolist(), charge_max_error=error,
                purpose='Starting guess only; separate full self-consistent SOC required')
