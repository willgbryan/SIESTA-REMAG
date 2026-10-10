"""Read and compare the pinned upstream's requested TSDE and NetCDF EDM.

TSDE follows m_ts_iodm.write_ts_dm and io_sparse.io_write_d2D: row counts,
columns by row, then DM/EDM by spin and row, and finally Ef (Rydberg).
These readers do not alter native spinor conventions or sparse ordering.
"""
import struct
from pathlib import Path

import numpy as np


def read_tsde(path):
    with Path(path).open('rb') as stream:
        first = stream.read(4)
        if first not in (struct.pack('<i', 20), struct.pack('>i', 20)):
            raise ValueError('Unsupported TSDE header')
        endian = '<' if first == struct.pack('<i', 20) else '>'
        stream.seek(0)

        def record(dtype, count):
            lead = stream.read(4)
            size = count*np.dtype(dtype).itemsize
            if len(lead) != 4 or struct.unpack(endian+'i', lead)[0] != size:
                raise ValueError('Invalid TSDE record length')
            data = stream.read(size)
            if len(data) != size or stream.read(4) != lead:
                raise ValueError('Truncated/corrupt TSDE record')
            return np.frombuffer(data, dtype=endian+dtype).copy()

        header = record('i4', 5)
        rows, spins = map(int, header[:2])
        nsc = header[2:]
        if not 0 < rows <= 1000000 or spins not in (1, 2, 4, 8) or np.any(nsc <= 0):
            raise ValueError('Invalid TSDE dimensions')
        counts = record('i4', rows)
        if np.any(counts < 0) or not 0 < counts.sum() <= 100000000:
            raise ValueError('Invalid TSDE sparsity')
        columns = np.concatenate([record('i4', int(n)) for n in counts])
        if np.any(columns < 1) or np.any(columns > rows*int(np.prod(nsc))):
            raise ValueError('Invalid TSDE columns')

        def matrix(components):
            return np.array([np.concatenate([record('f8', int(n)) for n in counts])
                             for _ in range(components)])

        dm, edm = matrix(spins), matrix(min(spins, 4))
        ef = record('f8', 1)
        if stream.read(1):
            raise ValueError('Unexpected trailing TSDE data')
        return dict(header=header, n_col=counts, list_col=columns, DM=dm, EDM=edm, Ef=ef)


def read_netcdf(path):
    from netCDF4 import Dataset
    with Dataset(path) as dataset:
        sparse = dataset.groups['SPARSE']
        result = {}
        for owner, names in ((dataset, ('nsc', 'cell', 'xa', 'Ef')),
                             (sparse, ('n_col', 'list_col', 'isc_off', 'S', 'DM', 'EDM'))):
            for name in names:
                value = owner.variables[name][:]
                if np.ma.is_masked(value):
                    raise ValueError('Requested NetCDF variable contains unwritten entries: '+name)
                result[name] = np.asarray(value).copy()
        if str(sparse.variables['EDM'].getncattr('unit')) != 'Ry':
            raise ValueError('Unexpected NetCDF EDM units')
        return result


def compare_outputs(reference, candidate, requests):
    from .compare import difference
    checks = {}
    # New output gates fixed before the first requested-output GPU run.
    tolerances = dict(DM=1e-7, EDM=1e-6, Ef=1e-5/13.605693122994)
    for kind, suffix, reader in (('tsde', 'TSDE', read_tsde), ('netcdf', 'nc', read_netcdf)):
        requested = requests[kind]
        if type(requested) is not bool:
            raise ValueError('Output requests must be boolean')
        a, b = reference/('remag.'+suffix), candidate/('remag.'+suffix)
        if not requested:
            checks[kind+'_not_requested'] = dict(passed=not a.exists() and not b.exists())
            continue
        try:
            left, right = reader(a), reader(b)
            for name in left:
                checks[kind+'_'+name] = difference(left[name], right[name], tolerances.get(name, 0.))
        except (OSError, ValueError, KeyError, AttributeError) as error:
            checks[kind+'_read'] = dict(passed=False, reason=str(error))
    return checks
