"""ctypes interface to native CUDA; deliberately no CPU fallback."""
import ctypes as ct
import os
from pathlib import Path
import numpy as np


class MftResult(ct.Structure):
    _fields_ = [(key, ct.c_double) for key in (
        'chemical_potential', 'charge', 'band_energy', 'free_energy',
        'delta_band', 'delta_free', 'delta_canonical_free')]+[
        ('spin', ct.c_double*3), ('torque', ct.c_double*3),
        ('max_residual', ct.c_double), ('max_orthogonality_error', ct.c_double),
        ('seconds', ct.c_double), ('workspace_bytes', ct.c_size_t)]


def library(path=None):
    selected = path or os.environ.get('REMAG_MAGNETIC_LIBRARY')
    if not selected:
        raise RuntimeError('Set REMAG_MAGNETIC_LIBRARY to the CUDA libremag_magnetic.so; no CPU solver is substituted')
    lib = ct.CDLL(str(Path(selected).expanduser().resolve()))
    ptr = ct.c_void_p
    lib.remag_magnetic_error.restype = ct.c_char_p
    lib.remag_mft_create.argtypes = [ct.c_int, ct.c_int]+[ptr]*5+[
        ct.c_double, ct.c_double, ct.c_size_t, ct.POINTER(ptr)]
    lib.remag_mft_create.restype = ct.c_int
    lib.remag_mft_sweep.argtypes = [ptr, ptr, ct.c_double, ct.c_int, ct.POINTER(MftResult)]+[ptr]*3
    lib.remag_mft_sweep.restype = ct.c_int
    lib.remag_mft_destroy.argtypes = [ptr]
    lib.remag_mft_destroy.restype = None
    return lib


class MftPlan:
    def __init__(self, state, *, library_path=None, memory_limit_bytes=2*1024**3):
        self.lib = library(library_path)
        self.handle = ct.c_void_p()
        self.nk, self.n, _ = state.shape
        # Fortran matrices inside a contiguous k bank, matching the native ABI.
        matrices = [np.ascontiguousarray(getattr(state, name).swapaxes(-1, -2))
                    for name in ('h_base', 'h_mag', 'h_soc', 'overlap')]
        weights = np.ascontiguousarray(state.weights)
        self.check(self.lib.remag_mft_create(self.n, self.nk, *[a.ctypes.data for a in matrices],
            weights.ctypes.data, state.metadata['temperature_ry'], state.metadata['electrons'],
            memory_limit_bytes, ct.byref(self.handle)))

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.remag_magnetic_error().decode('utf8', errors='replace'))

    def solve(self, quaternion, *, soc=True, reference=False):
        q = np.ascontiguousarray(quaternion, dtype=np.float64)
        if q.shape != (4,) or not np.isfinite(q).all() or not np.any(q):
            raise ValueError('Finite quaternion required')
        e, occ = np.empty((self.nk, self.n)), np.empty((self.nk, self.n))
        population = np.empty((self.nk, self.n, self.n//2))
        result = MftResult()
        self.check(self.lib.remag_mft_sweep(self.handle, q.ctypes.data, float(soc), int(reference),
            ct.byref(result), e.ctypes.data, occ.ctypes.data, population.ctypes.data))
        if not all(np.isfinite(a).all() for a in (e, occ, population)):
            raise RuntimeError('Native MFT returned nonfinite arrays')
        values = {key: (list(getattr(result, key)) if key in ('spin', 'torque') else getattr(result, key))
                  for key, _ in result._fields_}
        return values, e, occ, population

    def close(self):
        if self.handle:
            self.lib.remag_mft_destroy(self.handle)
            self.handle = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
