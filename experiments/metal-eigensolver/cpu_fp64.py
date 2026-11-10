"""Direct batched Accelerate CPU comparator, independent of Python task scheduling."""
import ctypes as c
from pathlib import Path

import numpy as np


class CpuFP64:
    def __init__(self, library, n, workers):
        self.n = n
        self.lib = c.CDLL(str(Path(library).resolve()))
        self.lib.cpu_fp64_create.argtypes = [c.c_uint, c.c_uint]
        self.lib.cpu_fp64_create.restype = c.c_void_p
        self.lib.cpu_fp64_destroy.argtypes = [c.c_void_p]
        self.lib.cpu_fp64_destroy.restype = None
        self.lib.cpu_fp64_solve.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_uint, c.c_int, c.c_void_p, c.c_void_p, c.c_void_p]
        self.lib.cpu_fp64_solve.restype = c.c_int
        self.handle = self.lib.cpu_fp64_create(n, workers)
        if not self.handle:
            raise RuntimeError('Cannot create Accelerate FP64 plan')

    def close(self):
        if self.handle:
            self.lib.cpu_fp64_destroy(self.handle)
            self.handle = None

    def solve(self, hs, factors=None):
        if not self.handle or not 1 <= len(hs) <= 512 or (factors is not None and len(factors) != len(hs)):
            raise ValueError('Invalid or closed CPU plan/batch')
        h = np.ascontiguousarray(np.stack([h.T for h, _ in hs]), dtype=np.complex128)
        s = np.ascontiguousarray(np.stack([s.T for _, s in hs] if factors is None else [l.T for l in factors]), dtype=np.complex128)
        if h.shape[1:] != (self.n, self.n) or s.shape != h.shape or not np.isfinite(h).all() or not np.isfinite(s).all():
            raise ValueError('Invalid CPU matrices')
        vectors, values = np.empty_like(h), np.empty((len(hs), self.n), dtype=np.float64)
        status = np.empty(len(hs), dtype=np.int32)
        code = self.lib.cpu_fp64_solve(self.handle, h.ctypes.data, s.ctypes.data, len(hs), int(factors is not None),
            vectors.ctypes.data, values.ctypes.data, status.ctypes.data)
        if code or np.any(status):
            raise RuntimeError(f'Accelerate FP64 solve failed: code={code}, status={status.tolist()}')
        return list(zip(values, vectors.transpose(0, 2, 1)))
