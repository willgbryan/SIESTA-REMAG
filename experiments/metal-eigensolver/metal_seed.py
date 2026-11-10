"""Isolated ctypes binding for the experimental Metal seed solver (serial use)."""
import ctypes as c
from pathlib import Path

import numpy as np


class MetalSeed:
    def __init__(self, library, n, capacity):
        self.n, self.capacity = n, capacity
        self.lib = c.CDLL(str(Path(library).resolve()))
        self.lib.metal_seed_create.argtypes = [c.c_char_p, c.c_uint, c.c_uint, c.c_void_p, c.c_size_t]
        self.lib.metal_seed_create.restype = c.c_void_p
        self.lib.metal_seed_destroy.argtypes = [c.c_void_p]
        self.lib.metal_seed_destroy.restype = None
        self.lib.metal_seed_device.argtypes = [c.c_void_p]
        self.lib.metal_seed_device.restype = c.c_char_p
        self.lib.metal_seed_solve.argtypes = [c.c_void_p, c.c_void_p, c.c_uint, c.c_uint, c.c_float,
            c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, c.c_size_t]
        self.lib.metal_seed_solve.restype = c.c_int
        error = c.create_string_buffer(4096)
        self.handle = self.lib.metal_seed_create(str(Path(__file__).with_name('jacobi.metal')).encode(),
                                                n, capacity, error, len(error))
        if not self.handle:
            raise RuntimeError(error.value.decode())
        self.device = self.lib.metal_seed_device(self.handle).decode()

    def close(self):
        if self.handle:
            self.lib.metal_seed_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def solve(self, matrices, *, maximum=30, tolerance=1e-6):
        if not self.handle:
            raise RuntimeError('Closed Metal plan')
        matrices = np.asarray(matrices)
        if (matrices.ndim != 3 or matrices.shape[1:] != (self.n, self.n)
                or not 1 <= len(matrices) <= self.capacity or not np.isfinite(matrices).all()
                or not 0 <= maximum <= 100 or not 1e-8 <= tolerance <= 1e-2):
            raise ValueError('Invalid matrix batch or seed controls')
        # One copy packs column-major complex<float> matrices into a shared buffer.
        packed = np.ascontiguousarray(matrices.transpose(0, 2, 1), dtype=np.complex64)
        vectors = np.empty_like(packed)
        values = np.empty((len(matrices), self.n), dtype=np.float32)
        status = np.empty(len(matrices), dtype=np.int32)
        sweeps = np.empty(len(matrices), dtype=np.uint32)
        residual = np.empty(len(matrices), dtype=np.float32)
        seconds = np.empty(2, dtype=np.float64)
        error = c.create_string_buffer(4096)
        code = self.lib.metal_seed_solve(self.handle, packed.ctypes.data, len(matrices), maximum, tolerance,
            vectors.ctypes.data, values.ctypes.data, status.ctypes.data, sweeps.ctypes.data,
            residual.ctypes.data, seconds.ctypes.data, error, len(error))
        if code:
            raise RuntimeError(error.value.decode())
        return dict(values=values, vectors=vectors.transpose(0, 2, 1), status=status,
                    sweeps=sweeps, relative_offdiagonal=residual,
                    bridge_seconds=float(seconds[0]), gpu_seconds=float(seconds[1]) if np.isfinite(seconds[1]) else None)
