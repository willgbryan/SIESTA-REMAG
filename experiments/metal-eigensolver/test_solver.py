"""Focused local Metal checks; run explicitly, outside the production suite."""
import os

import numpy as np
import pytest

from metal_seed import MetalSeed

LIBRARY = os.environ.get('REMAG_METAL_LIBRARY', 'build/metal-eigensolver/libmetal_seed.dylib')


@pytest.mark.parametrize('n', [4, 96, 112])
def test_complex_batch_and_repeated_eigenvalues(n):
    rng = np.random.default_rng(n)
    u, _ = np.linalg.qr(rng.normal(size=(n, n))+1j*rng.normal(size=(n, n)))
    w = np.linspace(-3, 7, n)
    w[1] = w[0]  # Test a repeated eigenspace, not individual eigenvectors.
    a = (u*w) @ u.conj().T
    batch = np.stack([a, a*0.25, np.eye(n)*2])
    with MetalSeed(LIBRARY, n, 3) as plan:
        result = plan.solve(batch)
        assert np.all(result['status'] == 0)
        assert result['gpu_seconds'] is not None and result['gpu_seconds'] > 0
        for h, e, v in zip(batch, result['values'], result['vectors']):
            assert np.max(abs(np.linalg.eigvalsh(h)-np.sort(e))) < 8e-5
            assert np.linalg.norm(h@v-v*e)/np.linalg.norm(h) < 4e-6
            assert np.max(abs(v.conj().T@v-np.eye(n))) < 1e-5


def test_explicit_nonconvergence_and_lifetime():
    with MetalSeed(LIBRARY, 4, 1) as plan:
        a = np.eye(4, dtype=complex)
        a[0, 1], a[1, 0] = 0.3j, -0.3j
        result = plan.solve(a[None], maximum=0)
        assert result['status'].tolist() == [2]
        assert result['sweeps'].tolist() == [0]
        with pytest.raises(ValueError):
            plan.solve(np.full((1, 4, 4), np.nan))
        with pytest.raises(ValueError):
            plan.solve(np.zeros((2, 4, 4)))
    with pytest.raises(RuntimeError, match='Closed'):
        plan.solve(a[None])


def test_invalid_plan():
    with pytest.raises(RuntimeError, match='even dimension'):
        MetalSeed(LIBRARY, 3, 1)
