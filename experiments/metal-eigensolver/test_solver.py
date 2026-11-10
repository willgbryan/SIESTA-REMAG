"""Focused local Metal checks; run explicitly, outside the production suite."""
import os

import numpy as np
import pytest

from metal_seed import MetalSeed
from hybrid import BatchSolver
from cpu_fp64 import CpuFP64

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


@pytest.mark.parametrize('n', [96, 112])
def test_corrected_saved_magnetic_matrices(n):
    from captures import load, check
    cases = [c for c in load() if c['provenance']['bands'] == n]
    hs = [(c['h'], c['s']) for c in cases]
    with MetalSeed(LIBRARY, n, len(cases)) as seed:
        solver = BatchSolver(seed)
        try:
            solutions, report = solver.mixed(hs)
            assert report['fallback_count'] == 0
            for case, solution in zip(cases, solutions):
                assert check(case, solution)['component_gates_passed']
        finally:
            solver.close()


@pytest.mark.parametrize('controls', [{'seed_maximum': 0}, {'correction_maximum': 0}])
def test_failure_preserved_with_fp64_fallback(controls):
    h = np.array([[3, 0.7j], [-0.7j, 1]], dtype=complex)
    s = np.array([[1, 0.1], [0.1, 1.5]], dtype=complex)
    with MetalSeed(LIBRARY, 2, 1) as seed:
        solver = BatchSolver(seed)
        result, report = solver.mixed([(h, s)], **controls)
        assert report['fallback_count'] == 1
        assert report['jobs'][0]['fallback_reason']
        assert report['jobs'][0]['fallback_acceptance']['accepted']
        w, v = result[0]
        assert np.linalg.norm(h@v-(s@v)*w) < 1e-12


@pytest.mark.parametrize('n', [96, 112])
def test_native_cpu_comparator_on_saved_matrices(n):
    from captures import load, check
    cases = [c for c in load() if c['provenance']['bands'] == n]
    hs = [(c['h'], c['s']) for c in cases]
    solver = CpuFP64(LIBRARY, n, 4)
    try:
        for factors in (None, [c['l'] for c in cases]):
            for case, solution in zip(cases, solver.solve(hs, factors)):
                assert check(case, solution)['component_gates_passed']
    finally:
        solver.close()
