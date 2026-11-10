"""Experimental Metal FP32 seeds with CPU FP64 correction and explicit fallback."""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
from scipy.linalg import cholesky, eigh, get_lapack_funcs, solve_triangular

RESIDUAL = 1e-10
ORTHOGONALITY = 1e-10


def hermitian(a):
    return (a+a.conj().T)*0.5


def standardize(h, s, factor=None):
    l = cholesky(s, lower=True, check_finite=False) if factor is None else factor
    a, info = get_lapack_funcs('hegst', (h,))(np.array(h, order='F'), l, lower=1, overwrite_a=1)
    if info:
        raise ValueError(f'FP64 HEGST failed: {info}')
    low = np.tril(a, -1)
    return low+low.conj().T+np.diag(a.diagonal().real), l


def backtransform(l, q):
    return solve_triangular(l.conj().T, q, lower=False, check_finite=False)


def acceptance(h, s, w, v):
    r = np.linalg.norm(h@v-(s@v)*w, axis=0)
    scale = (np.linalg.norm(h, 'fro')+abs(w)*np.linalg.norm(s, 'fro'))*np.linalg.norm(v, axis=0)
    residual = float(np.max(r/scale))
    ortho = float(np.max(abs(v.conj().T@s@v-np.eye(len(w)))))
    return dict(relative_residual=residual, s_orthogonality=ortho,
                accepted=bool(np.isfinite([residual, ortho]).all()
                              and residual <= RESIDUAL and ortho <= ORTHOGONALITY))


def correct(a, seed, maximum=4):
    """Correct against FP64 A; no reference eigenpairs enter this calculation."""
    q = np.asarray(seed, dtype=np.complex128)
    largest_cluster = 1
    history = []
    norm_a = np.linalg.norm(a, 'fro')
    for step in range(maximum+1):
        c = cholesky(hermitian(q.conj().T@q), lower=True, check_finite=False)
        q = solve_triangular(c, q.conj().T, lower=True, check_finite=False).conj().T
        aq = a@q
        b = hermitian(q.conj().T@aq)
        order = np.argsort(b.diagonal().real)
        q, aq, b = q[:, order], aq[:, order], b[np.ix_(order, order)]
        w = b.diagonal().real.copy()
        residual = float(np.max(np.linalg.norm(aq-q*w, axis=0)/(norm_a+abs(w))))
        history.append(residual)
        if np.isfinite(residual) and residual <= 1e-12:
            return w, q, dict(converged=True, steps=step, largest_cluster=largest_cluster, residual_history=history)
        if step == maximum:
            break
        gap = max(1e-5, 16*float(np.max(abs(b-np.diag(w)))))
        cuts = [0, *(np.flatnonzero(np.diff(w)>gap)+1).tolist(), len(w)]
        rotation = np.eye(len(w), dtype=np.complex128)
        labels = np.empty(len(w), dtype=int)
        for group, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
            labels[start:end] = group
            largest_cluster = max(largest_cluster, end-start)
            if end-start > 1:
                _, rotation[start:end, start:end] = eigh(b[start:end, start:end], check_finite=False)
        q = q@rotation
        b = hermitian(rotation.conj().T@b@rotation)
        w = b.diagonal().real.copy()
        denominator = w[None, :]-w[:, None]
        different = labels[:, None] != labels[None, :]
        if np.any(abs(denominator[different]) < gap/2):
            raise ValueError('Correction lost cluster separation')
        change = np.zeros_like(b)
        np.divide(b, denominator, out=change, where=different)
        q = q@(np.eye(len(w))+change)
    return w, q, dict(converged=False, steps=maximum, largest_cluster=largest_cluster, residual_history=history)


class BatchSolver:
    """One serial Metal plan; optional matched CPU task parallelism across jobs.

    Cached factors must come from these exact S matrices. The experiment's
    benchmark owns that pairing; this is not a general production cache API.
    """
    def __init__(self, seed_plan, workers=1):
        self.seed_plan = seed_plan
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def close(self):
        if self.pool:
            self.pool.shutdown()

    def map(self, fn, jobs):
        return list(self.pool.map(fn, jobs)) if self.pool else list(map(fn, jobs))

    def cpu(self, hs, factors=None):
        def one(job):
            index, (h, s) = job
            if factors is None:
                return eigh(h, s, driver='gvd', check_finite=False)
            a, l = standardize(h, s, factors[index])
            w, q = eigh(a, driver='evd', check_finite=False)
            return w, backtransform(l, q)
        return self.map(one, enumerate(hs))

    def mixed(self, hs, factors=None, *, seed_maximum=30, correction_maximum=4):
        start = perf_counter()
        prepared = self.map(lambda job: standardize(*job[1], None if factors is None else factors[job[0]]), enumerate(hs))
        preparation_seconds = perf_counter()-start
        start = perf_counter()
        seed_error = None
        try:
            seeds = self.seed_plan.solve(np.stack([a for a, _ in prepared]), maximum=seed_maximum)
        except RuntimeError as error:
            seed_error = str(error)
            seeds = None
        seed_seconds = perf_counter()-start

        def one(job):
            index, (h, s) = job
            a, l = prepared[index]
            record = dict(seed_status=None if seeds is None else int(seeds['status'][index]),
                          seed_sweeps=None if seeds is None else int(seeds['sweeps'][index]),
                          correction=None, fallback_reason=None)
            if seeds is None:
                record['fallback_reason'] = 'Metal command failure: '+seed_error
            elif seeds['status'][index] != 0:
                record['fallback_reason'] = 'Metal seed did not converge'
            else:
                try:
                    w, q, record['correction'] = correct(a, seeds['vectors'][index], correction_maximum)
                    if not record['correction']['converged']:
                        raise ValueError('FP64 correction exhausted its iteration limit')
                    v = backtransform(l, q)
                    record['acceptance'] = acceptance(h, s, w, v)
                    if not record['acceptance']['accepted']:
                        raise ValueError('Corrected generalized eigenpairs failed acceptance')
                except (ValueError, np.linalg.LinAlgError) as error:
                    record['fallback_reason'] = str(error)
            if record['fallback_reason'] is not None:
                # Full original FP64 solve; retain failure and include its cost.
                w, v = eigh(h, s, driver='gvd', check_finite=False)
                record['fallback_acceptance'] = acceptance(h, s, w, v)
                if not record['fallback_acceptance']['accepted']:
                    raise ValueError('FP64 fallback failed acceptance')
            return (w, v), record

        start = perf_counter()
        completed = self.map(one, enumerate(hs))
        correction_seconds = perf_counter()-start
        return [x[0] for x in completed], dict(preparation_seconds=preparation_seconds,
            seed_wall_seconds=seed_seconds, seed_gpu_seconds=None if seeds is None else seeds['gpu_seconds'],
            correction_and_acceptance_seconds=correction_seconds,
            jobs=[x[1] for x in completed], fallback_count=sum(x[1]['fallback_reason'] is not None for x in completed))
