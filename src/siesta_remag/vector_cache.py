"""Bind configured eigenvector storage to the supported corpus and EIG shapes.

Solve counts describe the captured algorithm, not independent kernel counters.
The corpus uses unshifted Gamma for [1,1,1]; other meshes use sampled k points.
"""


def record_errors(cycles, eigenvalues, benchmark, enabled, solver_lanes=1, warm_start=False):
    errors = []
    if type(warm_start) is not bool:
        return ['Invalid requested eigensolver warm-start mode']
    if type(solver_lanes) is not int or solver_lanes not in (1, 2, 4, 8):
        return ['Invalid requested solver workspace count']
    if not cycles or not eigenvalues:
        return ['Missing vector-cache cycle/spectrum evidence']
    try:
        dimension, spins, nk = (eigenvalues[k] for k in ('nbands', 'nspin', 'nk'))
        gamma = benchmark['kmesh'] == [1, 1, 1]
        if spins not in (1, 2, 4, 8) or dimension < 1 or nk < 1 or (gamma and nk != 1):
            return ['Invalid vector-cache spectrum dimensions']
        jobs = nk*(spins if spins < 4 else 1)
        retained = gamma or enabled
        vector_bytes = dimension*dimension*(8 if gamma and spins < 4 else 16)
        expected = dict(jobs=jobs, solves_per_iteration=jobs*(1 if retained else 2),
                        retains_all_vectors=retained, vector_bank_bytes=vector_bytes*(jobs if retained else 1))
        for cycle in cycles:
            actual_warm_start = cycle.get('warm_start', False)
            if (type(actual_warm_start) is not bool or actual_warm_start != warm_start or
                    (actual_warm_start and not retained)):
                errors.append('Eigensolver warm-start mode differs from selected mode or vector storage')
            actual = dict(cycle.get('eigensystems', {}))
            # Older serial results have no lane field. The new field records
            # effective workspaces, capped by independent eigensystem jobs.
            lanes = actual.pop('solver_lanes', 1)
            if (type(lanes) is not int or lanes != min(solver_lanes, jobs) or
                    (lanes > 1 and not retained)):
                errors.append('Solver workspace count differs from selected mode or vector storage')
            if actual != expected:
                errors.append('Vector bank/solve configuration differs from EIG and selected mode')
            terminal = cycle.get('terminal', {})
            if terminal.get('spins') != spins or terminal.get('bank_size') != dimension*jobs:
                errors.append('Terminal spectra do not match the configured vector bank')
            if cycle.get('workspace_bytes', 0) < expected['vector_bank_bytes']:
                errors.append('Vector bank exceeds recorded workspace')
    except (KeyError, TypeError, ValueError):
        errors.append('Incomplete vector-cache shape evidence')
    return errors
