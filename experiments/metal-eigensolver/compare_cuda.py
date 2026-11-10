#!/usr/bin/env python3
"""Matched Mac CPU / H100 component comparison on the saved magnetic captures."""
import argparse
import ctypes as c
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SETTINGS = dict(REMAG_BLOCK_WIDTH='32', REMAG_BLOCK_LOCAL_SOLVER='jacobi',
    REMAG_BLOCK_SHARED_JACOBI='1', REMAG_BLOCK_FUSED_MATRIX='0',
    REMAG_BLOCK_FP64_VECTORS='0', REMAG_BLOCK_FP64_MATRIX='0', REMAG_SCF_WARM_START='1',
    CUDA_DISABLE_PTX_JIT='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def prepare(folder):
    from captures import ROOT, load
    cases = load()
    previous = ROOT/'benchmarks/evidence/apple-metal-solver-20261110/benchmark-03-native.json.gz'
    cpu = json.loads(gzip.decompress(previous.read_bytes()))
    if cpu['state'] != 'complete' or cpu['cpu_backend'] != 'native':
        raise ValueError('Completed native CPU timing evidence required')
    if cpu['inputs'] != [case['provenance'] for case in cases]:
        raise ValueError('CPU timing inputs differ')
    for name in ('cpu_fp64.cpp', 'cpu_fp64.py'):
        path = HERE/name
        if digest(path) != cpu['source_files'][str(path.relative_to(ROOT))]:
            raise ValueError('CPU implementation differs from retained timings')
    arrays = {}
    for n in (96, 112):
        group = [case for case in cases if case['provenance']['bands'] == n]
        for key in ('h', 's', 'reference'):
            arrays[f'{n}_{key}'] = np.stack([case[key] for case in group])
        for index, key in enumerate(('l', 'mu', 'kt', 'density')):
            arrays[f'{n}_{key}'] = np.stack([case['density_context'][index] for case in group])
    selected = [sample for sample in cpu['samples'] if sample['cpu_workers'] == 4 and not sample['overlap_cached']]
    if len(selected) != 4 or not all(sample['component_qualified'] for sample in selected):
        raise ValueError('Four qualified fresh-overlap CPU configurations required')
    folder.mkdir(parents=True, exist_ok=False)
    np.savez(folder/'inputs.npz', **arrays)
    sources = [ROOT/'native'/name for name in ('block_jacobi.cu', 'block_jacobi.hpp', 'shared_local_jacobi.inc')]
    sources += [HERE/name for name in ('cuda_compare.cu', 'compare_cuda.py', 'cuda_app.py')]
    oracle = ROOT/'benchmarks/evidence/apple-mixed-precision-20261109/replay.py'
    sources.append(oracle)
    save(folder/'experiment.json', dict(schema=1, state='prepared', inputs=[case['provenance'] for case in cases],
        input_sha256=digest(folder/'inputs.npz'), source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        cpu_report=str(previous.relative_to(ROOT)), cpu_report_sha256=digest(previous),
        cpu_library_sha256=cpu['library_sha256'], cpu_platform=cpu['platform'], cpu_samples=selected,
        settings=SETTINGS, repetitions=5, lanes=8, maximum_sweeps=32, tolerance=2e-13,
        requested_gpu='H100!:1', proposed_budget_seconds=600, gpu_budget_authorized=False,
        scope='Cross-hardware selected-matrix component timing; not full SCF, MAE, new k points, or upstream-GPU speedup',
        warm_scope='Identical-input eigenbasis reuse is a best-case control, not a changing-Hamiltonian SCF benchmark',
        shipment_gate='No production promotion from component replay alone; matched end-to-end validation remains required'))
    print(folder)


class Result(c.Structure):
    _fields_ = [('status', c.c_int), ('sweeps', c.c_int), ('relative_offdiagonal', c.c_double),
                ('history', c.c_double*32)]


def bind(library):
    lib = c.CDLL(str(library.resolve()))
    lib.remag_compare_create.argtypes = [c.c_int, c.c_int, c.c_void_p, c.c_void_p]
    lib.remag_compare_create.restype = c.c_void_p
    lib.remag_compare_solve.argtypes = [c.c_void_p, c.c_int, c.c_int]+[c.c_void_p]*6
    lib.remag_compare_solve.restype = c.c_int
    lib.remag_compare_fetch.argtypes = [c.c_void_p]*3
    lib.remag_compare_fetch.restype = c.c_int
    lib.remag_compare_destroy.argtypes = [c.c_void_p]
    lib.remag_compare_destroy.restype = None
    return lib


def worker(folder, output, library, oracle_path):
    os.environ.update(SETTINGS)
    manifest = json.loads((folder/'experiment.json').read_text())
    if digest(folder/'inputs.npz') != manifest['input_sha256']:
        raise ValueError('Input bank changed')
    if digest(oracle_path) != manifest['source_sha256']['benchmarks/evidence/apple-mixed-precision-20261109/replay.py']:
        raise ValueError('Independent metrics changed')
    spec = importlib.util.spec_from_file_location('precision_oracle', oracle_path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    lib = bind(library)
    if output.exists():
        raise ValueError('Retain prior reports; choose a new output')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True).splitlines()
    if len(gpu) != 1 or 'H100' not in gpu[0]:
        raise ValueError('Exactly one H100 required')
    report = dict(schema=1, state='running', gpu=gpu, library_sha256=digest(library),
        manifest_sha256=digest(folder/'experiment.json'), settings=SETTINGS, samples=[],
        performance_scope=manifest['scope'], warm_scope=manifest['warm_scope'], production_promoted=False,
        resident_timing='Graph CUDA events; host wall additionally includes status download and synchronization. Eigenpair fetch outside timing.',
        host_timing='Wall includes input packing/finite checks, H/S upload, graph execution, eigenpair/status download; setup and independent verification excluded.')
    save(output, report)
    try:
        with np.load(folder/'inputs.npz', allow_pickle=False) as bank:
            for baseline in manifest['cpu_samples']:
                n, multiplier = baseline['bands'], baseline['exact_replay_multiplier']
                hs = [(h, s) for h, s in zip(bank[f'{n}_h'], bank[f'{n}_s'], strict=True)]*multiplier
                count = len(hs)
                if count != baseline['batch']:
                    raise ValueError('CPU and GPU batch sizes differ')
                def pack():
                    h = np.ascontiguousarray(np.stack([h.T for h, _ in hs]), dtype=np.complex128)
                    s = np.ascontiguousarray(np.stack([s.T for _, s in hs]), dtype=np.complex128)
                    if not np.isfinite(h).all() or not np.isfinite(s).all():
                        raise ValueError('Nonfinite input')
                    return h, s
                h, s = pack()
                started = time.perf_counter()
                plan = lib.remag_compare_create(n, count, h.ctypes.data, s.ctypes.data)
                setup = time.perf_counter()-started
                if not plan:
                    raise RuntimeError('CUDA plan creation failed')
                try:
                    for cold in (True, False):
                        for resident in (True, False):
                            sample = dict(material=baseline['material'], bands=n, batch=count,
                                unique_captures=baseline['unique_captures'], exact_replay_multiplier=multiplier,
                                cold_basis=cold, resident=resident, lanes=min(8, count), setup_seconds=setup,
                                cpu_median_seconds=baseline['cpu_median_seconds'], repetitions=[])
                            report['samples'].append(sample)
                            # Every warm sample is first seeded from a fresh solve.
                            for repetition in range(-1, 5):
                                z = np.empty_like(h); w = np.empty((count, n), dtype=np.float64)
                                statuses = (Result*count)(); graph_seconds = c.c_double()
                                start = time.perf_counter()
                                if not resident:
                                    h, s = pack()
                                    # Include host output allocation as in CpuFP64.solve.
                                    z = np.empty_like(h); w = np.empty((count, n), dtype=np.float64)
                                code = lib.remag_compare_solve(plan, int(cold or repetition == -1), int(resident),
                                    h.ctypes.data, s.ctypes.data, z.ctypes.data, w.ctypes.data, statuses, c.byref(graph_seconds))
                                elapsed = time.perf_counter()-start
                                row = dict(index=repetition, wall_seconds=elapsed, graph_seconds=graph_seconds.value,
                                    returncode=code, statuses=[r.status for r in statuses], sweeps=[r.sweeps for r in statuses])
                                if repetition == -1:
                                    sample['warmup'] = row
                                else:
                                    sample['repetitions'].append(row)
                                save(output, report)
                                if code:
                                    raise RuntimeError('CUDA eigensolve failed; partial timing/status records retained')
                                if resident and lib.remag_compare_fetch(plan, z.ctypes.data, w.ctypes.data):
                                    raise RuntimeError('CUDA eigenpair fetch failed')
                                if any(row['statuses']) or not np.isfinite(w).all() or not np.isfinite(z).all():
                                    np.savez_compressed(output.parent/f'failed-{n}-{count}-{cold}-{resident}-{repetition}.npz', w=w, z=z.transpose(0, 2, 1))
                                    raise RuntimeError('CUDA status or finite-output check failed; partial eigenpairs retained')
                                checks = []
                                unique = baseline['unique_captures']
                                for job, (hj, sj) in enumerate(hs):
                                    index = job % unique
                                    context = tuple(bank[f'{n}_{key}'][index] for key in ('l', 'mu', 'kt', 'density'))
                                    checks.append(oracle.metrics(w[job], z[job].T, hj, sj, bank[f'{n}_reference'][index], context))
                                row['verification'] = dict(passed=all(check['component_gates_passed'] for check in checks),
                                    count=count, maxima={key: max(check[key] for check in checks) for key in checks[0]
                                                        if key != 'component_gates_passed'})
                                save(output, report)
                                if not row['verification']['passed']:
                                    np.savez_compressed(output.parent/f'failed-{n}-{count}-{cold}-{resident}-{repetition}.npz', w=w, z=z.transpose(0, 2, 1))
                                    raise RuntimeError('Independent accuracy gates failed; no qualified speed ratio')
                            np.savez_compressed(output.parent/f'eigenpairs-{n}-{count}-{cold}-{resident}.npz', w=w, z=z.transpose(0, 2, 1))
                            sample['component_qualified'] = True
                            sample['wall_median_seconds'] = statistics.median(row['wall_seconds'] for row in sample['repetitions'])
                            sample['graph_median_seconds'] = statistics.median(row['graph_seconds'] for row in sample['repetitions'])
                            sample['cpu_over_gpu_wall'] = sample['cpu_median_seconds']/sample['wall_median_seconds']
                            save(output, report)
                            print(json.dumps({k: sample[k] for k in ('bands', 'batch', 'cold_basis', 'resident', 'wall_median_seconds', 'graph_median_seconds', 'cpu_over_gpu_wall')}), flush=True)
                finally:
                    lib.remag_compare_destroy(plan)
        report['state'] = 'complete'
    except Exception as error:
        report.update(state='failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        save(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'gpu'])
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--oracle', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.folder)
    else:
        if any(value is None for value in (args.output, args.library, args.oracle)):
            parser.error('GPU worker requires --output, --library and --oracle')
        worker(args.folder, args.output, args.library, args.oracle)
