"""One bounded H100 comparison; launch only through scripts/run_modal.py."""
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREPARED = ROOT/'build/metal-cuda-comparison'
REMOTE = Path('/opt/cpu-gpu-compare')
app = modal.App('siesta-remag-single-gpu')
volume = modal.Volume.from_name('siesta-remag-results')
image = modal.Image.from_id('im-4glsGDWvS83fq6PCciqp8I')
for source in [ROOT/'native'/name for name in ('block_jacobi.cu', 'block_jacobi.hpp', 'shared_local_jacobi.inc')]+[
        HERE/name for name in ('cuda_compare.cu', 'compare_cuda.py', 'cuda_app.py')]+[
        ROOT/'benchmarks/evidence/apple-mixed-precision-20261109/replay.py']:
    image = image.add_local_file(source, str(REMOTE/source.name), copy=True)
image = image.run_commands(
    'nvcc -O3 -std=c++17 --fmad=false -arch=sm_90 -Xcompiler=-fPIC -shared '
    '/opt/cpu-gpu-compare/block_jacobi.cu /opt/cpu-gpu-compare/cuda_compare.cu '
    '-lcublas -lcusolver -o /opt/cpu-gpu-compare/libcuda_compare.so')
for name in ('experiment.json', 'inputs.npz'):
    image = image.add_local_file(PREPARED/name, str(REMOTE/name), copy=True)


@app.function(image=image, gpu='H100!:1', cpu=8, memory=32768, timeout=540,
              max_containers=1, min_containers=0, buffer_containers=0,
              scaledown_window=2, retries=0, volumes={'/results': volume})
def benchmark(run, deadline):
    import json
    import os
    import subprocess
    import sys
    import time
    from siesta_remag.attempts import prepare_attempt, complete_attempt, invocation_digest, package_evidence
    from siesta_remag.corpus import sha256
    from importlib.util import spec_from_file_location, module_from_spec

    spec = spec_from_file_location('comparison', REMOTE/'compare_cuda.py')
    comparison = module_from_spec(spec)
    spec.loader.exec_module(comparison)
    manifest = json.loads((REMOTE/'experiment.json').read_text())
    for source, expected in manifest['source_sha256'].items():
        if sha256(REMOTE/Path(source).name) != expected:
            raise ValueError('Prepared source changed: '+source)
    if sha256(REMOTE/'inputs.npz') != manifest['input_sha256']:
        raise ValueError('Prepared inputs changed')
    root = Path('/results')/run
    if root.exists():
        # No platform reschedule may consume a second comparison allowance.
        raise RuntimeError('Attempt already exists; recover its evidence without rerunning')
    library = REMOTE/'libcuda_compare.so'
    attempt, _ = prepare_attempt(root, invocation_digest(
        dict(deadline=deadline, manifest_sha256=sha256(REMOTE/'experiment.json')),
        manifest['source_sha256'], {str(library): sha256(library)}))
    comparison.save(root/'experiment.json', manifest)
    comparison.save(root/'allocation.json', dict(gpu='H100!:1', deadline_epoch=deadline,
        maximum_function_seconds=540, evidence_reserve_seconds=60, attempt_id=attempt['attempt_id']))
    volume.commit()
    remaining = min(480, deadline-time.time()-60)
    outcome = dict(returncode=None, reason=None)
    try:
        if remaining < 30:
            raise TimeoutError('Insufficient budget after startup')
        with (root/'stdout.log').open('w') as log:
            process = subprocess.run([sys.executable, str(REMOTE/'compare_cuda.py'), 'gpu',
                '--folder', str(REMOTE), '--output', str(root/'comparison.json'),
                '--library', str(library), '--oracle', str(REMOTE/'replay.py')],
                env=dict(os.environ, **comparison.SETTINGS), stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
        outcome['returncode'] = process.returncode
        outcome['reason'] = 'completed' if process.returncode == 0 else 'worker_failure'
    except (subprocess.TimeoutExpired, TimeoutError) as error:
        outcome.update(reason='budget_limit', error=str(error))
    finally:
        comparison.save(root/'outcome.json', outcome)
        # This marker records finished evidence collection, not scientific pass.
        complete_attempt(root)
        volume.commit()
    return package_evidence(root)


@app.local_entrypoint()
def main(budget_seconds: int = 0):
    import datetime
    import io
    import tarfile
    import time
    if not 120 <= budget_seconds <= 600:
        raise ValueError('Pass the explicitly approved budget (120..600 seconds); there is no default allowance')
    run = 'mac-cpu-remag-gpu-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    deadline = time.time()+budget_seconds
    call = benchmark.spawn(run, deadline)
    print('Run ID:', run, flush=True)
    print('Function call:', call.object_id, flush=True)
    print('Fixed deadline epoch:', deadline, flush=True)
    payload = call.get()
    target = ROOT/'results'/run
    target.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        archive.extractall(target, filter='data')
    print('Saved comparison:', target, flush=True)
