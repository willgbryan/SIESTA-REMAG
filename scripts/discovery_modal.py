"""Launch only through scripts/run_modal.py; cached image, one finite A100."""
from pathlib import Path
import os
import json
import sys
import uuid

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src') if modal.is_local() else '/opt/discovery/src')
from siesta_remag.discovery.common import load, save
from siesta_remag.discovery.pipeline import verify_bundle

if modal.is_local():
    bundle = Path(os.environ['REMAG_DISCOVERY_BUNDLE']).resolve()
    runtime_path = Path(os.environ['REMAG_DISCOVERY_RUNTIME']).resolve()
    image_id = os.environ['REMAG_DISCOVERY_IMAGE']
    verify_bundle(bundle)
    settings = load(bundle/'request.json')['settings']
    image = (modal.Image.from_id(image_id)
             # Small CPU-only cached dependency layer; no compiler or GPU build.
             .pip_install('ase==3.26.0')
             .add_local_dir(ROOT/'src', '/opt/discovery/src', copy=True, ignore=['__pycache__'])
             .add_local_dir(bundle, '/opt/discovery/bundle', copy=True, ignore=['*.traj', '*.lock'])
             .add_local_file(runtime_path, '/opt/discovery/runtime.json', copy=True))
    timeout = int(settings['total_dft_seconds']) + 120
else:
    image = None
    timeout = 86400  # Remote decorator is not used for scheduling; local policy above is authoritative.

app = modal.App('siesta-remag-single-gpu')
volume = modal.Volume.from_name('siesta-remag-results', create_if_missing=True)


@app.function(image=image, gpu='A100-80GB:1', cpu=8, memory=98304, timeout=timeout,
              max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2,
              retries=0, volumes={'/results': volume})
def qualify(run_id):
    sys.path.insert(0, '/opt/discovery/src')
    from siesta_remag.discovery.pipeline import run_dft
    from siesta_remag.discovery.collection import pack
    runtime = load('/opt/discovery/runtime.json')
    # Clear inherited experimental flags; the supplied runtime is the full intended set.
    for key in list(os.environ):
        if key.startswith('REMAG_'): os.environ.pop(key)
    destination = Path('/results')/run_id
    run_dft('/opt/discovery/bundle', destination, runtime['executable'],
            executable_sha256=runtime['executable_sha256'], runtime_env=runtime.get('environment', {}),
            threads=runtime.get('threads', 1), checkpoint=volume.commit)
    payload = pack(destination)
    volume.commit()
    return payload


@app.local_entrypoint()
def main(receipt: str, output: str):
    from siesta_remag.discovery.collection import collect
    path = Path(receipt).resolve()
    if Path(output).exists():
        raise RuntimeError('Output already exists; inspect it before launching another calculation.')
    if path.exists():
        raise RuntimeError('Submission receipt already exists. Use remag-discover collect; never resubmit to recover.')
    run_id = 'discovery-'+uuid.uuid4().hex
    # Persist intent before spawn. Ambiguous submissions must be inspected, not repeated.
    save(path, dict(status='submitting', run_id=run_id, app_id=app.app_id, bundle=str(bundle), image=image_id))
    call = qualify.spawn(run_id)
    save(path, dict(status='running', run_id=run_id, app_id=app.app_id, function_call_id=call.object_id,
                    bundle=str(bundle), image=image_id, output=str(Path(output).resolve())))
    print('Submission receipt:', path, flush=True)
    report = collect(path, output, timeout=timeout+60)
    print(json.dumps(dict(status=report['status'], report=str(Path(output).resolve()/'report.json'))), flush=True)
