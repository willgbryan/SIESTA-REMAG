"""One bounded physics campaign, launched through scripts/run_modal.py only."""
from pathlib import Path
import json
import math
import os
import sys
import uuid

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src') if modal.is_local() else '/opt/research/src')
from siesta_remag.discovery.common import load, save, digest

if modal.is_local():
    campaign = Path(os.environ['REMAG_RESEARCH_CAMPAIGN']).resolve()
    runtime_path = Path(os.environ['REMAG_RESEARCH_RUNTIME']).resolve()
    image_id = os.environ['REMAG_RESEARCH_IMAGE']
    policy = load(campaign/'execution.json')
    from siesta_remag.research.budget import BudgetLedger, reservation_cents
    budget = BudgetLedger(os.environ['REMAG_RESEARCH_BUDGET_LEDGER'])
    reserved_cents = reservation_cents(policy['timeout_seconds'], policy['hourly_reservation_usd'])
    if budget.remaining_cents() < reserved_cents:
        raise ValueError('Insufficient cumulative budget before image preparation')
    volume_name = policy.get('volume', 'siesta-remag-results')
    diagnostic_axes = policy.get('diagnostic_axes', ['z', 'x', 'y'])
    if not isinstance(diagnostic_axes, list) or len(set(diagnostic_axes)) != len(diagnostic_axes) or any(a not in ('x','y','z') for a in diagnostic_axes):
        raise ValueError('Diagnostic axes must be a unique subset of x,y,z')
    timeout = policy['timeout_seconds']
    if type(timeout) is not int or not 120 <= timeout <= 7200:
        raise ValueError('Campaign timeout must be 120..7200 seconds')
    # Conservative allocation reservation, not a claim about a billing invoice.
    if any(type(policy[k]) not in (int, float) or not math.isfinite(policy[k]) or policy[k] <= 0
           for k in ['hourly_reservation_usd', 'budget_usd']):
        raise ValueError('Positive finite allocation rate and budget required')
    if timeout/3600*policy['hourly_reservation_usd'] > policy['budget_usd'] or policy['budget_usd'] > 100:
        raise ValueError('Campaign exceeds its declared allocation budget')
    from siesta_remag.research.experiments import verify_experiment
    for name in policy['experiments']:
        if not (campaign/name).resolve().is_relative_to(campaign): raise ValueError('Unsafe experiment path')
        verify_experiment(campaign/name)
    image = (modal.Image.from_id(image_id)
        .add_local_dir(ROOT/'src', '/opt/research/src', copy=True, ignore=['__pycache__'])
        .add_local_dir(campaign, '/opt/research/campaign', copy=True)
        .add_local_file(runtime_path, '/opt/research/runtime.json', copy=True))
else:
    image, timeout = None, 7200
    volume_name = load('/opt/research/campaign/execution.json').get('volume', 'siesta-remag-results')

app = modal.App('siesta-remag-single-gpu')
volume = modal.Volume.from_name(volume_name, create_if_missing=True)


@app.function(image=image, gpu='A100-80GB:1', cpu=8, memory=98304, timeout=timeout,
              max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2,
              retries=0, volumes={'/results': volume})
def calculate(run_id):
    import time
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[key] = '1'
    from siesta_remag.research.experiments import run_experiment
    from siesta_remag.research.electronic import spectrum_from_run
    from siesta_remag.discovery.collection import pack
    from siesta_remag.corpus import sha256
    root = Path('/results')/run_id
    root.mkdir(exist_ok=False)
    policy = load('/opt/research/campaign/execution.json')
    runtime = load('/opt/research/runtime.json')
    for key in list(os.environ):
        if key.startswith('REMAG_'): os.environ.pop(key)
    started = time.monotonic()
    report = dict(status='running', invocation_sha256=digest(policy), policy=policy, experiments=[],
                  physical_accuracy_qualified=False, billed_cost_usd=None)
    def flush():
        report['allocation_elapsed_seconds'] = time.monotonic()-started
        report['reserved_cost_bound_usd'] = policy['timeout_seconds']/3600*policy['hourly_reservation_usd']
        save(root/'report.json', report); volume.commit()
    flush()
    try:
        for name in policy['experiments']:
            request = load(Path('/opt/research/campaign')/name/'experiment.json')
            if time.monotonic()-started + request['settings']['total_dft_seconds'] + 60 > policy['timeout_seconds']:
                report['status'] = 'held_budget'; break
            result = run_experiment(Path('/opt/research/campaign')/name, root/name, runtime['executable'],
                executable_sha256=runtime['executable_sha256'], runtime_env=runtime.get('environment', {}),
                threads=runtime.get('threads', 1), checkpoint=volume.commit)
            report['experiments'].append(dict(id=name, status=result['status'], report_sha256=sha256(root/name/'report.json')))
            if result['status'] != 'measured':
                report['status'] = 'held_experiment'; flush(); break
            for axis in policy.get('diagnostic_axes', ['z', 'x', 'y']):
                try:
                    spectrum_from_run(root/name/('soc-'+axis), root/name/('electronic-'+axis))
                except Exception as exc:
                    save(root/name/('electronic-'+axis+'-error.json'), dict(status='held_diagnostic', reason=str(exc)))
            flush()
        else:
            report['status'] = 'measured'
    except Exception as exc:
        report.update(status='held_error', reason=str(exc))
    finally:
        flush()
    return pack(root, include_arrays=True)


@app.local_entrypoint()
def main(receipt: str, output: str):
    from siesta_remag.discovery.collection import collect
    path = Path(receipt)
    if path.exists() or Path(output).exists():
        raise ValueError('Existing submission/output: collect it instead of resubmitting')
    run_id = 'physics-'+uuid.uuid4().hex
    budget.reserve(run_id, reserved_cents, metadata=dict(policy_sha256=digest(policy), app_id=app.app_id,
                   receipt=str(path.resolve()), campaign=str(campaign)))
    save(path, dict(status='submitting', run_id=run_id, app_id=app.app_id, image=image_id,
                   volume=volume_name, reserved_cents=reserved_cents))
    call = calculate.spawn(run_id)
    save(path, dict(status='running', run_id=run_id, app_id=app.app_id, image=image_id,
                   volume=volume_name, reserved_cents=reserved_cents,
                   function_call_id=call.object_id, output=str(Path(output).resolve())))
    budget.record(run_id, load(path))
    print('Submission receipt:', path, flush=True)
    result = collect(path, output, timeout=timeout+60)
    budget.record(run_id, load(path))
    print(json.dumps(dict(status=result['status'], report=str(Path(output)/'report.json'))), flush=True)
