"""Append-only, saved-only recovery of a stopped relaxation review error.

No solver, allocation, ownership or ledger effects. Original execution and
collection remain immutable. Consumers independently replay current validators.
"""
from datetime import datetime, timezone
from pathlib import Path
from ..corpus import sha256
from ..discovery.common import load
from ..discovery.pipeline import lock
from .relaxation_execution import review_relaxation

NAME = 'review-recovery.json'
FORMAT = 'REMAG-RELAXATION-REVIEW-RECOVERY/1'
BOUND = ('execution.json', 'authorization.json', 'runtime.json', 'receipt.json', 'preflight.json',
         'campaign/execution.json', 'campaign/structure.json', 'collected/report.json',
         'collected/query/report.json', 'collected/archive-manifest.json')
CORE = ('healthy', 'native_status', 'report_sha256', 'source_binding', 'observations', 'screen', 'relaxation')


def _replay(attempt):
    attempt=Path(attempt);execution=load(attempt/'execution.json')
    if (execution.get('format')!='REMAG-RELAXATION-EXECUTION/1' or execution.get('status')!='held_review' or
            execution.get('allocation_stopped') is not True or
            execution.get('app_lifecycle',{}).get('state','').lower()!='stopped' or
            execution.get('app_id')!=load(attempt/'receipt.json').get('app_id')):
        raise ValueError('Saved review recovery requires a stopped relaxation review error')
    authorization=load(attempt/'authorization.json')
    if authorization['config']['runtime_sha256']!=sha256(attempt/'runtime.json'):
        raise ValueError('Recovered runtime differs from authorization')
    review=review_relaxation(attempt,load(attempt/'campaign/query/request.json'),load(attempt/'runtime.json'),require_current_code=False)
    if (execution['source_sha256']!=review['source_binding']['source_sha256'] or
            execution['request_sha256']!=sha256(attempt/'campaign/query/request.json')):
        raise ValueError('Recovered source/request differs from original execution')
    return execution,review


def verified_execution(attempt):
    """Return the effective execution; preserve and bind its historical status."""
    attempt=Path(attempt);original=load(attempt/'execution.json')
    if original.get('status')!='held_review' or not (attempt/NAME).exists():return original
    row=load(attempt/NAME);original,review=_replay(attempt)
    if (row.get('format')!=FORMAT or row.get('files_sha256')!={n:sha256(attempt/n) for n in BOUND} or
            row.get('status')!=('measured' if review['healthy'] else 'held_native_review') or
            row.get('review')!={k:review[k] for k in CORE}):
        raise ValueError('Saved review recovery no longer reproduces')
    elapsed=(datetime.fromisoformat(row['recorded_at'])-datetime.fromisoformat(original['requested_at'])).total_seconds()
    if elapsed<original['request_through_review_seconds'] or row['request_through_recovery_seconds']!=elapsed:
        raise ValueError('Saved review recovery timing differs')
    return dict(original,status=row['status'],review=review,
                usable_result_seconds=elapsed if review['healthy'] else None,
                original_status=original['status'],review_recovery_sha256=sha256(attempt/NAME))


def recover_saved_review(attempt, *, reason):
    """Operator-only local replay. Never overwrites an earlier review or result."""
    attempt=Path(attempt).resolve()
    if not isinstance(reason,str) or not 20<=len(reason)<=8000:raise ValueError('Explicit saved-review diagnosis required')
    with lock(attempt/'.review-recovery.lock'):
        if (attempt/NAME).exists():return verified_execution(attempt)
        original,review=_replay(attempt)
        now=datetime.now(timezone.utc).isoformat()
        elapsed=(datetime.fromisoformat(now)-datetime.fromisoformat(original['requested_at'])).total_seconds()
        if elapsed<original['request_through_review_seconds']:raise ValueError('Recovery precedes original review')
        row=dict(format=FORMAT,recorded_at=now,reason=reason,
            status='measured' if review['healthy'] else 'held_native_review',
            files_sha256={n:sha256(attempt/n) for n in BOUND},review={k:review[k] for k in CORE},
            verification=review['verification'],request_through_recovery_seconds=elapsed,
            native_executions=0,reservations=0,original_status=original['status'],
            validator_sha256={n:sha256(Path(__file__).parent/n) for n in
                              ('relaxation_recovery.py','relaxation_execution.py','input_reproduction.py')})
        # The lock serializes writers, and the exclusive creation prevents replacement.
        import json
        with (attempt/NAME).open('x') as stream:json.dump(row,stream,indent=2);stream.write('\n')
        return verified_execution(attempt)
