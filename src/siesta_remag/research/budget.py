"""Cumulative reservations against an explicitly authorized research budget.

Reservations remain charged to the cap after completion or an uncertain launch.
This ledger does not estimate invoices or automatically credit unused time.
"""
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import fcntl
import math
from pathlib import Path
from datetime import datetime, timezone

from ..discovery.common import load, save


def reservation_cents(timeout_seconds, hourly_usd):
    if type(timeout_seconds) is not int or not 120 <= timeout_seconds <= 7200:
        raise ValueError('Campaign timeout must be 120..7200 seconds')
    if type(hourly_usd) not in (int, float) or not math.isfinite(hourly_usd) or hourly_usd <= 0:
        raise ValueError('Positive finite reservation rate required')
    return int((Decimal(timeout_seconds)*Decimal(str(hourly_usd))*100/3600).to_integral_value(rounding=ROUND_CEILING))


class BudgetLedger:
    def __init__(self, path):
        self.path = Path(path).resolve()

    @contextmanager
    def _lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix+'.lock').open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def create(self, cap_cents, *, authorization):
        if type(cap_cents) is not int or not 0 < cap_cents <= 10000 or not authorization:
            raise ValueError('Explicit authorization and a positive cap up to $100 required')
        with self._lock():
            if self.path.exists(): raise ValueError('Budget already exists; do not reset its history')
            save(self.path, dict(format='REMAG-BUDGET/1', cap_cents=cap_cents,
                                 authorization=authorization, reservations=[]))

    def _read(self):
        state = load(self.path)
        if (state.get('format') != 'REMAG-BUDGET/1' or type(state.get('cap_cents')) is not int
            or not 0 < state['cap_cents'] <= 10000):
            raise ValueError('Invalid budget ledger')
        ids = set()
        for row in state['reservations']:
            if row['id'] in ids or type(row['cents']) is not int or row['cents'] <= 0:
                raise ValueError('Invalid or duplicate budget reservation')
            ids.add(row['id'])
        if sum(r['cents'] for r in state['reservations']) > state['cap_cents']:
            raise ValueError('Budget history exceeds its cap')
        return state

    def remaining_cents(self):
        with self._lock():
            s = self._read()
            return s['cap_cents']-sum(r['cents'] for r in s['reservations'])

    def reserve(self, reservation_id, cents, *, metadata):
        if not isinstance(reservation_id, str) or not reservation_id or type(cents) is not int or cents <= 0:
            raise ValueError('Named positive whole-cent reservation required')
        with self._lock():
            s = self._read()
            old = next((r for r in s['reservations'] if r['id'] == reservation_id), None)
            if old:
                if old['cents'] != cents or old['metadata'] != metadata:
                    raise ValueError('A reservation cannot be changed')
                return old
            if sum(r['cents'] for r in s['reservations'])+cents > s['cap_cents']:
                raise ValueError('Cumulative research budget exhausted')
            row = dict(id=reservation_id, cents=cents, metadata=metadata,
                       recorded_at=datetime.now(timezone.utc).isoformat(), events=[])
            s['reservations'].append(row); save(self.path, s)
            return row

    def record(self, reservation_id, event):
        with self._lock():
            s = self._read()
            row = next((r for r in s['reservations'] if r['id'] == reservation_id), None)
            if row is None: raise ValueError('Unknown reservation')
            row['events'].append(dict(recorded_at=datetime.now(timezone.utc).isoformat(), data=event))
            save(self.path, s)
