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

MAX_CAMPAIGN_SECONDS = 21600


def reservation_cents(timeout_seconds, hourly_usd):
    if type(timeout_seconds) is not int or not 120 <= timeout_seconds <= MAX_CAMPAIGN_SECONDS:
        raise ValueError(f'Campaign timeout must be 120..{MAX_CAMPAIGN_SECONDS} seconds')
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
        cap = state['cap_cents']; ids = set()
        for extension in state.get('authorization_extensions', []):
            if (not isinstance(extension.get('id'), str) or not extension['id'] or extension['id'] in ids
                or type(extension.get('additional_cents')) is not int or extension['additional_cents'] <= 0
                or not isinstance(extension.get('authorization'), str) or not extension['authorization'].strip()
                or extension.get('previous_cap_cents') != cap):
                raise ValueError('Invalid authorization extension')
            cap += extension['additional_cents']; ids.add(extension['id'])
            if extension.get('new_cap_cents') != cap: raise ValueError('Broken authorization history')
        stages = {}
        for stage in state.get('stages', []):
            if (not isinstance(stage.get('id'), str) or not stage['id'] or stage['id'] in stages
                or type(stage.get('cap_cents')) is not int or not 0 < stage['cap_cents'] <= cap
                or not isinstance(stage.get('authorization'), str) or not stage['authorization'].strip()):
                raise ValueError('Invalid budget stage')
            stages[stage['id']] = stage['cap_cents']
        ids = set(); stage_totals = {key: 0 for key in stages}
        for row in state['reservations']:
            if row['id'] in ids or type(row['cents']) is not int or row['cents'] <= 0:
                raise ValueError('Invalid or duplicate budget reservation')
            ids.add(row['id'])
            if 'stage' in row:
                if row['stage'] not in stages: raise ValueError('Unknown reservation stage')
                stage_totals[row['stage']] += row['cents']
        if any(stage_totals[key] > limit for key, limit in stages.items()):
            raise ValueError('Budget stage history exceeds its cap')
        if sum(r['cents'] for r in state['reservations']) > cap:
            raise ValueError('Budget history exceeds its cap')
        return state

    @staticmethod
    def _cap(state):
        return state['cap_cents'] + sum(e['additional_cents'] for e in state.get('authorization_extensions', []))

    @classmethod
    def _remaining(cls, state, stage=None):
        left = cls._cap(state)-sum(r['cents'] for r in state['reservations'])
        if stage is not None:
            match = next((s for s in state.get('stages', []) if s['id'] == stage), None)
            if match is None: raise ValueError('Unknown budget stage')
            left = min(left, match['cap_cents']-sum(r['cents'] for r in state['reservations'] if r.get('stage') == stage))
        return left

    def effective_cap_cents(self):
        with self._lock():
            return self._cap(self._read())

    def authorize_additional(self, authorization_id, additional_cents, *, authorization):
        """Record an explicit new owner authorization without rewriting the base cap."""
        if (not isinstance(authorization_id, str) or not authorization_id
            or type(additional_cents) is not int or additional_cents <= 0
            or not isinstance(authorization, str) or not authorization.strip()):
            raise ValueError('Named explicit authorization and positive additional cents required')
        with self._lock():
            s = self._read()
            old = next((e for e in s.get('authorization_extensions', []) if e['id'] == authorization_id), None)
            if old:
                if old['additional_cents'] != additional_cents or old['authorization'] != authorization:
                    raise ValueError('An authorization cannot be changed')
                return old
            cap = self._cap(s)
            row = dict(id=authorization_id, additional_cents=additional_cents, authorization=authorization,
                       previous_cap_cents=cap, new_cap_cents=cap+additional_cents,
                       recorded_at=datetime.now(timezone.utc).isoformat())
            s.setdefault('authorization_extensions', []).append(row); save(self.path, s)
            return row

    def define_stage(self, stage_id, cap_cents, *, authorization):
        """Add an immutable sub-limit; this never increases the overall authorization."""
        if (not isinstance(stage_id, str) or not stage_id or type(cap_cents) is not int or cap_cents <= 0
            or not isinstance(authorization, str) or not authorization.strip()):
            raise ValueError('Named positive stage cap and authorization required')
        with self._lock():
            s = self._read()
            if cap_cents > self._cap(s): raise ValueError('Stage exceeds overall authorization')
            old = next((r for r in s.get('stages', []) if r['id'] == stage_id), None)
            if old:
                if old['cap_cents'] != cap_cents or old['authorization'] != authorization:
                    raise ValueError('A stage limit cannot be changed')
                return old
            row = dict(id=stage_id, cap_cents=cap_cents, authorization=authorization,
                       recorded_at=datetime.now(timezone.utc).isoformat())
            s.setdefault('stages', []).append(row); save(self.path, s)
            return row

    def remaining_cents(self, stage=None):
        with self._lock():
            s = self._read()
            return self._remaining(s, stage)

    def snapshot(self):
        """Read a consistent reservation history without changing it."""
        with self._lock():
            return self._read()

    def reserve(self, reservation_id, cents, *, metadata, stage=None):
        if not isinstance(reservation_id, str) or not reservation_id or type(cents) is not int or cents <= 0:
            raise ValueError('Named positive whole-cent reservation required')
        with self._lock():
            s = self._read()
            old = next((r for r in s['reservations'] if r['id'] == reservation_id), None)
            if old:
                if old['cents'] != cents or old['metadata'] != metadata or old.get('stage') != stage:
                    raise ValueError('A reservation cannot be changed')
                return old
            if s.get('stages') and stage is None: raise ValueError('A budget stage is required for new reservations')
            if cents > self._remaining(s, stage):
                raise ValueError('Cumulative research budget exhausted')
            row = dict(id=reservation_id, cents=cents, metadata=metadata,
                       recorded_at=datetime.now(timezone.utc).isoformat(), events=[])
            if stage is not None: row['stage'] = stage
            s['reservations'].append(row); save(self.path, s)
            return row

    def record(self, reservation_id, event):
        with self._lock():
            s = self._read()
            row = next((r for r in s['reservations'] if r['id'] == reservation_id), None)
            if row is None: raise ValueError('Unknown reservation')
            row['events'].append(dict(recorded_at=datetime.now(timezone.utc).isoformat(), data=event))
            save(self.path, s)
