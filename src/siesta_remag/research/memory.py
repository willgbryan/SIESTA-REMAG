"""Small transactional research notebook with pre-observation predictions."""
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np


class Notebook:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, kind TEXT NOT NULL, parent TEXT, recorded_at TEXT NOT NULL, payload TEXT NOT NULL)')
        self.db.commit()

    def close(self):
        self.db.close()

    def _append(self, kind, data, parent=None):
        encoded = json.dumps(data, allow_nan=False, sort_keys=True)
        if len(encoded) > 65536:
            raise ValueError('Store large numerical arrays as artifacts')
        uid = uuid4().hex
        self.db.execute('INSERT INTO events VALUES (?,?,?,?,?)',
            (uid, kind, parent, datetime.now(timezone.utc).isoformat(), encoded))
        return uid

    def get(self, uid):
        row = self.db.execute('SELECT id,kind,parent,recorded_at,payload FROM events WHERE id=?', (uid,)).fetchone()
        if row is None: raise ValueError('Unknown notebook event')
        return dict(id=row[0], kind=row[1], parent=row[2], recorded_at=row[3], payload=json.loads(row[4]))

    def record_hypothesis(self, statement, *, predictions, alternatives):
        if not statement or not predictions or not alternatives:
            raise ValueError('A hypothesis needs testable predictions and alternatives')
        with self.db:
            return self._append('hypothesis', dict(statement=statement, predictions=predictions, alternatives=alternatives))

    def propose_experiment(self, hypothesis_id, *, structure_sha256, action, rationale):
        if self.get(hypothesis_id)['kind'] != 'hypothesis' or not action or not rationale:
            raise ValueError('Experiment requires a hypothesis, action and rationale')
        if not isinstance(structure_sha256, str) or len(structure_sha256) != 64 or any(c not in '0123456789abcdef' for c in structure_sha256):
            raise ValueError('An exact source SHA-256 is required')
        with self.db:
            return self._append('experiment', dict(structure_sha256=structure_sha256, action=action, rationale=rationale), hypothesis_id)

    def record_prediction(self, experiment_id, *, observable, value, unit, uncertainty=None):
        if type(value) not in (int, float) or not np.isfinite(value) or not observable or not unit:
            raise ValueError('A finite prediction with observable and unit is required')
        if uncertainty is not None and (type(uncertainty) not in (int, float) or not np.isfinite(uncertainty) or uncertainty < 0):
            raise ValueError('Prediction uncertainty must be finite and nonnegative')
        # Serialize the read/write so concurrent outcome recording cannot race it.
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if self.get(experiment_id)['kind'] != 'experiment': raise ValueError('Expected an experiment')
            if self.db.execute("SELECT 1 FROM events WHERE parent=? AND kind IN ('prediction','outcome')", (experiment_id,)).fetchone():
                raise ValueError('Prediction is already recorded or the outcome is known')
            return self._append('prediction', dict(observable=observable, value=value, unit=unit, uncertainty=uncertainty), experiment_id)

    def record_outcome(self, experiment_id, *, observable, value, unit, report_sha256, status):
        if status not in ('measured', 'failed', 'inconclusive'):
            raise ValueError('Retain measured, failed or inconclusive outcomes')
        if value is not None and (type(value) not in (int, float) or not np.isfinite(value)):
            raise ValueError('Outcome must be finite or unknown')
        if status == 'measured' and value is None:
            raise ValueError('Measured outcome needs a value')
        if not isinstance(report_sha256, str) or len(report_sha256) != 64 or any(c not in '0123456789abcdef' for c in report_sha256):
            raise ValueError('Outcome requires an evidence SHA-256')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if self.get(experiment_id)['kind'] != 'experiment': raise ValueError('Expected an experiment')
            if self.db.execute("SELECT 1 FROM events WHERE parent=? AND kind='outcome'", (experiment_id,)).fetchone():
                raise ValueError('Outcome is immutable; record a new experiment')
            return self._append('outcome', dict(observable=observable, value=value, unit=unit, report_sha256=report_sha256, status=status), experiment_id)

    def evaluate(self, experiment_id):
        events = [self.get(row[0]) for row in self.db.execute('SELECT id FROM events WHERE parent=?', (experiment_id,))]
        by_kind = {e['kind']: e['payload'] for e in events}
        prediction, outcome = by_kind.get('prediction'), by_kind.get('outcome')
        if not prediction or not outcome or outcome['status'] != 'measured':
            return dict(status='inconclusive', mechanism_verified=False)
        if (prediction['observable'], prediction['unit']) != (outcome['observable'], outcome['unit']):
            raise ValueError('Prediction and observation units/quantities differ')
        error = outcome['value']-prediction['value']
        return dict(status='evaluated', error=error, absolute_error=abs(error), unit=outcome['unit'],
            within_predicted_uncertainty=None if prediction['uncertainty'] is None else abs(error) <= prediction['uncertainty'],
            mechanism_verified=False)

    def retrieve(self, query='', limit=50):
        if type(limit) is not int or not 1 <= limit <= 200: raise ValueError('Limit must be 1..200')
        return [self.get(r[0]) for r in self.db.execute('SELECT id FROM events WHERE instr(lower(payload),lower(?))>0 ORDER BY rowid DESC LIMIT ?', (query, limit))]

    def record_analysis(self, parent_id, *, conclusion, report_sha256, status='inconclusive'):
        """Append an evidence-bound interpretation without changing predictions.

        This can retain replication or convergence findings. It is an analyst's
        interpretation, not automatic verification of a physical mechanism.
        """
        if self.get(parent_id)['kind'] not in ('hypothesis', 'experiment') or not conclusion:
            raise ValueError('Analysis requires an existing hypothesis/experiment and conclusion')
        if status not in ('evaluated', 'inconclusive'):
            raise ValueError('Analysis status must be evaluated or inconclusive')
        if not isinstance(report_sha256, str) or len(report_sha256) != 64 or any(c not in '0123456789abcdef' for c in report_sha256):
            raise ValueError('Analysis requires an evidence SHA-256')
        with self.db:
            return self._append('analysis', dict(conclusion=conclusion, report_sha256=report_sha256,
                status=status, mechanism_verified=False), parent_id)


def evaluate_holdout(training, heldout, *, agent_predictions):
    """Compare preregistered predictions to linear and interpolation baselines.

    The caller must obtain predictions before revealing held-out observations;
    use Notebook to record that ordering. This function cannot certify blinding.
    """
    train, test, predicted = np.asarray(training, float), np.asarray(heldout, float), np.asarray(agent_predictions, float)
    if (train.ndim != 2 or train.shape[1] != 2 or len(train) < 2 or test.ndim != 2 or test.shape[1] != 2
        or len(test) < 1 or predicted.shape != (len(test),) or not all(np.isfinite(x).all() for x in (train, test, predicted))):
        raise ValueError('Finite train/heldout (N,2) samples and matching predictions required')
    train = train[np.argsort(train[:, 0])]
    if len(set(train[:, 0])) != len(train) or set(train[:, 0]) & set(test[:, 0]):
        raise ValueError('Training abscissae must be distinct and disjoint from holdout')
    if np.any(test[:, 0] < train[0, 0]) or np.any(test[:, 0] > train[-1, 0]):
        raise ValueError('Interpolation baseline requires bracketed held-out samples')
    linear = np.polyval(np.polyfit(train[:, 0], train[:, 1], 1), test[:, 0])
    interpolation = np.interp(test[:, 0], train[:, 0], train[:, 1])
    return {name: dict(mae=float(np.mean(abs(p-test[:, 1]))), rmse=float(np.sqrt(np.mean((p-test[:, 1])**2))))
            for name, p in [('agent', predicted), ('linear', linear), ('interpolation', interpolation)]}
