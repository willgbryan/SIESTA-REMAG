from concurrent.futures import ThreadPoolExecutor
import json
import pytest
from siesta_remag.research.budget import BudgetLedger, reservation_cents


def test_round_up_and_nonfinite_reservations():
    assert reservation_cents(2400, 10) == 667
    assert reservation_cents(7200, 10) == 2000
    assert reservation_cents(10800, 10) == 3000
    assert reservation_cents(21600, 10) == 6000
    for duration in [119, 21601, 3600.0, True]:
        with pytest.raises(ValueError): reservation_cents(duration, 10)
    for rate in [float('nan'), float('inf'), -1, True]:
        with pytest.raises(ValueError): reservation_cents(3600, rate)


def test_completed_jobs_do_not_silently_release_budget(tmp_path):
    b = BudgetLedger(tmp_path/'budget.json')
    b.create(10000, authorization='fixture')
    b.reserve('prior1', 2000, metadata={})
    b.reserve('prior2', 667, metadata={})
    b.record('prior1', {'status': 'collected'})
    assert b.remaining_cents() == 7333
    b.reserve('prior1', 2000, metadata={})
    assert b.remaining_cents() == 7333
    with pytest.raises(ValueError): b.reserve('prior1', 1000, metadata={})
    with pytest.raises(ValueError): b.reserve('excess', 7334, metadata={})
    with pytest.raises(ValueError): b.create(10000, authorization='reset')


def test_concurrent_reservations_cannot_overspend(tmp_path):
    path = tmp_path/'budget.json';BudgetLedger(path).create(10000, authorization='fixture')
    def submit(i):
        try:
            BudgetLedger(path).reserve(str(i), 6000, metadata={})
            return True
        except ValueError: return False
    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(submit, [0, 1])) == 1
    assert BudgetLedger(path).remaining_cents() == 4000


def test_authorization_extension_preserves_history_and_is_idempotent(tmp_path):
    path = tmp_path/'budget.json'; b = BudgetLedger(path)
    b.create(10000, authorization='original $100')
    b.reserve('old', 9667, metadata={'prior': True})
    b.record('old', {'status': 'collected', 'actual_invoice_usd': None})
    before = json.loads(path.read_text())
    b.authorize_additional('extension', 25000, authorization='owner adds $250')
    after = json.loads(path.read_text())
    assert {key: after[key] for key in before} == before
    assert b.effective_cap_cents() == 35000
    assert b.remaining_cents() == 25333
    contents = path.read_bytes()
    b.authorize_additional('extension', 25000, authorization='owner adds $250')
    assert path.read_bytes() == contents
    for cents, authorization in [(30000, 'owner adds $250'), (25000, 'changed')]:
        with pytest.raises(ValueError): b.authorize_additional('extension', cents, authorization=authorization)
    b.reserve('new', 25333, metadata={})
    with pytest.raises(ValueError): b.reserve('over', 1, metadata={})
    assert b.remaining_cents() == 0


def test_stage_limits_cannot_be_bypassed_by_otherwise_available_funds(tmp_path):
    path = tmp_path/'budget.json'; b = BudgetLedger(path)
    b.create(10000, authorization='original')
    b.authorize_additional('additional', 25000, authorization='owner')
    b.define_stage('mesh', 5000, authorization='mesh first')
    b.define_stage('mesh', 5000, authorization='mesh first')
    with pytest.raises(ValueError): b.define_stage('mesh', 6000, authorization='mesh first')
    with pytest.raises(ValueError): b.reserve('unscoped', 1, metadata={})
    with pytest.raises(ValueError): b.reserve('unknown', 1, metadata={}, stage='missing')
    b.reserve('k36', 2000, metadata={}, stage='mesh')
    b.record('k36', {'status': 'collected'})
    b.reserve('k40', 3000, metadata={}, stage='mesh')
    b.reserve('k40', 3000, metadata={}, stage='mesh')
    assert b.remaining_cents('mesh') == 0
    assert b.remaining_cents() == 30000
    with pytest.raises(ValueError): b.reserve('k44', 1, metadata={}, stage='mesh')
    with pytest.raises(ValueError): b.reserve('k40', 3000, metadata={})


def test_concurrent_stage_reservations_and_global_limit(tmp_path):
    path = tmp_path/'budget.json'; b = BudgetLedger(path)
    b.create(10000, authorization='owner')
    b.reserve('prior', 6000, metadata={})
    b.define_stage('mesh', 5000, authorization='owner')
    def submit(i):
        try:
            BudgetLedger(path).reserve(str(i), 3000, metadata={}, stage='mesh')
            return True
        except ValueError: return False
    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(submit, [0, 1])) == 1
    assert b.remaining_cents('mesh') == 1000


def test_broken_extension_and_stage_history_are_rejected(tmp_path):
    path = tmp_path/'budget.json'; b = BudgetLedger(path)
    b.create(10000, authorization='owner')
    for cents, auth in [(0, 'owner'), (-1, 'owner'), (True, 'owner'), (25000, ' ')]:
        with pytest.raises(ValueError): b.authorize_additional('extra', cents, authorization=auth)
    b.authorize_additional('extra', 25000, authorization='owner')
    valid = json.loads(path.read_text())
    for field, value in [('previous_cap_cents', 0), ('new_cap_cents', 40000)]:
        changed = json.loads(json.dumps(valid)); changed['authorization_extensions'][0][field] = value
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError): b.remaining_cents()
    path.write_text(json.dumps(valid))
    b.define_stage('mesh', 5000, authorization='owner')
    b.reserve('run', 5000, metadata={}, stage='mesh')
    changed = json.loads(path.read_text()); changed['reservations'][0]['cents'] = 5001
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError): b.remaining_cents()
