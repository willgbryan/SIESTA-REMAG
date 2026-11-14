from concurrent.futures import ThreadPoolExecutor
import pytest
from siesta_remag.research.budget import BudgetLedger, reservation_cents


def test_round_up_and_nonfinite_reservations():
    assert reservation_cents(2400, 10) == 667
    assert reservation_cents(7200, 10) == 2000
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
