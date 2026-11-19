from types import SimpleNamespace

import pytest

from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save
from siesta_remag.research import relaxation_execution
from siesta_remag.research.execution import _review_prior_scf_failure


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    attempt = tmp_path/'prior'
    report = attempt/'collected/query/report.json'
    stage = report.parent/'dft-relax'
    stage.mkdir(parents=True)
    grant = dict(previous_app_id='ap-prior', authorization_id='another-composition')
    owner = dict(attempt='prior', status='held_native_review', allocation_stopped=True, app_id='ap-prior')
    save(report, dict(status='held_dft-relax'))
    grant['previous_scf_failure_report_sha256'] = sha256(report)
    save(stage/'execution-status.json', dict(returncode=-6, timed_out=False))
    (stage/'stderr.log').write_text('REMAG device SCF export: Final electronic state requires convergence\n')
    save(attempt/'receipt.json', dict(app_id='ap-prior', run_id='run-prior'))
    save(attempt/'authorization.json', dict(config=dict(authorization_id='prior'), config_sha256='prior-config'))
    save(attempt/'execution.json', {**owner, 'app_lifecycle': dict(state='stopped'),
         'config_sha256': 'prior-config', 'source_sha256': 'prior-source',
         'review': dict(report_sha256=sha256(report))})
    save(attempt/'campaign/structure.json', dict(geometry=dict(atomic_numbers=[26,26,27])))
    save(attempt/'campaign/query/request.json', {})
    save(attempt/'runtime.json', {})
    source = tmp_path/'next.json'
    save(source, dict(geometry=dict(atomic_numbers=[39,27,26])))
    reservations = [dict(id='run-prior', metadata=dict(app_id='ap-prior', receipt=str(attempt/'receipt.json')))]
    ledger = SimpleNamespace(snapshot=lambda: dict(reservations=reservations))
    reproduced = dict(healthy=False, native_status='held_dft-relax', source_binding=dict(source_sha256='prior-source'))
    monkeypatch.setattr(relaxation_execution, 'review_relaxation', lambda *a, **kw: reproduced)
    return tmp_path, owner, dict(config=grant, ledger=ledger), source


def test_review_allows_new_composition(evidence):
    assert _review_prior_scf_failure(*evidence) == 'prior/collected/query/report.json'


@pytest.mark.parametrize('change', ['supercell', 'segfault', 'other-abort', 'live', 'report', 'receipt'])
def test_independent_grant_fails_closed(evidence, change):
    root, owner, plan, source = evidence
    stage = root/'prior/collected/query/dft-relax'
    if change == 'supercell':
        save(source, dict(geometry=dict(atomic_numbers=[27,26,26,26,27,26])))
    elif change == 'segfault':
        save(stage/'execution-status.json', dict(returncode=-11, timed_out=False))
    elif change == 'other-abort':
        (stage/'stderr.log').write_text('aborted')
    elif change == 'live':
        owner['allocation_stopped'] = False
    elif change == 'report':
        plan['config']['previous_scf_failure_report_sha256'] = 'f' * 64
    else:
        path = root/'prior/receipt.json'
        save(path, {**load(path), 'run_id': 'different'})
    with pytest.raises(ValueError):
        _review_prior_scf_failure(*evidence)
