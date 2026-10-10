"""Validate the distinct production collinear and spinor population readers."""
import copy
import pytest
from siesta_remag.initial_state import record_errors


@pytest.mark.parametrize('spins', [1, 2, 4, 8])
def test_mulliken_requires_the_consumed_views_without_unused_raw_exports(spins):
    # NC layout from v89's completed fe2_noncollinear_gamma__tsde process.
    # That process was wrongly rejected for not exporting an unused raw total.
    reads = [0, 0, 0, 1, 2, 1, 2] if spins >= 4 else [1, 1, 1, 0, 1, 0, 2]
    components = min(spins, 4)
    sizes = [components*18, components*2, components, 90, 12, 5, 10]
    record = dict(token=1, epoch=1, atoms=2, rows=18, capacity=324, spins=spins,
        species=1, requests=3, status=0, density_upload_bytes=2592*spins,
        population_setup_bytes=2796, status_export_bytes=4, evaluated=True,
        closed=True, borrowed=True, released=True, reads=reads,
        export_bytes=[8*n*bool(count) for n, count in zip(sizes, reads)])
    cycles = [dict(epoch=1, terminal=dict(capacity=324))]
    eigenvalues = dict(nspin=spins, nbands=36 if spins >= 4 else 18)
    assert not record_errors([record], cycles, 2, eigenvalues, 1)
    reused=copy.deepcopy(record)
    reused.update(density_from_history=True,density_upload_bytes=0)
    assert record_errors([reused], cycles, 2, eigenvalues, 1)
    assert not record_errors([reused], cycles, 2, eigenvalues, 1, [dict(reused=True)])
    reused['density_upload_bytes']=2592*spins
    assert record_errors([reused], cycles, 2, eigenvalues, 1, [dict(reused=True)])
    for kind in (3, 4, 5, 6) if spins >= 4 else (0, 1, 2, 4, 6):
        missing = copy.deepcopy(record)
        missing['reads'][kind] = missing['export_bytes'][kind] = 0
        assert 'Initial population outputs differ from requests' in record_errors(
            [missing], cycles, 2, eigenvalues, 1)
