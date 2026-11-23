"""Portable review of retained moments, without changing branch acceptance."""
from copy import deepcopy

import pytest

from siesta_remag.discovery.config import Settings
from siesta_remag.research.relaxation_execution import verify_branch_reproduction


def check(saved, replayed, settings=None):
    # Order-one site moments set the precision of their subtracted projections.
    moments = {'atom_moments_muB': [[0., 0., 1.6], [0., 0., 1.6]]}
    return verify_branch_reproduction(saved, replayed, moments, moments, settings or Settings())


@pytest.mark.parametrize('saved_rms,replayed_rms', [
    (0.0018016016796008276, 0.0018016016796009397),
    (0.00042081290608194565, 0.00042081290608205835),
    (0.0004209461272337535, 0.0004209461272338677),
])
def test_retained_co_projection_roundoff_is_reported(saved_rms, replayed_rms):
    saved = dict(passed=True, site_rms_mub=saved_rms,
                 net_fractional_change=.001, robust_site_flip=False)
    replayed = dict(saved, site_rms_mub=replayed_rms)
    original = deepcopy(saved)
    review = check(saved, replayed)
    assert not review['exact']
    assert 0 < review['site_rms_delta_mub'] < 2e-16
    assert review['projection_roundoff_bound_mub'] < 3e-15
    assert review['magnetic_thresholds_unchanged']
    assert saved == original


@pytest.mark.parametrize('change', [
    {'site_rms_mub': .001 + 1e-10},
    {'site_rms_mub': float('nan')},
    {'site_rms_mub': -1e-16},
    {'site_rms_mub': True},
    {'net_fractional_change': .001 + 1e-16},
    {'robust_site_flip': True},
    {'passed': False},
    {'passed': 1},
    {'extra': 'unbound'},
])
def test_changed_metrics_decisions_or_shape_still_fail(change):
    replayed = dict(passed=True, site_rms_mub=.001,
                    net_fractional_change=.001, robust_site_flip=False)
    with pytest.raises(ValueError, match='magnetic branch'):
        check(dict(replayed, **change), replayed)


def test_roundoff_never_bridges_a_threshold_crossing():
    limit = Settings().branch_rms_mub
    replayed = dict(passed=True, site_rms_mub=limit-1e-16,
                    net_fractional_change=.001, robust_site_flip=False)
    saved = dict(replayed, passed=False, site_rms_mub=limit+1e-16)
    with pytest.raises(ValueError, match='does not reproduce'):
        check(saved, replayed)
    with pytest.raises(ValueError, match='contradicts'):
        check(dict(saved, passed=True), replayed)


def test_failed_branch_is_preserved_as_failed():
    failed = dict(passed=False, site_rms_mub=.3,
                  net_fractional_change=.001, robust_site_flip=False)
    assert check(failed, failed)['exact']



def test_feco_net_change_replays_only_attainable_values():
    import math
    current = dict(atom_moments_muB=[[0., 0., 2.]], total_moment_muB=[.00205, 4.55175, .00043])
    reference = dict(current, total_moment_muB=[.00017, -.00043, 4.55157])
    local = dict(passed=True, robust_site_flip=False, site_rms_mub=.001,
                 net_fractional_change=3.964751923995779e-5)
    saved = dict(local, net_fractional_change=3.964751923976265e-5)
    def verify(candidate, settings):
        return verify_branch_reproduction(candidate, local, current, reference, settings)
    assert not verify(saved, Settings())['exact']
    with pytest.raises(ValueError, match='does not reproduce'):
        verify(dict(saved, net_fractional_change=math.nextafter(saved['net_fractional_change'], 1)), Settings())
    with pytest.raises(ValueError):
        verify(saved, Settings(branch_net_fraction=local['net_fractional_change']))


def test_fused_dot_reproduces_saved_fe7co5_branch():
    import math
    cur = dict(atom_moments_muB=[[0., 0., 2.]], total_moment_muB=[-.00017, .00005, 28.16662])
    ref = dict(cur, total_moment_muB=[0., 0., 28.18967])
    local = dict(passed=True, robust_site_flip=False, site_rms_mub=.0019473526957242597,
                 net_fractional_change=.0008176753911131946)
    fused = dict(local, net_fractional_change=.0008176753911130685)
    assert not verify_branch_reproduction(fused, local, cur, ref, Settings())['exact']
    unattainable = dict(fused, net_fractional_change=math.nextafter(fused['net_fractional_change'], 1.))
    with pytest.raises(ValueError, match='does not reproduce'):
        verify_branch_reproduction(unattainable, local, cur, ref, Settings())
