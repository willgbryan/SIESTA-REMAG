"""Cross-platform angular roundoff must not change native acceptance."""
from copy import deepcopy
import math

import numpy as np
import pytest

from siesta_remag.discovery.config import Settings
from siesta_remag.research.relaxation_execution import verify_native_reproduction


def review(angle):
    failed = angle is not None and angle > Settings().axis_tolerance_deg
    return dict(accepted=not failed, errors=['Final moment left the requested axis'] if failed else [],
                axis_deviation_deg=angle, max_force_ev_ang=.025, native_kpoints=216,
                artifact_sha256={'result.json':'retained'}, timed_out=False)


def test_actual_fcc_co_roundoff_in_acos_is_retained_and_reported():
    saved=review(.0010583670282510356); replayed=review(.0010583666838858306)
    before=deepcopy(saved)
    result=verify_native_reproduction(saved,replayed,Settings())
    assert not result['exact'] and result['native_thresholds_unchanged']
    assert 0 < result['axis_cosine_delta'] <= np.finfo(float).eps
    assert result['axis_cosine_roundoff_bound'] < 2e-15
    assert saved == before


def test_exact_alignment_and_scalar_none():
    roundoff=math.degrees(math.acos(1-np.finfo(float).eps))
    assert not verify_native_reproduction(review(0),review(roundoff),Settings())['exact']
    assert verify_native_reproduction(review(None),review(None),Settings())['exact']
    with pytest.raises(ValueError,match='presence'):
        verify_native_reproduction(review(None),review(0),Settings())


@pytest.mark.parametrize('angle',[float('nan'),float('inf'),-.0001,90.1,True])
def test_invalid_angles_fail(angle):
    with pytest.raises(ValueError):verify_native_reproduction(review(angle),review(0),Settings())


@pytest.mark.parametrize('change',[
    {'accepted':False}, {'errors':['new error']}, {'native_kpoints':215},
    {'max_force_ev_ang':.02500000000000001}, {'artifact_sha256':{'result.json':'changed'}},
    {'timed_out':True}, {'axis_deviation_deg':.0011},
])
def test_other_metrics_flags_and_artifacts_remain_exact(change):
    replayed=review(.0010583666838858306)
    with pytest.raises(ValueError):verify_native_reproduction(dict(replayed,**change),replayed,Settings())


def test_threshold_crossing_cannot_be_accepted_as_roundoff():
    limit=Settings().axis_tolerance_deg
    with pytest.raises(ValueError):
        verify_native_reproduction(review(limit-1e-14),review(limit+1e-14),Settings())
    forged=review(limit-1e-14);forged['axis_deviation_deg']=limit+1e-14
    with pytest.raises(ValueError,match='contradicts'):
        verify_native_reproduction(forged,review(limit-1e-14),Settings())


def test_failed_native_axis_remains_failed():
    row=review(2.)
    assert not row['accepted']
    assert verify_native_reproduction(row,row,Settings())['exact']
