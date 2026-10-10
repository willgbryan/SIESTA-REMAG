import copy

import pytest

from siesta_remag.cuda_trace import (API_COLUMNS, COLUMNS, EXTERNAL_COLUMNS, GRAPH_EVENT_COLUMNS,
                                    GRAPH_LAUNCH_COLUMNS, GRAPH_REGISTRATION_COLUMNS, summarize_cuda)


def fixture():
    def event(kind, begin, end, size=0, direction=0):
        return [kind, begin, end, size, 0, 1, 7, 123, 456, 0, direction, 0, 0]
    trace = dict(schema_version=1, event_columns=COLUMNS, memcpy_enabled=True, kernel_enabled=True,
                 dropped_records=0, record_overflow=0, callback_failures=0, errors=[], end_ns=500,
                 phases=[dict(begin_ns=t, phase=p, geometry=1, iteration=i)
                         for t, p, i in ((100, 0, 0), (200, 1, 1), (300, 1, 2), (400, 2, 0))],
                 kernel_names=["library_kernel"], events=[event(1, 150, 160, 80, 1),
                    event(1, 250, 260, 40, 2), event(1, 299, 301, 4, 2),
                    event(2, 320, 350), event(1, 390, 410, 20, 8)])
    residency = dict(total_explicit=dict(host_to_device_bytes=80, device_to_host_bytes=40), accounting_errors=[])
    return trace, residency


def test_library_copies_and_conservative_phase_attribution():
    trace, residency = fixture()
    r = summarize_cuda(trace, residency, expected_scf_iterations=2, process_succeeded=True)
    assert r["coverage_complete"] and not r["strict_residency_verified"]
    assert not r["phase_attribution_verified"]
    assert r["bytes_beyond_explicit"] == dict(host_to_device_bytes=0, device_to_host_bytes=4)
    assert r["total"]["copy_calls"] == 4
    assert r["boundary_overlaps"] == 2
    assert r["steady_state"]["device_to_host_bytes"] == 4
    assert r["phases"]["scf"]["device_to_device_bytes"] == 20
    assert r["phases"]["post_scf"]["device_to_device_bytes"] == 20
    assert r["scf_copy_sizes"]["device_to_host"] == {"4": 1, "40": 1}
    assert r["scf_kernels"]["library_kernel"]["kernel_nanoseconds"] == 30


@pytest.mark.parametrize("defect", ["drop", "missing_counter", "overflow", "callback", "timestamp", "order",
                                  "direction", "missing_events", "too_few_bytes", "markers", "failed_process"])
def test_incomplete_activity_never_qualifies_coverage(defect):
    trace, residency = copy.deepcopy(fixture())
    if defect == "drop": trace["dropped_records"] = 1
    elif defect == "missing_counter": del trace["dropped_records"]
    elif defect == "overflow": trace["record_overflow"] = 1
    elif defect == "callback": trace["callback_failures"] = 1
    elif defect == "timestamp": trace["events"][0][1:3] = [0, 0]
    elif defect == "order": trace["phases"][1]["begin_ns"] = 50
    elif defect == "direction": trace["events"][0][10] = 0
    elif defect == "missing_events": trace["events"] = []
    elif defect == "too_few_bytes": residency["total_explicit"]["device_to_host_bytes"] = 45
    elif defect == "markers": trace["phases"] = []
    r = summarize_cuda(trace, residency, expected_scf_iterations=2, process_succeeded=defect != "failed_process")
    assert not r["coverage_complete"] and r["errors"]
    assert r.get("bytes_beyond_explicit") is None
    assert not r["strict_residency_verified"]


def correlated_fixture():
    trace, residency = fixture()
    trace.update(schema_version=2, runtime_enabled=True, driver_enabled=True, external_enabled=True,
                 external_kind=3, api_columns=API_COLUMNS, external_columns=EXTERNAL_COLUMNS,
                 apis=[], external=[])
    for i, p in enumerate(trace["phases"], 1):
        p["external_id"] = i
    for i, (event, phase_id) in enumerate(zip(trace["events"], [1, 2, 3, 3, 4]), 1):
        event[7:9] = [i, i + 10]
        # Include both API paths; record delivery order need not be temporal.
        for correlation in (i, i + 10):
            trace["apis"].insert(0, [5, correlation, 110, 120, 1, 1, 1])
            trace["external"].insert(0, [correlation, 3, phase_id])
        event[1:3] = [150, 160]  # Deliberately wrong host/GPU alignment.
    return trace, residency


def test_launch_correlation_survives_clock_skew_and_async_completion():
    trace, residency = correlated_fixture()
    trace["events"][3][1:3] = [600, 630]  # Work can complete outside its host region.
    r = summarize_cuda(trace, residency, expected_scf_iterations=2, process_succeeded=True)
    assert r["coverage_complete"] and r["phase_attribution_verified"]
    assert r["phase_attribution"] == "launch_external_correlation"
    assert r["timestamp_launch_disagreements"] == 4
    assert r["events_outside_host_timestamps"] == 1
    assert r["phases"]["scf"]["device_to_host_bytes"] == 44
    assert r["steady_state"]["device_to_host_bytes"] == 4
    assert r["steady_state"]["kernel_count"] == 1
    assert r["phases"]["post_scf"]["device_to_device_bytes"] == 20
    assert "device_to_device_bytes" not in r["phases"]["scf"]
    assert not r["strict_residency_verified"]


@pytest.mark.parametrize("defect", ["no_external", "no_api", "no_schema", "unmapped", "ambiguous",
                                  "conflicting_alias", "unknown_phase", "duplicate_phase", "disabled"])
def test_correlation_defects_do_not_fall_back_to_timestamps(defect):
    trace, residency = correlated_fixture()
    if defect == "no_external": trace["external"] = []
    elif defect == "no_api": trace["apis"] = []
    elif defect == "no_schema": del trace["external_columns"]
    elif defect == "unmapped": trace["events"][0][7:9] = [99, 100]
    elif defect == "ambiguous": trace["external"].append([1, 3, 2])
    elif defect == "conflicting_alias": trace["external"][0][2] = 1
    elif defect == "unknown_phase": trace["external"][0][2] = 100
    elif defect == "duplicate_phase": trace["phases"][1]["external_id"] = 1
    elif defect == "disabled": trace["runtime_enabled"] = False
    r = summarize_cuda(trace, residency, expected_scf_iterations=2, process_succeeded=True)
    assert not r["coverage_complete"] and r["errors"]
    assert not r["phase_attribution_verified"]
    assert r["bytes_beyond_explicit"] is None


@pytest.mark.parametrize("defect", [None, "unmapped", "host", "unknown", "undeclared"])
def test_memset_is_counted_separately_and_requires_device_and_phase_evidence(defect):
    trace, residency = correlated_fixture()
    trace["memset_enabled"] = True
    event = trace["events"][3].copy()
    event[0], event[3], event[12] = 3, 4096, 3
    if defect == "unmapped": event[7:9] = [0, 0]
    elif defect == "host": event[12] = 2
    elif defect == "unknown": event[12] = 0
    elif defect == "undeclared": del trace["memset_enabled"]
    trace["events"].append(event)
    r = summarize_cuda(trace, residency, expected_scf_iterations=2, process_succeeded=True)
    assert r["coverage_complete"] is (defect is None)
    assert r["total"]["memset_calls"] == 1 and r["total"]["memset_bytes"] == 4096
    assert r["total"]["copy_calls"] == 4 and r["total"]["kernel_count"] == 1
    if defect is None:
        assert r["steady_state"]["memset_bytes"] == 4096
        assert r["scf_kernels"]["library_kernel"]["kernel_count"] == 1
    else:
        assert r["errors"] and not r["phase_attribution_verified"]


def graph_scope_fixture():
    trace, residency = correlated_fixture()
    trace.update(cuda_build_version=12060, graph_launch_callbacks_enabled=True,
                 graph_launch_columns=GRAPH_LAUNCH_COLUMNS, graph_registration_columns=GRAPH_REGISTRATION_COLUMNS,
                 event_graph_columns=GRAPH_EVENT_COLUMNS, graph_launches=[[7, 99, 5, 311, 0]],
                 graph_registrations=[[7, 0]], event_graphs=[[0, 0] for _ in trace["events"]])
    trace["apis"].append([5, 99, 310, 320, 1, 1, 311])
    trace["external"].append([99, 3, 3])
    trace["event_graphs"][3] = [7, 71]
    trace["events"][3][7:9] = [0, 0]
    return trace, residency


@pytest.mark.parametrize("defect", [None, "undeclared", "kernels_enabled", "memset_disabled", "incidental_kernel"])
def test_copy_only_observation_is_explicit_and_separate_from_kernel_coverage(defect):
    trace, residency = correlated_fixture()
    trace.update(kernel_activity_requested=False,kernel_enabled=False,memset_enabled=True)
    trace['events'] = [row for row in trace['events'] if row[0]!=2]
    if defect == "undeclared": trace.pop('kernel_activity_requested')
    elif defect == "kernels_enabled": trace['kernel_enabled'] = True
    elif defect == "memset_disabled": trace['memset_enabled'] = False
    elif defect == "incidental_kernel": trace['events'].append(correlated_fixture()[0]['events'][3])
    ordinary = summarize_cuda(trace,residency,expected_scf_iterations=2,process_succeeded=True)
    assert not ordinary['coverage_complete']
    result = summarize_cuda(trace,residency,expected_scf_iterations=2,process_succeeded=True,memory_only=True)
    assert result['coverage_complete'] == (defect in (None,"incidental_kernel")),result['errors']
    assert not result['kernel_activity_requested'] and not result['kernel_coverage_complete']
    assert result['kernel_activity_observed'] == (defect=="incidental_kernel")
    assert not result['strict_residency_verified']


def test_unique_registered_host_graph_scope_is_opt_in_and_not_timestamp_based():
    trace, residency = graph_scope_fixture()
    ordinary = summarize_cuda(trace, residency, expected_scf_iterations=2, process_succeeded=True)
    assert not ordinary["coverage_complete"]
    traced = summarize_cuda(trace, residency, expected_scf_iterations=2, process_succeeded=True,
                            allow_unique_graph_scope=True)
    assert traced["coverage_complete"] and traced["phase_attribution_verified"]
    assert traced["graph_scope_attributed_events"] == 1
    assert traced["steady_state"]["kernel_count"] == 1
    assert traced["timestamp_launch_disagreements"] == 4
    assert not traced["strict_residency_verified"]


@pytest.mark.parametrize("defect", ["multiple_phases", "missing_callback", "missing_api", "missing_external",
    "ambiguous_launch", "device_flag", "unregistered", "duplicate_registration", "foreign_event_graph",
    "zero_node", "version", "callbacks_disabled", "node_records", "partial_id", "duplicate_callback",
    "duplicate_api", "contradictory_event"])
def test_graph_scope_fails_closed_on_missing_ambiguous_or_ineligible_evidence(defect):
    trace, residency = graph_scope_fixture()
    if defect == "multiple_phases":
        trace["apis"].append([5, 100, 210, 220, 1, 1, 311])
        trace["external"].append([100, 3, 2])
        trace["graph_launches"].append([7, 100, 5, 311, 0])
    elif defect == "missing_callback": trace["graph_launches"] = []
    elif defect == "missing_api": trace["apis"].pop()
    elif defect == "missing_external": trace["external"].pop()
    elif defect == "ambiguous_launch": trace["external"].append([99, 3, 2])
    elif defect == "device_flag": trace["graph_registrations"][0][1] = 4
    elif defect == "unregistered": trace["graph_registrations"] = []
    elif defect == "duplicate_registration": trace["graph_registrations"].append([7, 0])
    elif defect == "foreign_event_graph": trace["event_graphs"][3][0] = 8
    elif defect == "zero_node": trace["event_graphs"][3][1] = 0
    elif defect == "version": trace["cuda_build_version"] = 13000
    elif defect == "callbacks_disabled": trace["graph_launch_callbacks_enabled"] = False
    elif defect == "node_records": trace["event_graphs"] = []
    elif defect == "partial_id": trace["events"][3][7] = 999
    elif defect == "duplicate_callback": trace["graph_launches"].append([7, 99, 5, 311, 0])
    elif defect == "duplicate_api": trace["apis"].append(trace["apis"][-1].copy())
    elif defect == "contradictory_event": trace["events"][3][7:9] = [1, 11]
    r = summarize_cuda(trace, residency, expected_scf_iterations=2, process_succeeded=True,
                       allow_unique_graph_scope=True)
    assert not r["coverage_complete"] and r["errors"]
    assert not r["phase_attribution_verified"] and r["bytes_beyond_explicit"] is None


@pytest.mark.parametrize("defect", [None, "missing_parent", "wrong_graph", "wrong_correlation", "no_nesting",
                                  "duplicate_driver", "runtime_nesting", "missing_parent_api"])
def test_registered_graph_requires_explicit_nested_driver_provenance(defect):
    trace, residency = graph_scope_fixture()
    trace["graph_launches"].append([7, 99, 4, 514, 99])
    if defect == "missing_parent": trace["graph_launches"].pop(0)
    elif defect == "wrong_graph": trace["graph_launches"][-1][0] = 8
    elif defect == "wrong_correlation": trace["graph_launches"][-1][1] = 100
    elif defect == "no_nesting": trace["graph_launches"][-1][-1] = 0
    elif defect == "duplicate_driver": trace["graph_launches"].append(trace["graph_launches"][-1].copy())
    elif defect == "runtime_nesting": trace["graph_launches"][0][-1] = 99
    elif defect == "missing_parent_api": trace["apis"].pop()
    r = summarize_cuda(trace, residency, expected_scf_iterations=2, process_succeeded=True,
                       allow_unique_graph_scope=True)
    assert r["coverage_complete"] == (defect is None), r["errors"]

def test_whole_device_cycle_keeps_first_launch_copy_in_steady_scope():
    trace,residency=correlated_fixture()
    residency['execution_mode']='device_cycle'
    r=summarize_cuda(trace,residency,expected_scf_iterations=2,process_succeeded=True)
    assert r['coverage_complete']
    assert r['steady_state']['device_to_host_bytes']==44
