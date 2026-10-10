"""CUPTI memory-copy/memset/kernel evidence, including CUDA-library activity.

The observer does not measure unified-memory migration or kernels accessing
mapped host memory. Complete coverage here never proves strict residency.
"""
from bisect import bisect_left
from collections import Counter, defaultdict

PHASES = ("initialization", "scf", "post_scf", "analysis_export")
COLUMNS = ["kind", "begin_ns", "end_ns", "bytes", "device", "context", "stream",
           "correlation", "runtime_correlation", "name_id", "copy_kind", "source_kind", "destination_kind"]
API_COLUMNS = ["kind", "correlation", "begin_ns", "end_ns", "process", "thread", "callback"]
EXTERNAL_COLUMNS = ["correlation", "external_kind", "external_id"]
GRAPH_LAUNCH_COLUMNS = ["graph_id", "correlation", "api_kind", "callback", "enclosing_runtime_correlation"]
GRAPH_REGISTRATION_COLUMNS = ["graph_id", "instantiation_flags"]
GRAPH_EVENT_COLUMNS = ["graph_id", "graph_node_id"]
# CUDA 12.6 CUpti_ActivityMemcpyKind. Arrays are device memory too.
DIRECTIONS = {1: "host_to_device", 2: "device_to_host", 3: "host_to_device",
              4: "device_to_host", 5: "device_to_device", 6: "device_to_device",
              7: "device_to_device", 8: "device_to_device", 9: "host_to_host", 10: "peer_to_peer"}


def registered_graph_phases(trace, api_records, correlations, external_phases, errors):
    """Prove a host graph's complete launch-phase set without using timestamps.

    Only explicitly queried zero-flag executable graphs are eligible. Device
    launches do not have host API callbacks and must never use this path.
    CUDA 12.6 is the only runtime version covered by these callback IDs.
    """
    initial_errors = len(errors)
    if trace.get("cuda_build_version") != 12060 or trace.get("graph_launch_callbacks_enabled") is not True:
        errors.append("Registered graph scope requires the supported runtime and complete launch callbacks")
    if (trace.get("graph_launch_columns") != GRAPH_LAUNCH_COLUMNS or
            trace.get("graph_registration_columns") != GRAPH_REGISTRATION_COLUMNS or
            trace.get("event_graph_columns") != GRAPH_EVENT_COLUMNS):
        errors.append("Missing or unsupported registered graph schema")
        return {}
    event_graphs = trace.get("event_graphs", [])
    if len(event_graphs) != len(trace.get("events", [])) or any(
            len(row) != 2 or any(type(v) is not int or v < 0 for v in row) for row in event_graphs):
        errors.append("Invalid event graph provenance")
    registered = {}
    for row in trace.get("graph_registrations", []):
        if (len(row) != 2 or any(type(v) is not int or v < 0 for v in row) or
                not row[0] or row[0] in registered):
            errors.append("Invalid or duplicate graph registration")
            continue
        registered[row[0]] = row[1]
    callbacks = {4: {514, 515}, 5: {311, 312}}  # driver and runtime GraphLaunch[/ptsz]
    expected = {row for row in api_records if row[2] in callbacks.get(row[0], set())}
    recorded_launch_count = sum(len(row) == len(API_COLUMNS) and
        (row[0], row[1], row[6]) in expected for row in trace.get("apis", []))
    if recorded_launch_count != len(expected):
        errors.append("Duplicate graph-launch API activity")
    launches = {}
    for row in trace.get("graph_launches", []):
        if len(row) != len(GRAPH_LAUNCH_COLUMNS) or any(type(v) is not int or v < 0 for v in row):
            errors.append("Invalid graph launch callback record")
            continue
        graph, correlation, kind, callback, enclosing = row
        key = (kind, correlation, callback)
        if not graph or not correlation or callback not in callbacks.get(kind, set()) or key in launches:
            errors.append("Graph launch callback lacks a unique matching API record")
            continue
        launches[key] = (graph, enclosing)
    seen, scopes = set(), defaultdict(set)
    for key, (graph, enclosing) in launches.items():
        kind, correlation, callback = key
        if enclosing:
            # CUDA 12.6 can report the nested driver callback using the outer
            # runtime correlation without a separate driver API activity. Only
            # explicitly observed callback nesting plus the matching runtime
            # callback AND API record can establish this equivalence.
            parents = [p for p, value in launches.items() if p[0] == 5 and p[1] == enclosing
                       and value == (graph, 0) and p in expected]
            if kind != 4 or enclosing != correlation or len(parents) != 1:
                errors.append("Nested driver graph callback has no unique evidenced runtime parent")
                continue
        if key in expected:
            seen.add(key)
        elif not enclosing:
            errors.append("Graph launch callback lacks a matching API record")
            continue
        phases = correlations.get(correlation, set())
        if len(phases) != 1 or not phases <= external_phases.keys():
            errors.append("Graph launch lacks a unique external phase")
            continue
        if graph not in registered:
            errors.append("Graph launch has no queried executable registration")
            continue
        # Reject all other flags, including device-launch-capable executables.
        # Do not infer eligibility from graph names, source capture or timing.
        if registered[graph] != 0:
            errors.append("Registered graph is not an eligible zero-flag host executable")
            continue
        scopes[graph].update(phases)
    if seen != expected:
        errors.append("Graph launch callbacks do not cover the complete API ledger")
    if len(errors) == initial_errors and any((graph == 0) != (node == 0) or
            (graph and graph not in scopes) for graph, node in event_graphs):
        errors.append("Graph activity has no complete registered launch history")
    return dict(scopes) if len(errors) == initial_errors else {}


def summarize_cuda(trace, residency, *, expected_scf_iterations, process_succeeded,
                   allow_unique_graph_scope=False, memory_only=False):
    errors = list(trace.get("errors", []))
    correlated = trace.get("schema_version") == 2
    result = dict(scope=("CUPTI memcpy/memset activity; kernel collection not requested, incidental graph kernels retained" if memory_only else
                        "CUPTI memcpy/memset and concurrent-kernel activity from process initialization through explicit finalization"),
                  kernel_activity_requested=not memory_only,
                  coverage_complete=False, errors=errors, strict_residency_verified=False,
                  unmeasured=["unified-memory migrations", "kernel access to mapped host memory",
                              "CPU library work and synchronization", "activity after trace finalization"],
                  phase_attribution=("launch_external_correlation_and_registered_graph_scope" if allow_unique_graph_scope else
                                     "launch_external_correlation") if correlated else "legacy_timestamp_intersection",
                  phase_attribution_verified=False,
                  notes=("Phases identify the launching API's region, not the physical interval of GPU execution. "
                         "Timestamp intersections are diagnostic only; asynchronous work may finish in a later phase. "
                         if correlated else
                         "Legacy timestamp-only attribution is unqualified: GPU/host clock alignment can misassign phases. "
                         "Events touching a timestamp boundary count in both intervals. ") +
                        "Process totals count each event once. Kernel durations sum activity and are not wall time. "
                        "Traced timings never qualify speedups.")
    if trace.get("schema_version") not in (1, 2) or trace.get("event_columns") != COLUMNS:
        errors.append("Missing or unsupported CUPTI trace schema")
        return result
    if memory_only:
        result["unmeasured"].append("complete kernel execution activity and timing")
        if (trace.get("kernel_activity_requested") is not False or trace.get("kernel_enabled") is not False or
                trace.get("memset_enabled") is not True):
            errors.append("Copy-only observation was not explicitly established")
    if "kernel_activity_requested" in trace and trace["kernel_activity_requested"] != trace.get("kernel_enabled"):
        errors.append("Requested and enabled kernel activity disagree")
    for flag in ("memcpy_enabled",) + (() if memory_only else ("kernel_enabled",)) + (("runtime_enabled", "driver_enabled", "external_enabled") if correlated else ()):
        if trace.get(flag) is not True:
            errors.append(f"Activity enablement missing: {flag}")
    if "memset_enabled" in trace and trace["memset_enabled"] is not True:
        errors.append("Activity enablement missing: memset_enabled")
    for counter in ("dropped_records", "record_overflow", "callback_failures"):
        if trace.get(counter) != 0:
            errors.append(f"Incomplete activity: {counter}={trace.get(counter)}")
    if not process_succeeded:
        errors.append("Successful traced-process completion was not established")
    markers, end = trace.get("phases", []), trace.get("end_ns", 0)
    try:
        starts = [p["begin_ns"] for p in markers]
        ends = starts[1:] + [end]
        valid_markers = bool(markers) and all(isinstance(t, int) and t > 0 for t in starts + [end])
        valid_markers &= all(a <= b for a, b in zip(starts, ends))
        valid_markers &= all(p["phase"] in range(4) and p["geometry"] >= 0 and p["iteration"] >= 0 for p in markers)
        valid_markers &= all(p["iteration"] > 0 for p in markers if p["phase"] == 1)
    except (KeyError, TypeError):
        valid_markers = False
    if not valid_markers:
        errors.append("CUPTI phase timestamps missing or invalid")
        return result
    external_phases, api_ids, api_records, correlations = {}, set(), set(), defaultdict(set)
    if correlated:
        for p in markers:
            external_id = p.get("external_id")
            if not isinstance(external_id, int) or external_id <= 0 or external_id in external_phases:
                errors.append("Missing, invalid or duplicate external phase ID")
            else:
                external_phases[external_id] = p
        if trace.get("api_columns") != API_COLUMNS or trace.get("external_columns") != EXTERNAL_COLUMNS:
            errors.append("Missing or unsupported correlation record schema")
        else:
            for row in trace.get("apis", []):
                if (len(row) != len(API_COLUMNS) or any(not isinstance(v, int) or v < 0 for v in row)
                        or not row[1]):
                    errors.append("Invalid API correlation record")
                    continue
                api_ids.add(row[1])
                api_records.add((row[0], row[1], row[6]))
            for row in trace.get("external", []):
                if len(row) != len(EXTERNAL_COLUMNS) or any(not isinstance(v, int) or v < 0 for v in row):
                    errors.append("Invalid external correlation record")
                    continue
                correlation, kind, external_id = row
                if kind != trace.get("external_kind"):
                    errors.append("Unexpected external correlation kind")
                    continue
                if not correlation or external_id not in external_phases:
                    errors.append("External correlation references an unknown phase or invalid API ID")
                correlations[correlation].add(external_id)
            if not api_ids or not correlations:
                errors.append("Missing API or external correlation records")
            if any(len(ids) != 1 for ids in correlations.values()):
                errors.append("Conflicting external phases for an API correlation ID")
    graph_phases = {}
    if allow_unique_graph_scope:
        if not correlated:
            errors.append("Registered graph attribution requires external-correlation schema")
        else:
            graph_phases = registered_graph_phases(trace, api_records, correlations, external_phases, errors)
    scf_count = sum(p["phase"] == 1 for p in markers)
    if not scf_count or scf_count != expected_scf_iterations:
        errors.append(f"SCF marker coverage: {scf_count} intervals for {expected_scf_iterations} iterations")
    phases, kernels, scf_kernels = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    sizes = defaultdict(Counter)
    total, steady, defects = Counter(), Counter(), Counter()
    overlaps, timestamp_disagreements, timestamp_outside = 0, 0, 0
    graph_scope_events = 0
    names = trace.get("kernel_names", [])
    for event_index, row in enumerate(trace.get("events", [])):
        if len(row) != len(COLUMNS) or any(not isinstance(v, int) or v < 0 for v in row):
            defects["malformed event"] += 1
            continue
        kind, begin, finish, size, _, _, _, _, _, name_id, copy_kind, _, _ = row
        count = Counter()
        if kind == 1:
            direction = DIRECTIONS.get(copy_kind, "unknown_direction")
            count[direction + "_bytes"] = size
            count[direction + "_copies"] = 1
            count["copy_calls"] = 1
            if direction == "unknown_direction":
                defects["unknown memcpy direction"] += 1
        elif kind == 2 and name_id < len(names):
            # CUDA graphs can emit kernel records even when this activity kind
            # was not enabled. Retain and attribute every observed record, but
            # do not claim complete kernel coverage in copy-only mode.
            count["kernel_count"] = 1
            if begin and finish >= begin:
                count["kernel_nanoseconds"] = finish - begin
            kernels[names[name_id]].update(count)
        elif kind == 3:
            count["memset_calls"] = 1
            count["memset_bytes"] = size
            if trace.get("memset_enabled") is not True:
                defects["memset without declared activity enablement"] += 1
            # CUpti_ActivityMemoryKind device, array, device-static. Managed,
            # host or unknown destinations cannot establish device-only work.
            if row[12] not in (3, 4, 6):
                defects["memset destination is not known device memory"] += 1
        else:
            defects["unknown event or kernel name"] += 1
            continue
        total.update(count)
        if not begin or not finish or finish < begin:
            defects["missing or reversed activity timestamps"] += 1
            phases["unattributed"].update(count)
            continue
        intersect = []
        for index in range(bisect_left(ends, begin), len(markers)):
            if starts[index] > finish:
                break
            intersect.append(markers[index])
        if begin < starts[0] or finish > end:
            timestamp_outside += 1
            if not correlated:
                defects["activity outside timestamp coverage"] += 1
        if len(intersect) > 1:
            overlaps += 1
        if correlated:
            # A memcpy can carry both driver and runtime IDs. One fully
            # evidenced path suffices, but all supplied external tags must
            # agree; no event falls back to timestamp assignment.
            ids = {row[7], row[8]} - {0}
            referenced = set().union(*(correlations.get(i, set()) for i in ids))
            evidenced = set().union(*(correlations.get(i, set()) for i in ids & api_ids))
            if not ids and graph_phases:
                graph, node = trace["event_graphs"][event_index]
                possible = graph_phases.get(graph, set()) if node else set()
                if len(possible) == 1:
                    referenced = evidenced = possible
                    graph_scope_events += 1
            if len(referenced) != 1 or evidenced != referenced or not referenced <= external_phases.keys():
                defects["unmapped or ambiguous launch phase"] += 1
                phases["unattributed"].update(count)
                continue
            if graph_phases:
                graph, _ = trace["event_graphs"][event_index]
                if graph and not referenced <= graph_phases[graph]:
                    defects["Event correlation contradicts registered graph launch phases"] += 1
                    phases["unattributed"].update(count)
                    continue
            marker = external_phases[next(iter(referenced))]
            if marker not in intersect:
                timestamp_disagreements += 1
            intersect = [marker]
        labels = {PHASES[p["phase"]] for p in intersect} or {"outside_markers"}
        for phase in labels:
            phases[phase].update(count)
        if any(p["phase"] == 1 and (residency.get("execution_mode")=="device_cycle" or p["iteration"] > 1) for p in intersect):
            steady.update(count)
        if "scf" in labels:
            if kind == 1:
                sizes[direction][str(size)] += 1
            elif kind == 2:
                scf_kernels[names[name_id]].update(count)
    if total["copy_calls"] == 0 or (not memory_only and total["kernel_count"] == 0):
        errors.append("No memcpy activity captured" if memory_only else "No memcpy or kernel activity captured")
    errors.extend(f"{key}: {value}" for key, value in sorted(defects.items()))
    explicit = residency.get("total_explicit")
    difference = {}
    if not explicit or residency.get("accounting_errors") != []:
        errors.append("Explicit bridge accounting unavailable or invalid")
    else:
        for key in ("host_to_device_bytes", "device_to_host_bytes"):
            value = total[key] - explicit[key]
            if value < 0:
                errors.append(f"CUPTI {key} {total[key]} is less than explicit bridge {explicit[key]}")
            difference[key] = value
    result.update(coverage_complete=not errors, phase_attribution_verified=correlated and not errors,
                  kernel_activity_observed=total["kernel_count"]>0,
                  kernel_coverage_complete=not memory_only and not errors,
                  scf_intervals=scf_count,
                  total=dict(total), phases=dict(phases), steady_state=dict(steady),
                  boundary_overlaps=overlaps, scf_copy_sizes=dict(sizes),
                  timestamp_launch_disagreements=timestamp_disagreements if correlated else None,
                  events_outside_host_timestamps=timestamp_outside,
                  graph_scope_attributed_events=graph_scope_events,
                  kernels=dict(kernels), scf_kernels=dict(scf_kernels),
                  bytes_beyond_explicit=difference if not errors else None)
    return result
