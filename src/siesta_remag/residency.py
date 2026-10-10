"""Phase-resolved explicit copies are evidence of boundaries, not proof of residency."""
from collections import defaultdict

COUNTERS = ("host_to_device_bytes", "device_to_host_bytes", "copy_calls")

def summarize_residency(trace, telemetry, expected_scf_iterations):
    phases = defaultdict(lambda: dict.fromkeys(COUNTERS, 0))
    boundaries = defaultdict(lambda: dict.fromkeys(COUNTERS, 0))
    total = dict.fromkeys(COUNTERS, 0)
    steady = dict.fromkeys(COUNTERS, 0)
    intervals = trace["intervals"]
    scf = [v for v in intervals if v["phase"] == "scf"]
    device_cycle = trace.get("execution_mode") == "device_cycle"
    expected_regions = len(telemetry.get("gpu_device_cycle",{}).get("cycles",[])) if device_cycle else expected_scf_iterations
    for interval in intervals:
        for name, traffic in interval["boundaries"].items():
            for key in COUNTERS:
                value = traffic[key]
                total[key] += value
                phases[interval["phase"]][key] += value
                if interval["phase"] == "scf":
                    boundaries[name][key] += value
                    if device_cycle or interval["iteration"] > 1:
                        steady[key] += value
    errors = []
    if not trace["phase_markers_seen"] or len(scf) != expected_regions:
        errors.append(f"SCF marker coverage: {len(scf)} intervals for {expected_regions} expected regions")
    for key in COUNTERS[:2]:
        expected = sum(telemetry.get(name, {}).get(key, 0) for name in
                       ("gpu_bridge", "gpu_grid", "gpu_density_matrix", "gpu_occupations", "gpu_matrix_assembly", "gpu_hartree", "gpu_grid_flow", "gpu_sparse_state", "gpu_device_cycle", "gpu_grid_force_shadow", "gpu_grid_force", "gpu_atomic_grid_force_shadow", "gpu_final_density", "gpu_orbital_basis", "gpu_atomic_basis", "gpu_two_center", "gpu_pair_force", "gpu_nonlocal_force", "gpu_integral_force", "gpu_population", "gpu_initial_state", "gpu_density_history", "gpu_final_spectrum", "gpu_nonlocal_h0", "gpu_initial_mesh"))
        if total[key] != expected:
            errors.append(f"Transfer counters disagree: {key} phase total {total[key]} vs bridge total {expected}")
    return dict(scope=trace["scope"], execution_mode="device_cycle" if device_cycle else "host_iterations",
                device_iterations=expected_scf_iterations if device_cycle else None, phases=dict(phases), scf_boundaries=dict(boundaries),
                steady_state_explicit=steady, total_explicit=total, scf_intervals=len(scf),
                geometry_epochs=trace["geometry_epochs"], accounting_errors=errors,
                # Bridge counters exclude library internals, migrations and syscalls.
                # Even all-zero explicit counters cannot prove the strict target.
                strict_residency_verified=False,
                unmeasured=["library transfers", "unified-memory migrations", "file I/O"],
                steady_state_definition="entire electronic cycle including the first accepted iteration" if device_cycle else "SCF iterations after the first in each geometry")
