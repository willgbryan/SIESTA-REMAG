#!/usr/bin/env python3
"""Produce a concise review of measured, upstream-verified runs."""
import argparse
import json
import statistics
from pathlib import Path
from siesta_remag.profile import summarize

def number(value, precision=3):
    return f"{value:.{precision}f}" if value is not None else "unmeasured"

def report(root):
    ledger = json.loads((root / "ledger.json").read_text())
    qualification=[]
    interruption=root/"interruption.json"
    if interruption.exists():
        stopped=json.loads(interruption.read_text())
        qualification += ["**Interrupted batch — partial results only.** " + stopped.get("reason", "Completion was not established.")]
        if stopped.get("unfinished_work"):
            qualification += ["Unfinished requested work: " + "; ".join(stopped["unfinished_work"]) + "."]
    metadata=root/"metadata.json"
    hardware="Single Modal GPU (physical model unrecorded)"
    reference_reused=False
    if metadata.exists():
        meta=json.loads(metadata.read_text());requested=meta.get("cases",[])
        reference_reused=bool(meta.get('reference_run'))
        inventory={}
        for row in meta.get("gpu","").splitlines():
            if ':' in row:
                key,value=row.split(':',1);inventory[key.strip()]=value.strip().split(' [Deprecated;')[0]
        model=inventory.get('Product Name');driver=inventory.get('Driver Version')
        if model:hardware=f"Single Modal GPU: {model}"+(f"; NVIDIA driver {driver}" if driver else "")
        retained={r["benchmark"] for r in ledger if
                  r.get("candidate_path",r["benchmark"]+"/candidate").endswith("/candidate")}
        missing=[case for case in requested if case not in retained]
        if missing:
            qualification += ["Requested cases without a retained primary candidate comparison: " + ", ".join(missing) +
                              ". This report does not qualify the complete requested batch."]
    repeated = any("candidate-repeat" in r.get("candidate_path", "") for r in ledger)
    timing_scope = ("Candidate repetitions are summarized below; each CPU reference and ablation control is measured once. "
                    if repeated else "Timings are preliminary single observations, not repeated-run statistics. ")
    lines = [f"# Measured results: {root.name}", "", *[line+"\n" for line in qualification],
             hardware+(". The serial upstream CPU reference is reused from allocation " + meta['reference_run'] +
                       "; cross-allocation CPU/GPU speedup is unqualified. One BLAS thread. " if reference_reused else
                       ". Serial upstream CPU and GPU candidate run on the same host, one BLAS thread. ")+
             "Each entry is one cold process run and includes setup, SCF, final forces and output. "
             "Cloud provisioning, image builds and artifact transfer are excluded. "
             + timing_scope +
             "Component categories are regenerated from raw time.json with the current reporter; "
             "original result.json and scientific comparisons remain unchanged.", "",
             "| Benchmark | Passed | CPU seconds | GPU seconds | Speedup | SCF CPU / GPU |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for record in ledger:
        speed = record["end_to_end_speedup"]
        formatted = f"{speed:.2f}x" if speed is not None else "not qualified"
        lines.append(f"| {record['benchmark']} | {record['passed']} | {record['reference_seconds']:.3f} | "
                     f"{record['candidate_seconds']:.3f} | {formatted} | "
                     f"{record['reference_iterations']} / {record['candidate_iterations']} |")
    standalone=root/"standalone-component-status.json"
    if standalone.exists():
        for name,item in json.loads(standalone.read_text()).items():
            lines += ["",f"Standalone {name}: passed={item['passed']}; integrated into SIESTA="
                      f"{item['integrated_into_siesta']}. Its qualification is separate from the SCF comparisons above."]
    for path in sorted(root.glob("mae-*.json")):
        mae = json.loads(path.read_text())
        lines += ["", f"{path.stem}: passed={mae['passed']}; E(x)−E(z) reference "
                  f"{mae.get('reference_mae_ev')} eV; candidate {mae.get('candidate_mae_ev')} eV; "
                  f"absolute difference {mae.get('max_absolute_error')} eV."]
    for path in sorted(root.glob("ablation-*.json")):
        delta=json.loads(path.read_text())
        feature=("grid geometry lifetime tracking" if path.name.startswith("ablation-epoch-") else
                 "retained GPU eigenvectors" if path.name.startswith("ablation-vector-cache-") else
                 "deferred SCF diagnostics" if path.name.startswith("ablation-log-") else
                 "device orbital derivatives" if path.name.startswith("ablation-orbital-basis-") else
                 "final density output reuse" if path.name.startswith("ablation-final-density-") else
                 "final-force grid reuse" if path.name.startswith("ablation-final-grid-") else
                 "shared GPU core/neutral forces" if path.name.startswith("ablation-atomic-force-") else
                 "GPU dfscf contraction" if path.name.startswith("ablation-grid-force-") else
                 "GPU occupations" if path.name.startswith("ablation-occupations-") else
                 "resident spectra" if path.name.startswith("ablation-spectra-") else
                 "GPU matrix assembly" if path.name.startswith("ablation-matrices-") else
                 "resident grid" if path.name.startswith("ablation-grid-flow-") else
                 "resident sparse SCF" if path.name.startswith("ablation-sparse-state-") else
                 "experimental block solver" if path.name.startswith("ablation-block-solver-") else
                 "GPU Hartree" if path.name.startswith("ablation-hartree-") else
                 "GPU charge normalization" if path.name.startswith("ablation-normalization-") else "GPU density construction")
        lines += ["", f"{path.stem}: direct old/new GPU scientific comparison passed={delta['passed']}; "
                  f"incremental speedup with {feature} {number(delta['end_to_end_speedup'],2)}×."]
    probe_path=root/"graph-probe.json"
    if probe_path.exists():
        probe=json.loads(probe_path.read_text())
        lines += ["", "Solver graph capability is a separate experiment; it cannot qualify SCF residency.", "",
                  "| Algorithm | Size | Complex | Isolated process | Warmup | Capture | Device loop | Numerical check | Failure stage / error |",
                  "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |"]
        for check in probe.get("checks",[]):
            fields=[check.get(k,"unmeasured") for k in ("algorithm","n","complex","isolated_process",
                    "warmup_succeeded","capture_succeeded","conditional_loop_succeeded","numerical_check_passed")]
            lines.append("| "+" | ".join(map(str,fields))+f" | {check.get('stage','')} / {check.get('error','')} |")
    if repeated:
        lines += ["", "Candidate repetitions are independent cold processes sharing one CPU reference measurement per case."]
        groups={}
        for record in ledger:
            path=record.get("candidate_path",f"{record['benchmark']}/candidate")
            if "candidate-without-" in path:
                continue
            groups.setdefault(path.split("/")[0],[]).append(record)
        lines += ["", "| Case | Candidate runs | Median seconds | Range seconds | CPU / median GPU |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name,records in groups.items():
            if len(records)<2:
                continue
            times=[r["candidate_seconds"] for r in records]
            median=statistics.median(times)
            qualified=all(r["passed"] and r["performance_comparable"] for r in records)
            speed=f"{records[0]['reference_seconds']/median:.2f}×" if qualified else "not qualified"
            lines.append(f"| {name} | {len(times)} | {median:.3f} | {min(times):.3f}–{max(times):.3f} | {speed} |")
    for record in ledger:
        name=record["benchmark"]
        paths=[root/record.get(f"{role}_path",f"{name}/{role}") for role in ("reference","candidate")]
        pair=[json.loads((path/"result.json").read_text()) for path in paths]
        profiles=[]
        for path, result in zip(paths, pair):
            timer = path/"time.json"
            profiles.append(summarize(json.loads(timer.read_text()), result["wall_seconds"])
                            if timer.exists() else (result.get("profile") or {}))
        lines += ["", f"## {name}", "", "| Metric | CPU reference | GPU candidate |", "| --- | ---: | ---: |"]
        for label, key in (("Host-controlled SCF seconds", "scf_seconds"), ("Mean host-controlled SCF iteration seconds", "average_iteration_seconds")):
            lines.append(f"| {label} | {number(profiles[0].get(key))} | {number(profiles[1].get(key))} |")
        cycle_profile=pair[1].get("profile") or {}
        if "device_cycle_seconds" in cycle_profile:
            lines += [f"| Whole device-cycle seconds | — | {number(cycle_profile.get('device_cycle_seconds'))} |",
                      f"| Mean device iteration seconds (cycle total / iterations) | — | {number(cycle_profile.get('average_device_iteration_seconds'))} |"]
        for label, key in (("Mean CPU utilization (% of one core)", "cpu_utilization_mean_percent"),
                           ("Peak host RSS (bytes)", "peak_rss_bytes"),
                           ("Mean GPU utilization (%)", "gpu_utilization_mean_percent"),
                           ("Peak GPU memory (MiB)", "gpu_memory_peak_mib")):
            lines.append(f"| {label} | {number(pair[0]['telemetry'].get(key), 1)} | {number(pair[1]['telemetry'].get(key), 1)} |")
        lines += ["", "| Component (exclusive) | CPU seconds | GPU seconds |", "| --- | ---: | ---: |"]
        a,b=[p.get("components",{}) for p in profiles]
        for component in sorted(set(a)|set(b), key=lambda k:-a.get(k,{}).get("wall_seconds",0)):
            lines.append(f"| {component} | {a.get(component,{}).get('wall_seconds',0):.3f} | {b.get(component,{}).get('wall_seconds',0):.3f} |")
        telemetry=pair[1]["telemetry"]
        for key in ("gpu_bridge","gpu_grid","gpu_density_matrix","gpu_occupations","gpu_matrix_assembly","gpu_hartree","gpu_grid_flow","gpu_sparse_state","gpu_device_cycle"):
            if key in telemetry:
                lines += ["", f"{key}: `{json.dumps(telemetry[key], sort_keys=True)}`"]
        residency=telemetry.get("residency")
        if residency:
            lines += ["", "Explicit bridge transfers by phase (any CUDA activity or filesystem trace is reported separately):", "",
                      "| Phase | Host → device bytes | Device → host bytes | Copy calls |",
                      "| --- | ---: | ---: | ---: |"]
            for phase,traffic in residency["phases"].items():
                lines.append(f"| {phase} | {traffic['host_to_device_bytes']} | {traffic['device_to_host_bytes']} | {traffic['copy_calls']} |")
            lines += ["", f"Steady-state explicit transfers ({residency.get('steady_state_definition', 'iterations > 1 per geometry')}): `{json.dumps(residency['steady_state_explicit'])}`. "
                      f"SCF intervals: {residency['scf_intervals']}; geometry epochs: {residency['geometry_epochs']}; "
                      f"accounting errors: {residency['accounting_errors']}. Full residency remains unverified."]
        io_trace=telemetry.get("io_trace")
        if io_trace:
            lines += ["", f"Filesystem trace coverage complete={io_trace['coverage_complete']}; "
                      f"errors={io_trace['errors']}. Traced timings do not qualify speedups.", "",
                      f"Phase syscall counts and bytes: `{json.dumps(io_trace.get('phases'))}`.", "",
                      f"SCF syscall types: `{json.dumps(io_trace.get('phase_operations',{}).get('scf'))}`. "
                      f"Steady-state activity: `{json.dumps(io_trace.get('steady_state'))}`. "
                      "Memory-mapped page faults and asynchronous kernel writeback remain unmeasured."]
        cuda_trace=telemetry.get("cuda_trace")
        if cuda_trace:
            lines += ["", f"CUDA activity coverage complete={cuda_trace['coverage_complete']}; "
                      f"errors={cuda_trace['errors']}. Traced timings do not qualify speedups.", "",
                      f"Phase copy/kernel activity: `{json.dumps(cuda_trace.get('phases'))}`.", "",
                      f"Steady-state CUDA activity: `{json.dumps(cuda_trace.get('steady_state'))}`. "
                      f"Process bytes beyond explicit bridge copies: `{json.dumps(cuda_trace.get('bytes_beyond_explicit'))}`. "
                      "Unified-memory migrations and mapped-host kernel access remain unmeasured."]
        if record.get("performance_reasons"):
            lines += ["", f"Performance qualification: {record['performance_reasons']}."]
        failed = [key for key, value in record["checks"].items() if not value["passed"]]
        if failed or record["reasons"]:
            lines += ["", f"Failed checks: {failed}; reasons: {record['reasons']}."]
    lines += ["", "These settings establish numerical regression only. Full device residency, production MAE convergence, "
              "complete CUDA activity tracing and optimized parallel CPU comparisons remain unvalidated.", ""]
    return "\n".join(lines)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    output=args.output or args.run/"REPORT.md"
    output.write_text(report(args.run))
    print(output)
