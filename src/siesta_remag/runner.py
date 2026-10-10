"""Isolated SIESTA processes with input provenance and sampled resource telemetry."""
import csv
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from .corpus import sha256
from .output import parse_output, read_eigenvalues, read_geometry, grid_artifact, read_grid
from .profile import summarize
from .residency import summarize_residency
from .io_trace import SYSCALLS, summarize_io
from .cuda_trace import summarize_cuda
from .grid_force import record_errors as grid_force_errors
from .orbital_basis import record_errors as orbital_basis_errors
from .atomic_basis import record_errors as atomic_basis_errors
from .two_center import record_errors as two_center_errors, traffic as two_center_traffic
from .nonlocal_force import record_errors as nonlocal_force_errors, traffic as nonlocal_force_traffic
from .pair_force import record_errors as pair_force_errors, traffic as pair_force_traffic
from .atomic_force import shadow_errors as atomic_force_errors
from .final_grid import record_errors as final_grid_errors
from .final_density import record_errors as final_density_errors
from .integral_force import record_errors as integral_force_errors
from .vector_cache import record_errors as vector_cache_errors
from .population import record_errors as population_errors
from .initial_state import record_errors as initial_state_errors
from .nonlocal_h0 import audit as audit_nonlocal_h0

def run_case(case: Path, executable: Path, destination: Path, *, threads=1, timeout=1800, extra_env=None):
    if threads < 1 or timeout <= 0:
        raise ValueError("Positive thread count and timeout required")
    case, executable, destination = case.resolve(), executable.resolve(), destination.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    prepared_manifest = json.loads((case / "manifest.json").read_text())
    allowed = {"manifest.json", "input.fdf", *prepared_manifest["pseudopotentials"]}
    unexpected = {p.name for p in case.iterdir()} - allowed
    if unexpected:
        raise ValueError(f"Prepared input contains unexpected files: {sorted(unexpected)}")
    shutil.copytree(case, destination)  # refuse reuse of stale DM or output files
    manifest = json.loads((destination / "manifest.json").read_text())
    if sha256(destination / "input.fdf") != manifest["input_sha256"]:
        raise ValueError("Input differs from benchmark manifest")
    for file, digest in manifest["pseudopotentials"].items():
        if sha256(destination / file) != digest:
            raise ValueError(f"Pseudopotential checksum mismatch: {file}")
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
           "MKL_NUM_THREADS": str(threads), **(extra_env or {})}
    trace_io = env.get("REMAG_TRACE_IO") == "1"
    trace_cuda = env.get("REMAG_TRACE_CUDA") == "1"
    device_cycle = env.get("REMAG_DEVICE_SCF") == "1"
    vector_cache = env.get('REMAG_DEVICE_VECTOR_CACHE') == '1'
    if env.get('REMAG_DEVICE_VECTOR_CACHE', '0') not in ('0', '1') or (vector_cache and not device_cycle):
        raise ValueError('Vector cache requires device SCF and a zero/one mode')
    solver_lanes = env.get('REMAG_SCF_SOLVER_LANES', '1')
    if solver_lanes not in ('1', '2', '4', '8') or (solver_lanes != '1' and not device_cycle):
        raise ValueError('Parallel solver workspaces require device SCF and a count of 1, 2, 4 or 8')
    warm_start = env.get('REMAG_SCF_WARM_START', '0')
    if warm_start not in ('0', '1') or (warm_start == '1' and not device_cycle):
        raise ValueError('Eigensolver warm start requires device SCF and a zero/one mode')
    grid_force = env.get("REMAG_GRID_FORCE") == "1"
    two_center_shadow=env.get('REMAG_TWO_CENTER_SHADOW')=='1'
    pair_force_shadow=env.get('REMAG_PAIR_FORCE_SHADOW')=='1'
    nonlocal_force_shadow=env.get('REMAG_NONLOCAL_FORCE_SHADOW')=='1'
    integral_force=env.get('REMAG_INTEGRAL_FORCE')=='1'
    radial_batch=env.get('REMAG_RADIAL_BATCH','0')
    if radial_batch not in ('0','1') or (radial_batch=='1' and not integral_force):
        raise ValueError('GPU radial tables require integral forces and a zero/one mode')
    h0_shadow=env.get('REMAG_NONLOCAL_H0_SHADOW')=='1'
    h0_production=env.get('REMAG_NONLOCAL_H0')=='1';h0_active=h0_shadow or h0_production
    h0_poisoned=env.get('REMAG_NONLOCAL_H0_POISON_HOST')=='1'
    if env.get('REMAG_NONLOCAL_H0_POISON_HOST','0') not in ('0','1') or (h0_poisoned and not h0_production):
        raise ValueError('Host H0 poisoning requires production H0 and zero/one mode')
    if env.get('REMAG_NONLOCAL_H0','0') not in ('0','1') or (h0_production and
            (h0_shadow or not (device_cycle and integral_force and env.get('REMAG_GPU_INITIALIZATION')=='1'))):
        raise ValueError('Resident H0 requires exclusive GPU initialization, device SCF and integral forces')
    if env.get('REMAG_NONLOCAL_H0_SHADOW','0') not in ('0','1') or (h0_shadow and not integral_force):
        raise ValueError('H0 shadow requires authoritative integral forces and a zero/one mode')
    if env.get('REMAG_FINAL_SPECTRA','0') not in ('0','1') or (env.get('REMAG_FINAL_SPECTRA')=='1' and not (device_cycle and integral_force)):
        raise ValueError('Final spectrum ownership requires device SCF, integral forces and zero/one mode')
    if env.get('REMAG_GATE_H_EXPORT','0') not in ('0','1') or (env.get('REMAG_GATE_H_EXPORT')=='1' and not (device_cycle and integral_force)):
        raise ValueError('Gated Hamiltonian exports require device SCF, integral forces and zero/one mode')
    population=env.get('REMAG_POPULATION')=='1'
    if env.get('REMAG_POPULATION','0') not in ('0','1') or (population and not integral_force):
        raise ValueError('Population contraction requires integral forces and zero/one mode')
    if nonlocal_force_shadow and not pair_force_shadow:raise ValueError('Nonlocal shadow requires pair handoff')
    if pair_force_shadow and not (two_center_shadow and grid_force and device_cycle):raise ValueError('Pair shadow requires integral shadow and retained electronic/grid owners')
    if two_center_shadow and not grid_force:raise ValueError('Two-center material shadow requires GPU grid-force geometry evidence')
    basis_shadow=env.get('REMAG_ORBITAL_BASIS_SHADOW')=='1'
    basis_device=env.get('REMAG_ORBITAL_BASIS')=='1'
    if (basis_shadow or basis_device) and (not grid_force or (basis_shadow and basis_device)):
        raise ValueError('Choose device or shadow orbital derivatives with GPU grid forces')
    atomic_force = env.get('REMAG_ATOMIC_GRID_FORCE')=='1'
    atomic_basis_shadow=env.get('REMAG_ATOMIC_BASIS_SHADOW')=='1'
    atomic_basis_device=env.get('REMAG_ATOMIC_BASIS')=='1'
    if (atomic_basis_shadow or atomic_basis_device) and (not (grid_force and atomic_force) or (atomic_basis_shadow and atomic_basis_device)):raise ValueError('Choose device or shadow atomic derivatives with GPU forces')
    final_density=env.get('REMAG_FINAL_DENSITY')=='1'
    density_shadow=env.get('REMAG_FINAL_DENSITY_SHADOW')=='1'
    if (final_density or density_shadow) and (env.get('REMAG_FINAL_GRID')!='1' or (final_density and density_shadow)):
        raise ValueError('Choose final-density production or shadow with final-grid production')
    final_grid=env.get('REMAG_FINAL_GRID')=='1'
    final_shadow=env.get('REMAG_FINAL_GRID_SHADOW')=='1'
    if (final_grid or final_shadow) and (not atomic_force or (final_grid and final_shadow)):
        raise ValueError('Choose final-grid production or shadow with shared atomic forces')
    if atomic_force and (not grid_force or env.get('REMAG_ATOMIC_FORCE_SHADOW')=='1'):
        raise ValueError('Authoritative atomic forces require GPU dfscf without atomic shadows')
    if env.get('REMAG_ATOMIC_FORCE_SHADOW')=='1' and not grid_force:
        raise ValueError('Retained atomic-force shadows require authoritative GPU dfscf')
    if grid_force and (not device_cycle or env.get('REMAG_FORCE_SHADOW')=='1' or env.get('REMAG_FORCE_SHADOW_RESIDENT')=='1'):
        raise ValueError('Authoritative GPU grid forces require a device cycle without force shadow')
    if integral_force and (not (device_cycle and grid_force and atomic_force and final_grid and basis_device and atomic_basis_device) or
                           any((two_center_shadow, pair_force_shadow, nonlocal_force_shadow, basis_shadow, atomic_basis_shadow))):
        raise ValueError('Integral forces require authoritative device/grid/basis modes without integral references')
    if env.get('REMAG_FORCE_SHADOW_RESIDENT')=='1' and (not device_cycle or env.get('REMAG_FORCE_SHADOW')!='1'):
        raise ValueError('Resident force shadow requires a device cycle and force-shadow validation')
    if trace_cuda:
        activity = executable.parent.parent / "remag-native/libremag_activity.so"
        if not activity.is_file():
            raise RuntimeError(f"REMAG_TRACE_CUDA requires {activity}")
        env["LD_PRELOAD"] = str(activity) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
        env["REMAG_ACTIVITY_EXECUTABLE"] = str(executable)
    command = [str(executable)]
    if trace_io:
        tracer = shutil.which("strace")
        if not tracer:
            raise RuntimeError("REMAG_TRACE_IO requires strace")
        command = [tracer, "-f", "-ttt", "-T", "-yy", "-o", "syscalls.log",
                   "-e", f"trace={SYSCALLS}", *command]
    samples, timed_out = [], False
    with (destination / "input.fdf").open("rb") as inp, (destination / "stdout.log").open("wb") as out, (destination / "stderr.log").open("wb") as err:
        start = time.perf_counter()
        process = subprocess.Popen(command, stdin=inp, stdout=out, stderr=err,
                                   cwd=destination, env=env, start_new_session=True)
        # strace is the parent in diagnostic runs; do not mislabel its CPU/RSS
        # as SIESTA usage. These runs are excluded from performance qualification.
        monitor = None if trace_io else psutil.Process(process.pid)
        if monitor:
            monitor.cpu_percent()
        gpu = None
        gpu_lines, gpu_reader = [], None
        if shutil.which("nvidia-smi"):
            gpu = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,name,utilization.gpu,memory.used",
                                    "--format=csv,nounits", "--loop-ms=200"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            def collect_gpu_samples():
                with gpu.stdout:
                    gpu_lines.extend(gpu.stdout)
            # Drain continuously to avoid pipe backpressure, but keep the
            # observer's evidence in memory until the SIESTA process finishes.
            gpu_reader = threading.Thread(target=collect_gpu_samples, daemon=True)
            gpu_reader.start()
        try:
            while process.poll() is None:
                elapsed = time.perf_counter() - start
                if monitor:
                    try:
                        cpu = monitor.cpu_times()
                        samples.append({"seconds": elapsed, "cpu_percent": monitor.cpu_percent(),
                                        "rss_bytes": monitor.memory_info().rss,
                                        "cpu_seconds": cpu.user + cpu.system})
                    except psutil.NoSuchProcess:
                        break
                if elapsed > timeout:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(.1)
            returncode = process.wait()
            wall = time.perf_counter() - start
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if gpu:
                gpu.terminate()
                gpu.wait()
                gpu_reader.join()
                (destination / "gpu.csv").write_bytes(b"".join(gpu_lines))
    output = parse_output((destination / "stdout.log").read_text(errors="replace"))
    eigenfile = destination / "remag.EIG"
    parse_errors = []
    if eigenfile.exists():
        try:
            output["eigenvalues"] = read_eigenvalues(eigenfile)
        except ValueError as error:
            parse_errors.append(str(error))
    geometry = destination / "remag.XV"
    if geometry.exists():
        try:
            output["geometry"] = read_geometry(geometry)
        except ValueError as error:
            parse_errors.append(str(error))
    force_tolerance = manifest.get("relaxation_force_tolerance_ev_ang")
    if force_tolerance is not None:
        forces = output.get("forces_ev_ang")
        output["relaxation_converged"] = bool(forces) and max(abs(f) for atom in forces for f in atom) < force_tolerance
        if not output["relaxation_converged"]:
            parse_errors.append("Structural relaxation did not reach its force tolerance")
    timefile = destination / "time.json"
    profile = summarize(json.loads(timefile.read_text()), wall) if timefile.exists() else None
    telemetry = dict(cpu_samples=samples, peak_rss_bytes=max((s["rss_bytes"] for s in samples), default=None),
                     host_to_device_bytes=None, device_to_host_bytes=None, kernel_launch_count=None,
                     device_allocation_count=None, gpu_samples_file="gpu.csv" if gpu else None,
                     notes="CPU/RSS sampled at 100ms; GPU at 200ms. Unmeasured counters are null.")
    if trace_io:
        telemetry["notes"] = "Filesystem tracing perturbs timing. CPU/RSS sampling disabled; GPU sampled at 200ms. Unmeasured counters are null."
    if trace_cuda:
        telemetry["notes"] += " CUPTI tracing perturbs timing and memory use; performance is not qualified."
    gpu_samples = []
    if gpu:
        for row in csv.reader((destination / "gpu.csv").read_text().splitlines()[1:]):
            if len(row) == 4:
                try:
                    gpu_samples.append({"utilization_percent": float(row[2]), "memory_mib": float(row[3])})
                except ValueError:
                    pass
    telemetry["gpu_utilization_mean_percent"] = sum(s["utilization_percent"] for s in gpu_samples)/len(gpu_samples) if gpu_samples else None
    telemetry["gpu_memory_peak_mib"] = max((s["memory_mib"] for s in gpu_samples), default=None)
    telemetry["cpu_utilization_mean_percent"] = sum(s["cpu_percent"] for s in samples)/len(samples) if samples else None
    bridgefile = destination / "remag-gpu.json"
    if bridgefile.exists():
        telemetry["gpu_bridge"] = json.loads(bridgefile.read_text())
    gridfile = destination / "remag-grid.json"
    if gridfile.exists():
        telemetry["gpu_grid"] = json.loads(gridfile.read_text())
    dmfile = destination / "remag-dm.json"
    if dmfile.exists():
        telemetry["gpu_density_matrix"] = json.loads(dmfile.read_text())
    occupationfile = destination / "remag-occupations.json"
    if occupationfile.exists():
        telemetry["gpu_occupations"] = json.loads(occupationfile.read_text())
    matrixfile = destination / "remag-hs.json"
    if matrixfile.exists():
        telemetry["gpu_matrix_assembly"] = json.loads(matrixfile.read_text())
    hartreefile = destination / "remag-hartree.json"
    if hartreefile.exists():
        telemetry["gpu_hartree"] = json.loads(hartreefile.read_text())
    flowfile = destination / "remag-grid-flow.json"
    if flowfile.exists():
        telemetry["gpu_grid_flow"] = json.loads(flowfile.read_text())
    statefile = destination / "remag-sparse-state.json"
    if statefile.exists():
        telemetry["gpu_sparse_state"] = json.loads(statefile.read_text())
    cyclefile=destination / "remag-device-scf.json"
    if device_cycle:
        if not cyclefile.exists():
            parse_errors.append("Device-cycle publication evidence is missing")
        else:
            cycles=json.loads(cyclefile.read_text())
            telemetry["gpu_device_cycle"]=cycles
            rows=cycles.get("cycles",[])
            cycles["host_to_device_bytes"]=sum(r["setup_bytes"] for r in rows)
            cycles["device_to_host_bytes"]=sum(r["export_bytes"] for r in rows)
            if not rows or sum(r["iterations"] for r in rows)!=output["scf_iterations"]:
                parse_errors.append("Device iteration count differs from the published SCF log")
            if any(r["status"] or not r["converged"] or r["forbidden_nodes"] or r["graph_nodes"]<1 for r in rows):
                parse_errors.append("Device cycle failed, did not converge or has invalid graph evidence")
            parse_errors.extend(vector_cache_errors(rows, output.get('eigenvalues'), manifest, vector_cache, int(solver_lanes), warm_start == '1'))
            profile["device_cycle_seconds"]=sum(r["seconds"] for r in rows)
            profile["device_iterations"]=sum(r["iterations"] for r in rows)
            profile["average_device_iteration_seconds"]=(profile["device_cycle_seconds"]/profile["device_iterations"]
                if profile["device_iterations"] else None)
            profile["device_cycle_timing_scope"]="whole GPU execution region; individual iteration times unavailable"
    if final_grid or final_shadow:
        path=destination/'remag-final-grid.json'
        records=json.loads(path.read_text()).get('cycles',[]) if path.is_file() else []
        for record in records:parse_errors.extend(final_grid_errors(record,destination,final_shadow))
        cycles=telemetry.get('gpu_device_cycle',{}).get('cycles',[])
        if not records or len(records)!=len(cycles) or any(a['epoch']>=b['epoch'] for a,b in zip(records,records[1:])):
            parse_errors.append('Final force-grid handoff does not cover every geometry')
        telemetry['final_grid']=records
    if final_density or density_shadow:
        path=destination/'remag-final-density.json'
        records=json.loads(path.read_text()).get('cycles',[]) if path.is_file() else []
        for record in records:parse_errors.extend(final_density_errors(record,destination,density_shadow))
        final_rows=telemetry.get('final_grid',[])
        expected=[(r['epoch'],kind) for r in final_rows for kind in (1,2)]
        if not records or [(r['epoch'],r['kind']) for r in records]!=expected:
            parse_errors.append('Retained output density does not cover every geometry/kind')
        for kind,suffix in ((1,'RHO'),(2,'DRHO')):
            rows=[r for r in records if r['kind']==kind];exports=[r for r in rows if r['exported']]
            if rows and rows[-1]['requested']:
                if len(exports)!=1 or exports[0]['epoch']!=rows[-1]['epoch']:
                    parse_errors.append('Requested final '+suffix+' export missing, duplicated or not final')
                try:
                    path=destination/('remag.'+suffix)
                    artifact=grid_artifact(path)
                    netcdf=manifest.get('energy_density_outputs',{}).get('netcdf',False)
                    expected=destination/('remag.nc' if netcdf else 'remag.'+suffix)
                    if artifact!=expected:parse_errors.append('Requested '+suffix+' output format differs')
                    if len(list(destination.glob('*.'+suffix)))!=(0 if netcdf else 1):
                        parse_errors.append('Requested '+suffix+' file coverage differs')
                    _,field=read_grid(path)
                    if list(field.shape)!=[rows[-1]['spins'],*reversed(rows[-1]['mesh'])]:
                        parse_errors.append('Requested '+suffix+' output shape differs')
                except (OSError,ValueError,KeyError,AttributeError) as exc:
                    parse_errors.append('Requested final '+suffix+' output invalid: '+str(exc))
            elif exports:parse_errors.append('Unrequested '+suffix+' export')
        telemetry['final_density']=records
        telemetry['gpu_final_density']=dict(host_to_device_bytes=0,device_to_host_bytes=sum(r['device_to_host_bytes'] for r in records))
    if integral_force:
        records=[json.loads(p.read_text()) for p in sorted(destination.glob('remag-integral-force-*.json'))]
        grids=[json.loads(p.read_text()) for p in sorted(destination.glob('remag-integral-grid-*.json'))]
        owners={}
        for key, name in (('tables','remag-two-center-tables.json'),('pairs','remag-pair-geometry.json'),('nonlocal','remag-nonlocal-geometry.json')):
            p=destination/name;owners[key]=json.loads(p.read_text()) if p.is_file() else {}
        epochs=[r.get('epoch') for r in telemetry.get('gpu_device_cycle',{}).get('cycles',[])]
        if not epochs or any(type(epoch) is not int or epoch<1 for epoch in epochs):
            parse_errors.append('Integral forces require the device-cycle geometry epochs')
        parse_errors.extend(integral_force_errors(records,grids,owners['tables'],owners['pairs'],owners['nonlocal'],epochs,manifest['atoms'],manifest['spin']=='spin-orbit',env.get('REMAG_GPU_HISTORY')=='1',h0_active,compact_queries=h0_production))
        telemetry['integral_force']=records;telemetry['integral_geometry']=owners;telemetry['grid_force']=grids
        telemetry['gpu_integral_force']=dict(host_to_device_bytes=sum(g['setup_bytes'] for g in grids)+sum(c['setup_bytes'] for r in records for c in r['consumers'])+
            sum(t['setup_bytes'] for t in owners['tables'].get('tables',[]))+sum(p['setup_bytes'] for p in owners['pairs'].get('cycles',[]))+
            sum(n['geometry_bytes'] for n in owners['nonlocal'].get('cycles',[])),device_to_host_bytes=sum(r['export_bytes'] for r in records))
        radial=owners['tables'].get('radial_setup',{})
        for key in ('host_to_device_bytes','device_to_host_bytes'):
            telemetry['gpu_integral_force'][key]+=radial.get(key,0)
        if env.get('REMAG_RADIAL_BATCH')=='1' and radial.get('calls',0)<2:
            parse_errors.append('GPU radial table setup did not execute')
        from .terminal_exports import record_errors as terminal_export_errors, requested_energy_exports, requested_density_exports, requested_hamiltonian_exports
        cycle_data=telemetry.get('gpu_device_cycle',{})
        if cycle_data.get('schema')!=3:parse_errors.append('Terminal matrix export schema is missing')
        export_reasons=requested_energy_exports((destination/'input.fdf').read_text(),True)
        h_reasons=requested_hamiltonian_exports((destination/'input.fdf').read_text(),env.get('REMAG_GATE_H_EXPORT')=='1')
        density_reasons=requested_density_exports((destination/'input.fdf').read_text(),env.get('REMAG_GPU_HISTORY')=='1',
            env.get('REMAG_FINAL_GRID_SHADOW')=='1' or env.get('REMAG_FINAL_DENSITY_SHADOW')=='1')
        parse_errors.extend(terminal_export_errors(cycle_data.get('cycles',[]),telemetry.get('final_grid',[]),True,export_reasons,
                                                  env.get('REMAG_GPU_INITIALIZATION') == '1',density_reasons,env.get('REMAG_FINAL_SPECTRA') == '1',h_reasons))
    elif grid_force:
        records=[]
        for path in sorted(destination.glob("remag-grid-force-*.json")):
            record=json.loads(path.read_text());records.append(record)
            parse_errors.extend(grid_force_errors(record,manifest['atoms']))
            if bool(record.get('atomic_basis_shadow'))!=atomic_basis_shadow or bool(record.get('atomic_basis_device'))!=atomic_basis_device:parse_errors.append('Atomic derivative mode differs from force record')
            if bool(record.get('orbital_basis_shadow'))!=basis_shadow or bool(record.get('orbital_basis_device'))!=basis_device:
                parse_errors.append('Orbital derivative mode differs from force record')
            if (record.get('atomic_terms_authoritative') is True)!=atomic_force:
                parse_errors.append('Atomic-force publication differs from the requested production mode')
        cycles=telemetry.get("gpu_device_cycle",{}).get("cycles",[])
        if not records or len(records)!=len(cycles) or any(a['epoch']>=b['epoch'] for a,b in zip(records,records[1:])):
            parse_errors.append('Authoritative grid forces do not cover every device geometry')
        telemetry['grid_force']=records
        telemetry['gpu_grid_force']=dict(host_to_device_bytes=sum(r['setup_bytes'] for r in records),
                                        device_to_host_bytes=sum(r['export_bytes'] for r in records))
    if env.get('REMAG_FINAL_SPECTRA')=='1':
        from .final_spectrum import record_errors as spectrum_errors
        path=destination/'remag-final-spectrum.json'
        data=json.loads(path.read_text()) if path.is_file() else {}
        records=data.get('cycles',[])
        if data.get('schema')!=1:parse_errors.append('Missing final spectrum schema')
        parse_errors.extend(spectrum_errors(records,telemetry.get('gpu_device_cycle',{}).get('cycles',[]),
                                            telemetry.get('integral_force',[])))
        telemetry['final_spectrum']=records
        telemetry['gpu_final_spectrum']=dict(host_to_device_bytes=0,
            device_to_host_bytes=sum(r['eigenvalue_export_bytes']+r['occupation_export_bytes'] for r in records))
    histories=None
    if env.get('REMAG_GPU_HISTORY')=='1':
        path=destination/'remag-density-history.json'
        data=json.loads(path.read_text()) if path.is_file() else {}
        histories=data.get('cycles',[])
        if data.get('schema')!=1:parse_errors.append('Missing production history schema')
    if population:
        path=destination/'remag-population.json'
        data=json.loads(path.read_text()) if path.is_file() else {}
        records=data.get('cycles',[])
        if data.get('schema')!=2:parse_errors.append('Missing population publication schema')
        parse_errors.extend(population_errors(records,telemetry.get('gpu_device_cycle',{}).get('cycles',[]),manifest['atoms'],output.get('eigenvalues',{}),len(manifest['pseudopotentials'])))
        telemetry['population']=records
        telemetry['gpu_population']=dict(host_to_device_bytes=sum(r['setup_bytes'] for r in records),device_to_host_bytes=sum(sum(r['export_bytes']) for r in records))
        if env.get('REMAG_GPU_INITIALIZATION') == '1':
            path=destination/'remag-initial-state.json'
            data=json.loads(path.read_text()) if path.is_file() else {}
            records=data.get('cycles',[])
            if data.get('schema')!=1:parse_errors.append('Missing initial density ownership schema')
            parse_errors.extend(initial_state_errors(records,telemetry.get('gpu_device_cycle',{}).get('cycles',[]),manifest['atoms'],output.get('eigenvalues',{}),len(manifest['pseudopotentials']),histories))
            telemetry['initial_state']=records
            telemetry['gpu_initial_state']=dict(host_to_device_bytes=sum(r['density_upload_bytes']+r['population_setup_bytes'] for r in records),
                device_to_host_bytes=sum(r['status_export_bytes']+sum(r['export_bytes']) for r in records))
    if histories is not None:
        from .density_history import record_errors as history_errors, policy as history_policy
        parse_errors.extend(history_errors(histories,telemetry.get('gpu_device_cycle',{}).get('cycles',[]),
            telemetry.get('initial_state',[]),telemetry.get('integral_force',[]),history_policy((destination/'input.fdf').read_text())))
        telemetry['density_history']=histories
        telemetry['gpu_density_history']=dict(host_to_device_bytes=sum(r['setup_bytes'] for r in histories),
            device_to_host_bytes=sum(r['initial_status_bytes']+r['post_status_bytes']+r['coefficient_export_bytes'] for r in histories))
    if two_center_shadow:
        path=destination/'remag-two-center.json'
        report=json.loads(path.read_text()) if path.is_file() else {}
        epochs=[r['epoch'] for r in telemetry.get('grid_force',[])]
        parse_errors.extend(two_center_errors(report,epochs,manifest['spin']=='spin-orbit',pair_force_shadow,nonlocal_force_shadow,pair_force_shadow))
        telemetry['two_center']=report
        telemetry['gpu_two_center']=two_center_traffic(report)
    if pair_force_shadow:
        path=destination/'remag-pair-force.json'
        report=json.loads(path.read_text()) if path.is_file() else {}
        path=destination/'remag-pair-geometry.json'
        geometry=json.loads(path.read_text()) if path.is_file() else {}
        parse_errors.extend(pair_force_errors(report,telemetry.get('two_center',{}),telemetry.get('grid_force',[]),geometry))
        telemetry['pair_force']=report
        telemetry['pair_geometry']=geometry
        telemetry['gpu_pair_force']=pair_force_traffic(report,geometry)
    if nonlocal_force_shadow:
        path=destination/'remag-nonlocal-force.json'
        report=json.loads(path.read_text()) if path.is_file() else {}
        parse_errors.extend(nonlocal_force_errors(report,telemetry.get('two_center',{}),telemetry.get('grid_force',[]),manifest['spin']=='spin-orbit'))
        telemetry['nonlocal_force']=report
        telemetry['gpu_nonlocal_force']=nonlocal_force_traffic(report)
    if basis_shadow or basis_device:
        path=destination/'remag-orbital-basis.json'
        records=json.loads(path.read_text()).get('cycles',[]) if path.is_file() else []
        forces=telemetry.get('grid_force',[])
        if not records or [r['epoch'] for r in records]!=[r['epoch'] for r in forces]:
            parse_errors.append('Orbital basis owner does not cover every geometry')
        for record,force in zip(records,forces):parse_errors.extend(orbital_basis_errors(record,force,basis_shadow))
        telemetry['orbital_basis']=records
        telemetry['gpu_orbital_basis']=dict(host_to_device_bytes=sum(r['compact_setup_bytes']+r['shadow_setup_bytes'] for r in records),
                                           device_to_host_bytes=sum(r['shadow_export_bytes'] for r in records))
    if atomic_basis_shadow or atomic_basis_device:
        path=destination/'remag-atomic-basis.json'
        report=json.loads(path.read_text()) if path.is_file() else {}
        records=report.get('cycles',[]);forces=telemetry.get('grid_force',[])
        if report.get('atol')!=1e-10 or report.get('rtol')!=1e-10 or report.get('displacements_exact') is not True:
            parse_errors.append('Atomic derivative comparison tolerances differ')
        if not records or [r['epoch'] for r in records]!=[r['epoch'] for r in forces]:parse_errors.append('Atomic basis owner does not cover every geometry')
        for record,force in zip(records,forces):parse_errors.extend(atomic_basis_errors(record,force,atomic_basis_shadow))
        telemetry['atomic_basis']=records
        telemetry['gpu_atomic_basis']=dict(host_to_device_bytes=sum(r['compact_setup_bytes']+sum(t['shadow_setup_bytes'] for t in r['terms']) for r in records),
                                           device_to_host_bytes=sum(t['shadow_export_bytes'] for r in records for t in r['terms']))
    if env.get('REMAG_ATOMIC_FORCE_SHADOW')=='1':
        records=[json.loads(p.read_text()) for p in sorted(destination.glob('remag-atomic-force-shadow-*.json'))]
        parse_errors.extend(atomic_force_errors(records,telemetry.get('grid_force',[]),manifest['atoms']))
        telemetry['atomic_force_shadow']=records
        telemetry['gpu_atomic_grid_force_shadow']=dict(host_to_device_bytes=sum(r['setup_bytes'] for r in records),
                                                      device_to_host_bytes=sum(r['export_bytes'] for r in records))
    if env.get("REMAG_FORCE_SHADOW") == "1":
        records=[]
        for path in sorted(destination.glob("remag-force-shadow-*.json")):
            record=json.loads(path.read_text());records.append(record)
            for name,tolerance in (("force",1.e-10),("stress",1.e-12)):
                expected=record.get("reference_"+name,[]);actual=record.get("device_"+name,[])
                count=3*manifest["atoms"] if name=="force" else 9
                if len(expected)!=count or len(actual)!=count or any(
                    not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or
                    not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>=tolerance
                    for a,b in zip(expected,actual)):
                    parse_errors.append("Grid-force shadow differs from upstream "+name)
            if not record.get("passed") or record.get("status")!=0 or record.get("graph_nodes")!=1:
                parse_errors.append("Grid-force shadow did not pass")
            if env.get('REMAG_FORCE_SHADOW_RESIDENT')=='1' and (record.get('electronic_inputs_resident') is not True or not record.get('retained_electronic_bytes')):
                parse_errors.append('Final force did not consume retained electronic fields')
        if not records:parse_errors.append("Requested grid-force shadow evidence is missing")
        # Geometry epochs count mesh invalidations; they are not MD/CG step
        # indices. This diagnostic supports one final dfscf call per cycle.
        if device_cycle and (len(records)!=len(telemetry.get("gpu_device_cycle",{}).get("cycles",[])) or
            any(a["epoch"]>=b["epoch"] for a,b in zip(records,records[1:]))):
            parse_errors.append("Grid-force shadow does not cover every device geometry")
        telemetry["grid_force_shadow"]=records
        telemetry["gpu_grid_force_shadow"]=dict(
            host_to_device_bytes=sum(r['setup_bytes'] for r in records),
            device_to_host_bytes=sum(r['export_bytes'] for r in records))
    if env.get("REMAG_BLOCK_SOLVER") == "1" and not device_cycle:
        counts=telemetry.get("gpu_bridge",{})
        if not counts.get("calls") or counts.get("experimental_block_calls")!=counts.get("calls"):
            parse_errors.append("Experimental block solver did not cover every electronic solve")
        if not counts.get("experimental_block_setups") or not counts.get("experimental_block_workspace_peak_bytes"):
            parse_errors.append("Experimental block solver graph/workspace evidence is missing")
    if env.get("REMAG_HS") == "1" and not device_cycle:
        counts=telemetry.get("gpu_matrix_assembly",{})
        if any(counts.get(key)!=output["scf_iterations"] for key in ("prepares","ends")):
            parse_errors.append("Device matrix assembly lifecycle does not cover every SCF iteration")
    if env.get("REMAG_OCCUPATIONS") == "1" and not device_cycle:
        counts=telemetry.get("gpu_occupations",{})
        if any(counts.get(key)!=output["scf_iterations"] for key in ("device_occupation_calls","prepares","ends")):
            parse_errors.append("Device occupation lifecycle does not cover every SCF iteration")
    if env.get("REMAG_NORMALIZE") == "1" and not device_cycle:
        count = telemetry.get("gpu_density_matrix", {}).get("device_charge_normalizations")
        if count != output["scf_iterations"]:
            parse_errors.append(f"Device normalization coverage: {count} calls for {output['scf_iterations']} SCF iterations")
    if h0_active:
        p=destination/'remag-nonlocal-h0.json';report=json.loads(p.read_text()) if p.is_file() else {}
        raw=destination/'remag-residency.json';trace=json.loads(raw.read_text()) if raw.is_file() else {}
        epochs=[c.get('epoch') for c in telemetry.get('gpu_device_cycle',{}).get('cycles',[])]
        h0_audit=audit_nonlocal_h0(report,destination,epochs,telemetry.get('integral_geometry',{}).get('nonlocal',{}),trace,production=h0_production,cycle_records=telemetry.get('gpu_device_cycle',{}).get('cycles',[]),poisoned=h0_poisoned,compact_queries=h0_production)
        parse_errors.extend(h0_audit['errors']);telemetry['nonlocal_h0']=report;telemetry['nonlocal_h0_audit']=h0_audit
        telemetry['gpu_nonlocal_h0']=h0_audit.get('traffic',{})
    if env.get('REMAG_INITIAL_MESH')=='1':
        path=destination/'remag-initial-mesh-production.json'
        mesh=json.loads(path.read_text()) if path.is_file() else {}
        rows=mesh.get('cycles',[]);cycles=telemetry.get('gpu_device_cycle',{}).get('cycles',[])
        if not rows or [r.get('epoch') for r in rows]!=[c.get('epoch') for c in cycles] or any(
                not all(r.get(k) is True for k in ('launched','phi_bound','density_borrowed','density_released','forces_complete','passed','converged'))
                or r.get('cycle_status')!=0 or r.get('cycle_producer_status')!=0 for r in rows):
            parse_errors.append('Initial GPU mesh did not cover completed SCF and forces')
        telemetry['initial_mesh']=mesh
        telemetry['gpu_initial_mesh']=dict(host_to_device_bytes=sum(r.get('setup_bytes',0) for r in rows),
                                          device_to_host_bytes=sum(r.get('export_bytes',0) for r in rows))
    residencyfile = destination / "remag-residency.json"
    residency_trace = {}
    if any(env.get(key)=="1" for key in ("REMAG_OCCUPATIONS","REMAG_HS","REMAG_HARTREE","REMAG_RESIDENT_GRID","REMAG_RESIDENT_SCF")) and not residencyfile.exists():
        parse_errors.append("Device consumer transfer accounting is missing")
    if residencyfile.exists():
        residency_trace = json.loads(residencyfile.read_text())
        if device_cycle:
            residency_trace["execution_mode"]="device_cycle"
        telemetry["residency"] = summarize_residency(residency_trace, telemetry,
                                                      output["scf_iterations"])
        parse_errors.extend(telemetry["residency"]["accounting_errors"])
        if device_cycle:
            rows=telemetry.get("gpu_device_cycle",{}).get("cycles",[])
            regions=[i for i in residency_trace["intervals"] if i["phase"]=="scf"]
            if [i["geometry"] for i in regions]!=[r["geometry"] for r in rows] or any(i["iteration"]!=1 for i in regions):
                parse_errors.append("Whole-cycle launch regions do not match published geometry cycles")
            if any(v for i in regions for t in i["boundaries"].values() for v in t.values()):
                parse_errors.append("An explicit host/device transfer occurred inside the complete device cycle")

        if env.get("REMAG_RESIDENT_GRID") == "1" and not device_cycle:
            scf=[i for i in residency_trace["intervals"] if i["phase"]=="scf"]
            counts=telemetry.get("gpu_grid_flow",{})
            grid_counts=telemetry.get("gpu_grid",{})
            if any(counts.get(k)!=output["scf_iterations"] for k in ("arms","solves","ends")):
                parse_errors.append("Resident grid lifecycle does not cover every SCF iteration")
            if any(grid_counts.get(k)!=output["scf_iterations"] for k in
                   ("resident_grid_density_producers","resident_grid_potential_consumers")):
                parse_errors.append("Resident grid producers/consumers do not cover every SCF iteration")
            if counts.get("geometry_setups")!=sum(i["iteration"]==1 for i in scf):
                parse_errors.append("Resident grid geometry setup is missing or repeated within SCF")
            for interval in scf:
                traffic=interval["boundaries"]
                if traffic.get("grid_flow_result",{}).get("device_to_host_bytes")!=224:
                    parse_errors.append("Resident grid scalar-result boundary is missing or duplicated")
                if any(traffic.get(k,{}).get(direction,0) for k,direction in
                       (("grid_potential_input","host_to_device_bytes"),("hartree_density_input","host_to_device_bytes"),
                        ("hartree_potential_output","device_to_host_bytes"),("hartree_result","device_to_host_bytes"))):
                    parse_errors.append("Resident grid used a host potential/Hartree boundary inside SCF")
                matrix_bytes=traffic.get("grid_sparse_input",{}).get("host_to_device_bytes",0)
                if env.get("REMAG_RESIDENT_SCF") == "1" and not device_cycle:
                    if matrix_bytes or traffic.get("grid_output",{}).get("device_to_host_bytes",0):
                        parse_errors.append("Resident sparse H crossed the SCF grid boundary")
                elif not matrix_bytes or traffic.get("grid_output",{}).get("device_to_host_bytes")!=matrix_bytes:
                    parse_errors.append("SCF grid downloads are not limited to the remaining sparse-matrix output")
                if interval["iteration"]>1 and traffic.get("grid_flow_geometry",{}).get("copy_calls",0):
                    parse_errors.append("Resident grid geometry restaged after its first SCF iteration")
        if env.get("REMAG_RESIDENT_SCF") == "1" and not device_cycle:
            scf=[i for i in residency_trace["intervals"] if i["phase"]=="scf"]
            cycles=sum(i["iteration"]==1 for i in scf);counts=telemetry.get("gpu_sparse_state",{})
            covered=("iterations","density_publications","seed_prepares","grid_commits","hamiltonian_finishes",
                     "scalar_reductions","hs_borrows","density_borrows")
            if any(counts.get(k)!=output["scf_iterations"] for k in covered):
                parse_errors.append("Sparse-state producers/consumers do not cover every SCF iteration")
            if counts.get("setups")!=cycles or counts.get("exports")!=cycles or counts.get("mixes")!=len(scf)-cycles:
                parse_errors.append("Sparse-state initialization/mixing/export lifecycle does not match the SCF cycles")
            last={i["geometry"]:i["iteration"] for i in scf}
            for interval in scf:
                traffic=interval["boundaries"]
                if any(traffic.get(k,{}).get("copy_calls",0) for k in
                       ("solver_matrices","solver_eigenvalues","solver_eigenvectors","dm_outputs","grid_density_input",
                        "grid_sparse_input","grid_output","sparse_hamiltonian_input","occupation_outputs","spectral_export",
                        "sparse_state_setup","sparse_state_export")):
                    parse_errors.append("Owned electronic arrays crossed an SCF boundary")
                if traffic.get("sparse_state_scalars",{}).get("device_to_host_bytes")!=144:
                    parse_errors.append("Sparse scalar summary is missing or repeated")
                expected=0 if interval["iteration"]==last[interval["geometry"]] else 4
                if traffic.get("sparse_state_mixer_status",{}).get("device_to_host_bytes",0)!=expected:
                    parse_errors.append("Sparse mixer status does not match the nonconverged iterations")
        if env.get("REMAG_HARTREE") == "1" and env.get("REMAG_RESIDENT_GRID") != "1":
            scf=[i for i in residency_trace["intervals"] if i["phase"]=="scf"]
            counts=telemetry.get("gpu_hartree",{})
            if not counts.get("calls") or not counts.get("geometry_setups"):
                parse_errors.append("Device Hartree calls or geometry setup are missing")
            if any(not i["boundaries"].get("hartree_density_input",{}).get("host_to_device_bytes") or
                   not i["boundaries"].get("hartree_potential_output",{}).get("device_to_host_bytes") or
                   i["boundaries"].get("hartree_result",{}).get("device_to_host_bytes",0)<88 for i in scf):
                parse_errors.append("Device Hartree accounting does not cover every SCF iteration")
        if env.get("REMAG_HS") == "1" and not device_cycle:
            traffic=telemetry["residency"]["scf_boundaries"]
            if traffic.get("solver_matrices",{}).get("host_to_device_bytes",0):
                parse_errors.append("SCF solver uploaded dense matrices with device assembly enabled")
            counts=telemetry.get("gpu_matrix_assembly",{})
            if not counts.get("matrix_builds") or counts.get("device_to_host_bytes")!=0:
                parse_errors.append("Device matrix assembly is missing or downloaded matrix data")
        if env.get("REMAG_OCCUPATIONS") == "1" and not device_cycle:
            traffic=telemetry["residency"]["scf_boundaries"]
            if any(traffic.get(name,{}).get("host_to_device_bytes",0) for name in ("dm_occupations","dm_energy_weights")):
                parse_errors.append("SCF density construction uploaded declared resident occupations or energy weights")
            if env.get("REMAG_RESIDENT_SPECTRA")=="1":
                counts=telemetry.get("gpu_occupations",{})
                cycles=sum(v["phase"]=="scf" and v["iteration"]==1 for v in residency_trace["intervals"])
                if counts.get("deferred_occupation_calls")!=output["scf_iterations"] or counts.get("deferred_spectral_exports")!=cycles:
                    parse_errors.append("Deferred spectral export does not cover every SCF cycle")
                if any(traffic.get(name,{}).get("device_to_host_bytes",0) for name in
                       ("solver_eigenvalues","occupation_outputs","spectral_export")):
                    parse_errors.append("Resident spectra downloaded an eigenvalue/occupation array inside SCF")
    if trace_io:
        syscallfile = destination / "syscalls.log"
        if syscallfile.exists():
            with syscallfile.open(errors="replace") as lines:
                telemetry["io_trace"] = summarize_io(lines, residency_trace,
                                                       process_succeeded=returncode == 0 and not timed_out)
        else:
            telemetry["io_trace"] = dict(coverage_complete=False, strict_no_io_verified=False,
                                          errors=["Requested syscall trace missing"])
    if trace_cuda:
        activityfile = destination / "remag-activity.json"
        try:
            activity = json.loads(activityfile.read_text())
            telemetry["cuda_trace"] = summarize_cuda(activity, telemetry.get("residency", {}),
                expected_scf_iterations=(len(telemetry.get("gpu_device_cycle",{}).get("cycles",[])) if device_cycle else output["scf_iterations"]),
                process_succeeded=returncode == 0 and not timed_out,
                allow_unique_graph_scope=device_cycle,memory_only=device_cycle)
        except (OSError, ValueError) as error:
            telemetry["cuda_trace"] = dict(coverage_complete=False, strict_residency_verified=False,
                                            errors=[f"Requested CUDA trace unavailable: {error}"])
    provenance = None
    for parent in Path(__file__).resolve().parents:
        lock = parent / "upstream.lock.json"
        if lock.exists():
            provenance = json.loads(lock.read_text())
            break
    result = dict(schema_version=1, benchmark=manifest, created_at=datetime.now(timezone.utc).isoformat(),
                  executable={"path": str(executable), "sha256": sha256(executable)},
                  hardware={"platform": platform.platform(), "machine": platform.machine(),
                            "cpu_count": os.cpu_count(), "blas_threads": threads,
                            "hostname": platform.node(),
                            "runtime_instance_id": env.get("REMAG_HOST_ID", platform.node())},
                  upstream=provenance, parse_errors=parse_errors,
                  runtime_settings={key: env.get(key) for key in
                                    ("REMAG_INITIAL_MESH", "REMAG_INITIAL_MESH_POISON_HOST", "REMAG_INITIAL_MESH_SHADOW", "REMAG_NONLOCAL_H0_POISON_HOST", "REMAG_NONLOCAL_H0", "REMAG_NONLOCAL_H0_SHADOW", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "REMAG_GPU", "REMAG_GRID", "REMAG_DM",
                                     "REMAG_GRID_POTENTIAL_SPINS", "REMAG_GRID_EPOCH", "REMAG_GRID_VERIFY_EPOCH", "REMAG_NORMALIZE", "REMAG_OCCUPATIONS", "REMAG_RESIDENT_SPECTRA", "REMAG_HS", "REMAG_HARTREE", "REMAG_RESIDENT_GRID", "REMAG_RESIDENT_SCF", "REMAG_DEVICE_SCF", "REMAG_DEVICE_VECTOR_CACHE", "REMAG_BLOCK_SOLVER", "REMAG_BLOCK_WIDTH", "REMAG_BLOCK_LOCAL_SOLVER", "REMAG_BLOCK_SHARED_JACOBI", "REMAG_BLOCK_FUSED_MATRIX", "REMAG_BLOCK_FP64_VECTORS", "REMAG_BLOCK_FP64_MATRIX", "REMAG_FORCE_SHADOW", "REMAG_FORCE_SHADOW_RESIDENT", "REMAG_GRID_FORCE", "REMAG_ATOMIC_GRID_FORCE", "REMAG_ATOMIC_FORCE_SHADOW", "REMAG_FINAL_GRID", "REMAG_FINAL_GRID_SHADOW", "REMAG_FINAL_DENSITY", "REMAG_FINAL_DENSITY_SHADOW", "REMAG_ORBITAL_BASIS_SHADOW", "REMAG_ORBITAL_BASIS", "REMAG_ATOMIC_BASIS_SHADOW", "REMAG_ATOMIC_BASIS", "REMAG_TWO_CENTER_SHADOW", "REMAG_PAIR_FORCE_SHADOW", "REMAG_NONLOCAL_FORCE_SHADOW", "REMAG_INTEGRAL_FORCE", "REMAG_POPULATION", "REMAG_TRACE_IO", "REMAG_TRACE_CUDA", "REMAG_SCF_STAGE_PROFILE", "REMAG_SCF_SOLVER_LANES", "REMAG_SCF_WARM_START", "REMAG_RADIAL_BATCH",
                                     "REMAG_GPU_INITIALIZATION", "REMAG_GPU_HISTORY", "REMAG_FINAL_SPECTRA", "REMAG_GATE_H_EXPORT", "REMAG_DEFER_SCF_LOG", "GFORTRAN_FORMATTED_BUFFER_SIZE",
                                     "GFORTRAN_UNBUFFERED_ALL", "GFORTRAN_UNBUFFERED_PRECONNECTED")},
                  returncode=returncode, timed_out=timed_out, wall_seconds=wall,
                  valid=returncode == 0 and not timed_out and not parse_errors and output["converged"] and output["completed"],
                  observables=output, profile=profile, telemetry=telemetry)
    (destination / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
