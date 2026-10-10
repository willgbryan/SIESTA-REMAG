"""Strict scientific regression and paired directional-energy comparisons."""
import numpy as np
from .output import read_grid

# Absolute tolerances in the named units; provisional regression gates, not physical accuracy claims.
TOLERANCES = {"energy_per_atom_ev": 1.e-5, "fermi_energy_ev": 1.e-5, "soc_energy_ev": 1.e-5,
              "forces_ev_ang": 1.e-4, "stress_ev_ang3": 1.e-5, "total_moment_muB": 1.e-4,
              "atom_moments_muB": 1.e-4, "atom_net_charges_e": 1.e-4,
              "density_components_e_bohr3": 1.e-5, "eigenvalues_ev": 1.e-4, "mae_ev": 5.e-6}

def difference(reference, candidate, tolerance):
    if reference is None or candidate is None:
        return dict(passed=False, reason="Required observable missing", max_absolute_error=None, tolerance=tolerance)
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(passed=False, reason="Shape mismatch, empty or nonfinite observable", max_absolute_error=None, tolerance=tolerance)
    error = float(np.max(np.abs(a-b)))
    return dict(passed=error <= tolerance, max_absolute_error=error, tolerance=tolerance)

def compare(reference, candidate, tolerances=None, reference_dir=None, candidate_dir=None):
    tol = {**TOLERANCES, **(tolerances or {})}
    reasons = []
    for r in (reference, candidate):
        if not r["valid"]:
            reasons.append("Run failed, did not complete, or did not converge")
    for key in ("input_sha256", "pseudopotentials", "atoms", "spin"):
        if reference["benchmark"].get(key) != candidate["benchmark"].get(key):
            reasons.append(f"Non-equivalent benchmark: {key}")
    a, b = reference["observables"], candidate["observables"]
    n = reference["benchmark"]["atoms"]
    energies = [r.get("total_energy_ev") for r in (a, b)]
    checks = {"energy_per_atom_ev": difference(*(e/n if e is not None else None for e in energies), tol["energy_per_atom_ev"])}
    fields = ["fermi_energy_ev", "forces_ev_ang", "stress_ev_ang3", "total_moment_muB",
              "atom_moments_muB", "atom_net_charges_e"]
    if reference["benchmark"]["spin"] == "spin-orbit":
        if a.get("soc_energy_representation") != b.get("soc_energy_representation"):
            reasons.append("SOC energy reporting differs")
        if a.get("soc_energy_representation") != "included_in_nonlocal":
            fields.append("soc_energy_ev")
    for field in fields:
        checks[field] = difference(a.get(field), b.get(field), tol[field])
    ea, eb = a.get("eigenvalues", {}), b.get("eigenvalues", {})
    checks["eigenvalues_ev"] = difference(ea.get("values_ev"), eb.get("values_ev"), tol["eigenvalues_ev"])
    if "relaxation_force_tolerance_ev_ang" in reference["benchmark"]:
        ga, gb = a.get("geometry", {}), b.get("geometry", {})
        checks["relaxed_positions_bohr"] = difference(ga.get("positions_bohr"), gb.get("positions_bohr"), 1.e-3)
    # Comparing integrated densities alone would miss spatial errors.
    density_check = dict(passed=False, reason="Raw RHO artifacts required for density regression")
    if reference_dir is not None and candidate_dir is not None:
        try:
            ca, ra = read_grid(reference_dir / "remag.RHO")
            cb, rb = read_grid(candidate_dir / "remag.RHO")
            density_check = difference(ra, rb, tol["density_components_e_bohr3"])
            if not np.array_equal(ca, cb):
                density_check.update(passed=False, reason="Density cells differ")
        except (OSError, ValueError, KeyError, AttributeError) as error:
            density_check = dict(passed=False, reason=str(error))
    checks["density_components_e_bohr3"] = density_check
    wants_drho=any(v.get('kind')==2 and v.get('requested') for r in (reference,candidate)
                   for v in r.get('telemetry',{}).get('final_density',[]))
    if wants_drho:
        drho_check=dict(passed=False,reason='Raw DRHO artifacts required for requested deformation-density regression')
        if reference_dir is not None and candidate_dir is not None:
            try:
                ca,ra=read_grid(reference_dir/'remag.DRHO');cb,rb=read_grid(candidate_dir/'remag.DRHO')
                drho_check=difference(ra,rb,tol['density_components_e_bohr3'])
                if not np.array_equal(ca,cb):drho_check.update(passed=False,reason='Deformation density cells differ')
            except (OSError,ValueError,KeyError,AttributeError) as error:drho_check=dict(passed=False,reason=str(error))
        checks['deformation_density_components_e_bohr3']=drho_check
    outputs = reference['benchmark'].get('energy_density_outputs')
    if outputs is not None:
        if candidate['benchmark'].get('energy_density_outputs') != outputs:
            reasons.append('Requested energy-density outputs differ')
        if reference_dir is None or candidate_dir is None:
            checks['energy_density_outputs'] = dict(passed=False, reason='Requested output artifacts required')
        else:
            from .energy_density_output import compare_outputs
            checks.update(compare_outputs(reference_dir, candidate_dir, outputs))
    passed = not reasons and all(c["passed"] for c in checks.values())
    same_hardware = reference["hardware"] == candidate["hardware"]
    traced = any(r.get("runtime_settings", {}).get("REMAG_TRACE_IO") == "1" for r in (reference, candidate))
    performance_reasons = []
    if not same_hardware:
        performance_reasons.append("Hardware or runtime allocation differs")
    if traced:
        performance_reasons.append("Filesystem tracing perturbs timing")
    if any(r.get("runtime_settings", {}).get("REMAG_TRACE_CUDA") == "1" for r in (reference, candidate)):
        performance_reasons.append("CUDA activity tracing perturbs timing")
    if any(r.get('runtime_settings', {}).get('REMAG_SCF_STAGE_PROFILE') == '1' or
           any(c.get('stage_profile') is not None for c in
               r.get('telemetry', {}).get('gpu_device_cycle', {}).get('cycles', []))
           for r in (reference, candidate)):
        performance_reasons.append('SCF stage profiling perturbs timing')
    if any(r.get("runtime_settings", {}).get("REMAG_FORCE_SHADOW") == "1" for r in (reference, candidate)):
        performance_reasons.append("Force shadow validation duplicates upstream computation")
    if any(r.get("runtime_settings", {}).get("REMAG_ATOMIC_FORCE_SHADOW") == "1" for r in (reference, candidate)):
        performance_reasons.append("Atomic-force shadow validation duplicates upstream computation")
    if any(r.get("runtime_settings", {}).get("REMAG_ORBITAL_BASIS_SHADOW") == "1" for r in (reference,candidate)):
        performance_reasons.append("Orbital derivative shadow duplicates derivative generation")
    if any(r.get("runtime_settings", {}).get("REMAG_ATOMIC_BASIS_SHADOW") == "1" for r in (reference,candidate)):
        performance_reasons.append("Atomic derivative shadow retains CPU derivatives and full comparisons")
    if any(r.get("runtime_settings", {}).get("REMAG_TWO_CENTER_SHADOW") == "1" for r in (reference,candidate)):
        performance_reasons.append("Two-center shadow retains CPU integral evaluation and full comparisons")
    if any(r.get("runtime_settings", {}).get("REMAG_NONLOCAL_H0_SHADOW") == "1" for r in (reference,candidate)):
        performance_reasons.append("H0 matrix shadow retains upstream assembly and full comparison exports")
    if any(r.get("runtime_settings", {}).get("REMAG_NONLOCAL_H0_POISON_HOST") == "1" for r in (reference,candidate)):
        performance_reasons.append("Host H0 poisoning is diagnostic")
    performance_comparable = not performance_reasons
    if any(r.get('runtime_settings',{}).get('REMAG_FINAL_DENSITY_SHADOW')=='1' for r in (reference,candidate)):
        performance_reasons.append('Final-density shadow validation retains full reference arrays')
        performance_comparable=False
    if any(r.get('runtime_settings',{}).get('REMAG_FINAL_GRID_SHADOW')=='1' for r in (reference,candidate)):
        performance_reasons.append('Final-grid shadow validation retains full reference arrays')
        performance_comparable=False
    return dict(passed=passed, reasons=reasons, checks=checks,
                end_to_end_speedup=reference["wall_seconds"]/candidate["wall_seconds"] if passed and performance_comparable else None,
                performance_comparable=performance_comparable, performance_reasons=performance_reasons,
                reference_seconds=reference["wall_seconds"], candidate_seconds=candidate["wall_seconds"],
                reference_iterations=a["scf_iterations"], candidate_iterations=b["scf_iterations"])

def compare_mae(reference_x, reference_z, candidate_x, candidate_z, tolerance_ev=5.e-6, directories=None):
    directories = directories or (None, None, None, None)
    rx, rz, cx, cz = directories
    regressions = [compare(reference_x, candidate_x, reference_dir=rx, candidate_dir=cx),
                   compare(reference_z, candidate_z, reference_dir=rz, candidate_dir=cz)]
    runs = (reference_x, reference_z, candidate_x, candidate_z)
    energies = [r["observables"].get("total_energy_ev") for r in runs]
    if any(v is None for v in energies):
        return {"passed": False, "reason": "Missing directional energy"}
    ref, cand = energies[0]-energies[1], energies[2]-energies[3]
    gate = difference(ref, cand, tolerance_ev)
    gate.update(reference_mae_ev=ref, candidate_mae_ev=cand, convention="E(x)-E(z)")
    gate["passed"] &= all(r["passed"] for r in regressions)
    signatures = [r["benchmark"].get("orientation_family_sha256") for r in runs]
    if None in signatures or len(set(signatures)) != 1:
        gate.update(passed=False, reason="Directional runs differ beyond magnetic initialization")
    if [r["benchmark"].get("theta_deg") for r in runs] != [90, 0, 90, 0]:
        gate.update(passed=False, reason="Expected x and z directional inputs")
    # Initial spins are not a constraint: verify the final moment has stayed on each requested axis.
    for run, axis in zip(runs, (0, 2, 0, 2)):
        moment = np.asarray(run["observables"].get("total_moment_muB"), dtype=float)
        if moment.shape != (3,) or not np.isfinite(moment).all() or np.linalg.norm(moment) < 1.e-8:
            gate.update(passed=False, reason="Missing or zero final magnetization")
        elif abs(moment[axis])/np.linalg.norm(moment) < np.cos(np.deg2rad(1.)):
            gate.update(passed=False, reason="Magnetization drifted from the requested axis")
    return gate
