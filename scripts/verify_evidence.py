#!/usr/bin/env python3
"""Verify a run's retained source archive against its recorded SHA256 hashes."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def verify(run):
    metadata = json.loads((run / "metadata.json").read_text())
    expected = {}
    for source, digest in metadata["implementation_sha256"].items():
        relative = Path(source).relative_to("/opt/remag")
        if relative.name == "patch_siesta.py":
            relative = Path("scripts") / relative.name
        expected[str(relative)] = digest
    expected.update({f"src/siesta_remag/{name}": digest for name, digest in metadata["harness_sha256"].items()})
    for field, path in (("native_check_sha256", "scripts/check_gpu_solver.py"),
                        ("fermi_check_sha256", "scripts/check_fermi.py"),
                        ("hartree_check_sha256", "scripts/check_hartree.py"),
                        ("pbe_oracle_sha256", "scripts/check_pbe.py"),
                        ("pbe_check_sha256", "scripts/check_pbe_gpu.py"),
                        ("grid_potential_check_sha256", "scripts/check_grid_potential.py"),
                        ("grid_force_oracle_sha256", "scripts/check_grid_force_formula.py"),
                        ("grid_force_check_sha256", "scripts/check_grid_force_gpu.py"),
                        ("atomic_grid_force_oracle_sha256", "scripts/check_atomic_grid_force_formula.py"),
                        ("atomic_grid_force_check_sha256", "scripts/check_atomic_grid_force_gpu.py"),
                        ("orbital_basis_oracle_sha256", "scripts/check_orbital_basis.py"),
                        ("atomic_basis_oracle_sha256", "scripts/check_atomic_basis.py"),
                        ("atomic_basis_check_sha256", "scripts/check_atomic_basis_gpu.py"),
                        ("atomic_comparison_check_sha256", "scripts/check_atomic_comparison.py"),
                        ("two_center_comparison_check_sha256", "scripts/check_two_center_comparison.py"),
                        ("two_center_owner_check_sha256", "scripts/check_two_center_owner.py"),
                        ("pair_handoff_check_sha256", "scripts/check_pair_handoff.py"),
                        ("pair_force_oracle_sha256", "scripts/check_pair_force.py"),
                        ("pair_force_check_sha256", "scripts/check_pair_force_gpu.py"),
                        ("population_oracle_sha256", "scripts/check_population.py"),
                        ("population_check_sha256", "scripts/check_population_gpu.py"),
                        ("population_owner_check_sha256", "scripts/check_population_owner.py"),
                        ("initial_state_check_sha256", "scripts/check_initial_state.py"),
                        ("population_observables_check_sha256", "scripts/check_population_observables.py"),
                        ("soc_projector_oracle_sha256", "scripts/check_soc_projector.py"),
                        ("soc_projector_check_sha256", "scripts/check_soc_projector_gpu.py"),
                        ("nonlocal_force_oracle_sha256", "scripts/check_nonlocal_force.py"),
                        ("nonlocal_force_check_sha256", "scripts/check_nonlocal_force_gpu.py"),
                        ("nonlocal_compact_oracle_sha256", "scripts/check_nonlocal_compact.py"),
                        ("nonlocal_compact_check_sha256", "scripts/check_nonlocal_compact_gpu.py"),
                        ("nonlocal_owner_check_sha256", "scripts/check_nonlocal_owner.py"),
                        ("pair_geometry_check_sha256", "scripts/check_pair_geometry.py"),
                        ("two_center_oracle_sha256", "scripts/check_two_center.py"),
                        ("two_center_check_sha256", "scripts/check_two_center_gpu.py"),
                        ("grid_owner_check_sha256", "scripts/check_grid_owner.py"),
                        ("atomic_owner_check_sha256", "scripts/check_atomic_owner.py"),
                        ("atomic_masked_force_check_sha256", "scripts/check_atomic_masked_force.py"),
                        ("orbital_owner_check_sha256", "scripts/check_orbital_owner.py"),
                        ("orbital_comparison_check_sha256", "scripts/check_orbital_comparison.py"),
                        ("orbital_basis_check_sha256", "scripts/check_orbital_basis_gpu.py"),
                        ("final_force_check_sha256", "scripts/check_final_force.py"),
                        ("integral_publication_check_sha256", "scripts/check_integral_publication.py"),
                        ("pulay_check_sha256", "scripts/check_mixing.py"),
                        ("scf_scalars_oracle_sha256", "scripts/check_scf_scalars.py"),
                        ("scf_scalars_check_sha256", "scripts/check_scf_scalars_gpu.py"),
                        ("hamiltonian_seeds_oracle_sha256", "scripts/check_hamiltonian_seeds.py"),
                        ("hamiltonian_seeds_check_sha256", "scripts/check_hamiltonian_seeds_gpu.py"),
                        ("sparse_state_check_sha256", "scripts/check_sparse_state.py"),
                        ("scf_convergence_oracle_sha256", "scripts/check_scf_convergence.py"),
                        ("scf_convergence_check_sha256", "scripts/check_scf_convergence_gpu.py"),
                        ("scf_cycle_check_sha256", "scripts/check_scf_cycle.py"),
                        ("scf_control_check_sha256", "scripts/check_scf_control_gpu.py"),
                        ("small_eigensolver_check_sha256", "scripts/check_small_eigensolver.py"),
                        ("block_jacobi_check_sha256", "scripts/check_block_jacobi_gpu.py"),
                        ("block_failure_replay_sha256", "scripts/replay_block_failure.py"),
                        ("cupti_observer_build_sha256", "scripts/build_cupti_observer.py"),
                        ("runtime_driver_sha256", "runtime/modal_app.py"),
                        ("experiment_driver_sha256", "modal_app.py")):
        if field in metadata:
            expected[path] = metadata[field]
    expected.update({f"benchmarks/solver-regressions/{name}":digest for name,digest in metadata.get("solver_regressions_sha256",{}).items()})
    checks = {}
    with tarfile.open(run / "implementation.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate source archive members")
        for path, digest in expected.items():
            try:
                member = archive.getmember(path)
                if not member.isfile():
                    raise ValueError(f"Expected regular source file: {path}")
                with archive.extractfile(member) as source:
                    actual = hashlib.file_digest(source, "sha256").hexdigest()
            except KeyError:
                actual = None
            checks[path] = dict(expected_sha256=digest, actual_sha256=actual, passed=actual == digest)
    return dict(passed=bool(checks) and all(item["passed"] for item in checks.values()),
                recorded_hash_count=len(checks), checks=checks,
                notes="Verifies recorded source hashes only; does not independently reproduce the executable build.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run / "source-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Source archive: {result['recorded_hash_count']} hashes, passed={result['passed']}")
    if not result["passed"]:
        raise SystemExit(1)
