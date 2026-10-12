"""One GPU container runs reference and candidate sequentially on identical hardware.

Launch through scripts/run_modal.py, which prevents overlapping local launches.
No map/spawn, retries, additional GPU functions, or persistent idle GPU containers.
"""
from pathlib import Path
import os
import modal

ROOT = Path(__file__).resolve().parent
if sum(os.environ.get(key)=="1" for key in ("REMAG_H0_COMPONENT_ONLY","REMAG_HISTORY_COMPONENT_ONLY","REMAG_MEMORY_COMPONENT_ONLY"))>1:
    raise ValueError("Choose one component image; concurrent probe modes are unsupported")
app = modal.App("siesta-remag-single-gpu")
volume = modal.Volume.from_name("siesta-remag-results", create_if_missing=True)
FLAGS = " ".join(f"-DSIESTA_WITH_{feature}=OFF" for feature in
                 ("MPI", "ELSI", "DFTD3", "FLOOK", "OPENMP", "FFTW", "HOST_OPTIMIZATION"))
FLAGS += " -DSIESTA_WITH_NETCDF=ON -DSIESTA_WITH_NCDF=ON"
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("gfortran", "g++", "cmake", "make", "pkg-config", "libopenblas-dev", "libnetcdff-dev", "libnetcdf-dev", "git")
    .pip_install("numpy==2.5.3", "scipy==1.18.1", "psutil==7.2.2", "netCDF4==1.7.2")
    .add_local_dir(ROOT / "upstream/siesta", "/opt/siesta-src", copy=True,
                   ignore=["**/.git", "**/.git/**"])
    .run_commands(f"cmake -S /opt/siesta-src -B /opt/reference -DCMAKE_BUILD_TYPE=Release {FLAGS}",
                  "cmake --build /opt/reference --target SIESTA.siesta --parallel 8")
    .add_local_dir(ROOT / "native", "/opt/remag/native", copy=True)
    .add_local_file(ROOT / "scripts/patch_siesta.py", "/opt/remag/patch_siesta.py", copy=True)
    .run_commands("cp -a /opt/siesta-src /opt/candidate-src",
                  "python /opt/remag/patch_siesta.py /opt/candidate-src",
                  f"cmake -S /opt/candidate-src -B /opt/candidate -DCMAKE_BUILD_TYPE=Release {FLAGS} -DREMAG_NATIVE_DIR=/opt/remag/native",
                  "cmake --build /opt/candidate --target SIESTA.siesta remag_graph_probe remag_activity remag_activity_check remag_fermi_check remag_hartree_check remag_pbe_check remag_grid_potential_check remag_pulay_check remag_scf_scalars_check remag_hamiltonian_seeds_check remag_sparse_state_check remag_scf_convergence_check remag_small_eigensolver_check remag_block_jacobi_check remag_scf_control_check remag_hamiltonian_graph_check remag_density_graph_check remag_grid_graph_check remag_scf_cycle_check remag_grid_force_check remag_atomic_grid_force_check remag_final_force_check remag_integral_publication_check remag_orbital_basis_check remag_orbital_owner_check remag_atomic_basis_check remag_atomic_owner_check remag_grid_owner_check remag_two_center_check remag_two_center_owner_check remag_pair_force_check remag_pair_handoff_check remag_soc_projector_check remag_nonlocal_force_check remag_nonlocal_compact_check remag_nonlocal_owner_check remag_pair_geometry_check remag_population_check remag_population_owner_check remag_initial_state_check --parallel 8",
                  "nvcc -std=c++17 -O3 -DNDEBUG --fmad=false -arch=sm_80 -ptx /opt/remag/native/pulay.cu -o /opt/candidate/pulay-diagnostic.ptx")
    .pip_install("nvidia-cuda-cupti-cu12==12.8.90")
    .add_local_file(ROOT / "scripts/build_cupti_observer.py", "/opt/remag/build_cupti_observer.py", copy=True)
    .run_commands("python /opt/remag/build_cupti_observer.py")
    .apt_install("strace")
    .env({"PYTHONPATH": "/opt/remag", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src/siesta_remag", "/opt/remag/siesta_remag",
                   ignore=["__pycache__/**", "**/__pycache__/**", "**/*.pyc"])
    .add_local_file(ROOT / "upstream.lock.json", "/opt/remag/upstream.lock.json")
    .add_local_file(ROOT / "scripts/check_gpu_solver.py", "/opt/remag/check_gpu_solver.py")
    .add_local_file(ROOT / "scripts/check_fermi.py", "/opt/remag/check_fermi.py")
    .add_local_file(ROOT / "scripts/check_hartree.py", "/opt/remag/check_hartree.py")
    .add_local_file(ROOT / "scripts/check_pbe.py", "/opt/remag/check_pbe.py")
    .add_local_file(ROOT / "scripts/check_pbe_gpu.py", "/opt/remag/check_pbe_gpu.py")
    .add_local_file(ROOT / "scripts/check_grid_potential.py", "/opt/remag/check_grid_potential.py")
    .add_local_file(ROOT / "scripts/check_grid_force_formula.py", "/opt/remag/check_grid_force_formula.py")
    .add_local_file(ROOT / "scripts/check_grid_force_gpu.py", "/opt/remag/check_grid_force_gpu.py")
    .add_local_file(ROOT / "scripts/check_atomic_grid_force_formula.py", "/opt/remag/check_atomic_grid_force_formula.py")
    .add_local_file(ROOT / "scripts/check_atomic_grid_force_gpu.py", "/opt/remag/check_atomic_grid_force_gpu.py")
    .add_local_file(ROOT / "scripts/check_atomic_comparison.py", "/opt/remag/check_atomic_comparison.py")
    .add_local_file(ROOT / "scripts/check_atomic_masked_force.py", "/opt/remag/check_atomic_masked_force.py")
    .add_local_file(ROOT / "scripts/check_two_center_comparison.py", "/opt/remag/check_two_center_comparison.py")
    .add_local_file(ROOT / "scripts/check_two_center_owner.py", "/opt/remag/check_two_center_owner.py")
    .add_local_file(ROOT / "scripts/check_pair_handoff.py", "/opt/remag/check_pair_handoff.py")
    .add_local_file(ROOT / "scripts/check_pair_force.py", "/opt/remag/check_pair_force.py")
    .add_local_file(ROOT / "scripts/check_pair_force_gpu.py", "/opt/remag/check_pair_force_gpu.py")
    .add_local_file(ROOT / "scripts/check_population.py", "/opt/remag/check_population.py")
    .add_local_file(ROOT / "scripts/check_initial_state.py", "/opt/remag/check_initial_state.py")
    .add_local_file(ROOT / "scripts/check_population_gpu.py", "/opt/remag/check_population_gpu.py")
    .add_local_file(ROOT / "scripts/check_population_owner.py", "/opt/remag/check_population_owner.py")
    .add_local_file(ROOT / "scripts/check_population_observables.py", "/opt/remag/check_population_observables.py")
    .add_local_file(ROOT / "scripts/check_soc_projector.py", "/opt/remag/check_soc_projector.py")
    .add_local_file(ROOT / "scripts/check_soc_projector_gpu.py", "/opt/remag/check_soc_projector_gpu.py")
    .add_local_file(ROOT / "scripts/check_nonlocal_force.py", "/opt/remag/check_nonlocal_force.py")
    .add_local_file(ROOT / "scripts/check_nonlocal_force_gpu.py", "/opt/remag/check_nonlocal_force_gpu.py")
    .add_local_file(ROOT / "scripts/check_nonlocal_compact.py", "/opt/remag/check_nonlocal_compact.py")
    .add_local_file(ROOT / "scripts/check_nonlocal_compact_gpu.py", "/opt/remag/check_nonlocal_compact_gpu.py")
    .add_local_file(ROOT / "scripts/check_nonlocal_owner.py", "/opt/remag/check_nonlocal_owner.py")
    .add_local_file(ROOT / "scripts/check_pair_geometry.py", "/opt/remag/check_pair_geometry.py")
    .add_local_file(ROOT / "scripts/check_two_center.py", "/opt/remag/check_two_center.py")
    .add_local_file(ROOT / "scripts/check_two_center_gpu.py", "/opt/remag/check_two_center_gpu.py")
    .add_local_file(ROOT / "scripts/check_grid_owner.py", "/opt/remag/check_grid_owner.py")
    .add_local_file(ROOT / "scripts/check_atomic_owner.py", "/opt/remag/check_atomic_owner.py")
    .add_local_file(ROOT / "scripts/check_atomic_basis.py", "/opt/remag/check_atomic_basis.py")
    .add_local_file(ROOT / "scripts/check_atomic_basis_gpu.py", "/opt/remag/check_atomic_basis_gpu.py")
    .add_local_file(ROOT / "scripts/check_orbital_owner.py", "/opt/remag/check_orbital_owner.py")
    .add_local_file(ROOT / "scripts/check_orbital_comparison.py", "/opt/remag/check_orbital_comparison.py")
    .add_local_file(ROOT / "scripts/check_orbital_basis.py", "/opt/remag/check_orbital_basis.py")
    .add_local_file(ROOT / "scripts/check_orbital_basis_gpu.py", "/opt/remag/check_orbital_basis_gpu.py")
    .add_local_file(ROOT / "scripts/check_final_force.py", "/opt/remag/check_final_force.py")
    .add_local_file(ROOT / "scripts/check_integral_publication.py", "/opt/remag/check_integral_publication.py")
    .add_local_file(ROOT / "scripts/check_mixing.py", "/opt/remag/check_mixing.py")
    .add_local_file(ROOT / "scripts/check_scf_scalars.py", "/opt/remag/check_scf_scalars.py")
    .add_local_file(ROOT / "scripts/check_scf_scalars_gpu.py", "/opt/remag/check_scf_scalars_gpu.py")
    .add_local_file(ROOT / "scripts/check_hamiltonian_seeds.py", "/opt/remag/check_hamiltonian_seeds.py")
    .add_local_file(ROOT / "scripts/check_hamiltonian_seeds_gpu.py", "/opt/remag/check_hamiltonian_seeds_gpu.py")
    .add_local_file(ROOT / "scripts/check_sparse_state.py", "/opt/remag/check_sparse_state.py")
    .add_local_file(ROOT / "scripts/check_scf_convergence.py", "/opt/remag/check_scf_convergence.py")
    .add_local_file(ROOT / "scripts/check_scf_convergence_gpu.py", "/opt/remag/check_scf_convergence_gpu.py")
    .add_local_file(ROOT / "scripts/check_scf_control_gpu.py", "/opt/remag/check_scf_control_gpu.py")
    .add_local_file(ROOT / "scripts/check_scf_cycle.py", "/opt/remag/check_scf_cycle.py")
    .add_local_file(ROOT / "scripts/check_initial_density.py", "/opt/remag/check_initial_density.py")
    .add_local_file(ROOT / "scripts/check_small_eigensolver.py", "/opt/remag/check_small_eigensolver.py")
    .add_local_file(ROOT / "scripts/check_block_jacobi_gpu.py", "/opt/remag/check_block_jacobi_gpu.py")
    .add_local_file(ROOT / "scripts/replay_block_failure.py", "/opt/remag/replay_block_failure.py")
    .add_local_dir(ROOT / "benchmarks/solver-regressions", "/opt/remag/solver-regressions")
    .add_local_file(ROOT / "modal_app.py", "/opt/remag/modal_app.py")
)


# The same serialized GPU function can run a bounded history component probe
# without rebuilding/running the unrelated material corpus. No second function
# or allocation is created; main dispatches exactly one invocation.
density_history_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("g++", "cmake", "make")
    .pip_install("numpy==2.5.3", "scipy==1.18.1")
    .add_local_dir(ROOT / "native", "/opt/remag/native", copy=True)
    .run_commands("cmake -S /opt/remag/native -B /opt/history -DCMAKE_BUILD_TYPE=Release",
                  "cmake --build /opt/history --target remag_density_history_check remag_initial_density_check remag_scf_cycle_check --parallel 8")
    .env({"PYTHONPATH":"/opt/remag", "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"})
    .add_local_dir(ROOT / "scripts", "/opt/remag/scripts", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file(ROOT / "results/preflight-initial-density-v86/reference.json", "/opt/reference/initial-density/reference.json")
    .add_local_file(ROOT / "results/preflight-initial-density-v86/reference.npz", "/opt/reference/initial-density/reference.npz")
    .add_local_dir(ROOT / "results/preflight-initial-cycle-v86", "/opt/reference/initial-cycle")
    .add_local_file(ROOT / "results/preflight-density-history-v84/coefficients.npz", "/opt/reference/coefficients.npz")
    .add_local_file(ROOT / "results/preflight-density-history-v84/coefficient-modes.npz", "/opt/reference/coefficient-modes.npz")
    .add_local_file(ROOT / "results/preflight-density-history-v84/coefficient-modes.json", "/opt/reference/coefficient-modes.json")
    .add_local_file(ROOT / "upstream.lock.json", "/opt/remag/upstream.lock.json")
    .add_local_file(ROOT / "modal_app.py", "/opt/remag/modal_app.py")
    .add_local_dir(ROOT / "src/siesta_remag", "/opt/remag/siesta_remag", ignore=["**/__pycache__/**", "**/*.pyc"])
)

# This component uses the SAME serialized benchmark function below. Its
# source is isolated from the production bridge until material integration.
h0_component_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("g++", "cmake", "make")
    .pip_install("numpy==2.5.3")
    .add_local_dir(ROOT / "build/nonlocal-h0-v93/native", "/opt/h0/native", copy=True)
    .run_commands("cmake -S /opt/h0/native -B /opt/h0-build -DCMAKE_BUILD_TYPE=Release",
                  "cmake --build /opt/h0-build --target remag_nonlocal_h0_check --parallel 8")
    .env({"PYTHONPATH":"/opt/remag", "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"})
    .add_local_dir(ROOT / "build/nonlocal-h0-v93/scripts", "/opt/h0/scripts", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file(ROOT / "results/preflight-nonlocal-h0-v93/pristine-h0.json", "/opt/h0/reference/pristine-h0.json")
    .add_local_file(ROOT / "results/preflight-nonlocal-h0-v93/pristine-h0.npz", "/opt/h0/reference/pristine-h0.npz")
    .add_local_file(ROOT / "results/preflight-nonlocal-h0-v93/pristine-h0.F", "/opt/h0/reference/pristine-h0.F")
    .add_local_file(ROOT / "upstream.lock.json", "/opt/remag/upstream.lock.json")
    .add_local_file(ROOT / "modal_app.py", "/opt/remag/modal_app.py")
    .add_local_dir(ROOT / "src/siesta_remag", "/opt/remag/siesta_remag", ignore=["**/__pycache__/**", "**/*.pyc"])
)

def run_nonlocal_h0_probe(run_id):
    import hashlib, json, shutil, subprocess, tarfile
    from siesta_remag.attempts import prepare_attempt,complete_attempt,package_evidence,invocation_digest
    def sha(path):
        with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
    if not run_id.isalnum():raise ValueError("Invalid run identifier")
    visible=subprocess.check_output(["nvidia-smi","--query-gpu=index","--format=csv,noheader"],text=True).splitlines()
    if len(visible)!=1:raise RuntimeError("Exactly one visible GPU is required")
    sources=[p for folder in (Path("/opt/h0/native"),Path("/opt/h0/scripts"),Path("/opt/h0/reference"),Path("/opt/remag/siesta_remag"))
             for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    sources += [Path("/opt/remag/modal_app.py"),Path("/opt/remag/upstream.lock.json")]
    hashes={str(p):sha(p) for p in sources}
    binary=Path("/opt/h0-build/libremag_nonlocal_h0_check.so");binaries={str(binary):sha(binary)}
    root=Path("/results")/run_id
    attempt,completed=prepare_attempt(root,invocation_digest(dict(probe="nonlocal_h0"),hashes,binaries))
    if completed:return package_evidence(root)
    metadata=dict(scope="Isolated nonlocal H0 CUDA component; production integration and material speedup remain unqualified",
                  gpu=subprocess.check_output(["nvidia-smi","-q"],text=True),gpu_count=1,attempt=attempt,
                  source_sha256=hashes,binaries_sha256=binaries)
    (root/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    with tarfile.open(root/"implementation.tar.gz","w:gz") as archive:
        for path in sources:archive.add(path,arcname=str(path).removeprefix("/opt/"))
    for name in ("pristine-h0.json","pristine-h0.npz","pristine-h0.F"):
        shutil.copyfile(Path("/opt/h0/reference")/name,root/name)
    command=["python","/opt/h0/scripts/check_nonlocal_h0_gpu.py","--reference",str(root/"pristine-h0.json"),
             "--library",str(binary),"--output",str(root/"nonlocal-h0.json")]
    try:
        with (root/"nonlocal-h0.log").open("w") as out:
            code=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,timeout=1200).returncode
    except subprocess.TimeoutExpired:code=124
    report=root/"nonlocal-h0.json"
    passed=code==0 and report.is_file() and json.loads(report.read_text()).get("passed") is True
    (root/"standalone-component-status.json").write_text(json.dumps(dict(nonlocal_h0=dict(passed=passed,exit_code=code)),indent=2)+"\n")
    (root/"ledger.json").write_text("[]\n")
    complete_attempt(root);volume.commit();return package_evidence(root)

def run_density_history_probe(run_id,initialization=False):
    import json, subprocess, tarfile
    from siesta_remag.attempts import prepare_attempt,complete_attempt,package_evidence,invocation_digest
    from siesta_remag.corpus import sha256
    if not run_id.isalnum():raise ValueError("Invalid run identifier")
    visible=subprocess.check_output(["nvidia-smi","--query-gpu=index","--format=csv,noheader"],text=True).splitlines()
    if len(visible)!=1:raise RuntimeError("Exactly one visible GPU is required")
    sources=[*Path("/opt/remag/native").glob("*"),Path("/opt/remag/scripts/check_density_history.py"),
             Path("/opt/remag/modal_app.py"),Path("/opt/remag/upstream.lock.json"),*Path("/opt/reference").rglob("*"),*Path("/opt/remag/scripts").glob("*.py")]
    binary=Path("/opt/history/libremag_gpu.so") if initialization else Path("/opt/history/libremag_density_history_check.so")
    binaries={str(p):sha256(p) for p in Path("/opt/history").glob("*.so")}
    hashes={str(p):sha256(p) for p in sources if p.is_file()}
    root=Path("/results")/run_id
    attempt,completed=prepare_attempt(root,invocation_digest(dict(probe="initialization" if initialization else "density_history"),hashes,binaries))
    if completed:return package_evidence(root)
    metadata=dict(scope="Isolated GPU initialization/history qualification; production SIESTA integration remains pending",
                  gpu=subprocess.check_output(["nvidia-smi","-q"],text=True),attempt=attempt,
                  source_sha256=hashes,binary_sha256=sha256(binary),binaries_sha256=binaries,gpu_count=1)
    (root/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    with tarfile.open(root/"implementation.tar.gz","w:gz") as archive:
        for path in sources:
            if path.is_file():archive.add(path,arcname=str(path).removeprefix("/opt/"))
    if initialization:
        commands=[("initial-density",["python","/opt/remag/scripts/check_initial_density_gpu.py","--reference","/opt/reference/initial-density/reference.json",
                  "--library","/opt/history/libremag_initial_density_check.so","--output",str(root/"initial-density.json")]),
                  ("initial-cycle",["python","/opt/remag/scripts/check_scf_cycle.py","--fixtures-from","/opt/reference/initial-cycle","--initialization",
                  "--library","/opt/history/libremag_scf_cycle_check.so","--output",str(root/"initial-cycle/reference.json")])]
    else:
        commands=[("density-history",["python","/opt/remag/scripts/check_density_history.py","--reference","/opt/reference",
                  "--library",str(binary),"--output",str(root/"density-history.json")])]
    status={}
    for label,command in commands:
        try:
            with (root/(label+".log")).open("w") as out:
                code=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,timeout=600).returncode
        except subprocess.TimeoutExpired:code=124
        report=Path(command[-1]);passed=code==0 and report.exists() and json.loads(report.read_text())["passed"]
        status[label]=dict(passed=passed,exit_code=code)
        (root/"standalone-component-status.json").write_text(json.dumps(status,indent=2)+"\n")
        volume.commit()
    (root/"ledger.json").write_text("[]\n")
    complete_attempt(root);volume.commit();return package_evidence(root)

# Isolated calibration still uses the sole serialized benchmark function.
# Neither these sources nor the CUPTI observer replace production math code.
memory_component_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("g++", "cmake", "make", "strace")
    .add_local_dir(ROOT / "build/memory-observer-v99c/native", "/opt/memory/native", copy=True)
    .run_commands("cmake -S /opt/memory/native -B /opt/memory-build -DCMAKE_BUILD_TYPE=Release",
                  "cmake --build /opt/memory-build --parallel 8")
    .pip_install("nvidia-cuda-cupti-cu12==12.8.90")
    .add_local_file(ROOT / "build/memory-observer-v99c/build_cupti128.py", "/opt/memory/build_cupti128.py", copy=True)
    .run_commands("python /opt/memory/build_cupti128.py")
    .env({"PYTHONPATH":"/opt/remag", "PYTHONDONTWRITEBYTECODE":"1"})
    .add_local_file(ROOT / "build/memory-observer-v99c/run_probe.py", "/opt/memory/run_probe.py")
    .add_local_file(ROOT / "build/memory-observer-v99c/verify_positive_control.py", "/opt/memory/verify_positive_control.py")
    .add_local_file(ROOT / "build/memory-observer-v99c/check_linux_io.py", "/opt/memory/check_linux_io.py")
    .add_local_file(ROOT / "upstream.lock.json", "/opt/remag/upstream.lock.json")
    .add_local_file(ROOT / "modal_app.py", "/opt/remag/modal_app.py")
    .add_local_dir(ROOT / "src/siesta_remag", "/opt/remag/siesta_remag", ignore=["**/__pycache__/**", "**/*.pyc"])
)

def run_memory_activity_probe(run_id):
    import sys
    sys.path.insert(0,"/opt/memory")
    from run_probe import run
    return run(run_id,volume)

@app.function(image=memory_component_image if os.environ.get("REMAG_MEMORY_COMPONENT_ONLY")=="1" else h0_component_image if os.environ.get("REMAG_H0_COMPONENT_ONLY")=="1" else density_history_image if os.environ.get("REMAG_HISTORY_COMPONENT_ONLY")=="1" else image, gpu="A100-40GB:1", cpu=8, memory=32768,
              max_containers=1, min_containers=0, buffer_containers=0,
              scaledown_window=2, retries=0, timeout=7200,
              volumes={"/results": volume})
def benchmark(cases: list[str], run_id: str, grid: bool = False, dm: bool = False,
              ablation: bool = False, candidate_repeats: int = 1, io_policy: str = "end-of-cycle",
              epoch_ablation: bool = False, verify_epoch: bool = False, normalize: bool = False,
              normalization_ablation: bool = False, probe_control: bool = False, trace_io: bool = False,
              defer_scf_log: bool = False, log_ablation: bool = False, trace_cuda: bool = False,
              occupations: bool = False, occupation_ablation: bool = False,
              resident_spectra: bool = False, spectral_ablation: bool = False,
              matrix_assembly: bool = False, matrix_ablation: bool = False,
              hartree: bool = False, hartree_ablation: bool = False,
              resident_grid: bool = False, grid_flow_ablation: bool = False,
              resident_scf: bool = False, sparse_state_ablation: bool = False, probe_small_solver: bool = False, probe_block_solver: bool = False,
              block_solver: bool = False, block_solver_ablation: bool = False, probe_scf_control: bool = False, probe_scf_cycle: bool = False,
              gpu_model: str = "A100-40GB", device_scf: bool = False, device_cycle_ablation: bool = False, reference_run: str = "", replay_block_failure: str = "", block_width: int = 64, block_local_solver: str = "syev", probe_grid_force: bool = False, force_shadow: bool = False, force_shadow_resident: bool = False, grid_force: bool = False, grid_force_ablation: bool = False, probe_atomic_grid_force: bool = False, atomic_force_shadow: bool = False, atomic_grid_force: bool = False, atomic_force_ablation: bool = False, final_grid_shadow: bool = False, final_grid: bool = False, final_grid_ablation: bool = False, final_density_shadow: bool = False, final_density: bool = False, final_density_ablation: bool = False, probe_orbital_basis: bool = False, orbital_basis_shadow: bool = False, orbital_basis: bool = False, orbital_basis_ablation: bool = False, probe_atomic_basis: bool = False, atomic_basis_shadow: bool = False, atomic_basis: bool = False, atomic_basis_ablation: bool = False, probe_two_center: bool = False, two_center_shadow: bool = False, probe_pair_force: bool = False, pair_force_shadow: bool = False, probe_soc_projector: bool = False, probe_nonlocal_force: bool = False, nonlocal_force_shadow: bool = False, integral_force: bool = False, probe_population: bool = False, vector_cache: bool = False, vector_cache_ablation: bool = False, population: bool = False, probe_density_history_only: bool = False, probe_initialization_only: bool = False, gpu_initialization: bool = False, initial_populations: bool = False, gpu_history: bool = False, final_spectra: bool = False, gate_h_export: bool = False, probe_nonlocal_h0_only: bool = False, nonlocal_h0_shadow: bool = False, nonlocal_h0: bool = False, poison_nonlocal_h0: bool = False, probe_memory_observer_only: bool = False, solver_lanes: int = 1, solver_warm_start: bool = False, gpu_radial_tables: bool = False, gpu_initial_mesh: bool = False, shared_local_solver: bool = False, grid_potential_spins: bool = False, fused_matrix_rotation: bool = False, fp64_vector_rotation: bool = False, fp64_matrix_rotation: bool = False):

    if sum((probe_density_history_only,probe_initialization_only,probe_nonlocal_h0_only,probe_memory_observer_only))>1:raise ValueError("Choose one component probe")
    if poison_nonlocal_h0 and not nonlocal_h0:raise ValueError("Host H0 poisoning requires production GPU H0")
    if nonlocal_h0 and (nonlocal_h0_shadow or not (integral_force and device_scf and gpu_initialization) or
                           any((probe_density_history_only,probe_initialization_only,probe_nonlocal_h0_only,probe_memory_observer_only))):
        raise ValueError("Resident H0 requires a material run with GPU initialization, device SCF and integral forces")
    if nonlocal_h0_shadow and (not integral_force or any((probe_density_history_only,probe_initialization_only,probe_nonlocal_h0_only,probe_memory_observer_only))):
        raise ValueError("H0 shadow requires a material run with authoritative integral forces")
    if probe_memory_observer_only:return run_memory_activity_probe(run_id)
    if probe_nonlocal_h0_only:return run_nonlocal_h0_probe(run_id)
    if probe_density_history_only or probe_initialization_only:return run_density_history_probe(run_id,probe_initialization_only)
    invocation_parameters=locals().copy()
    if shared_local_solver and (block_width!=32 or block_local_solver!="jacobi"):
        raise ValueError("Shared local solver requires 32-state Jacobi blocks")
    if grid_potential_spins and not grid:raise ValueError("Shared-spin grid potential requires the GPU grid path")
    if gpu_initialization and not device_scf:raise ValueError("GPU initialization requires device SCF")
    if gpu_initial_mesh and (not (device_scf and gpu_initialization and orbital_basis and final_grid and integral_force)
                            or device_cycle_ablation or orbital_basis_ablation or final_grid_ablation):
        raise ValueError("GPU initial mesh requires production device SCF, initialization, orbital basis, final grid and integral forces")
    if initial_populations and not (gpu_initialization and population):raise ValueError("Initial population validation requires GPU initialization and population owners")
    if gate_h_export and not (device_scf and final_grid and integral_force):
        raise ValueError("Gated Hamiltonian exports require device SCF, final grid and integral forces")
    if final_spectra and not (device_scf and final_grid and integral_force):
        raise ValueError("Final spectrum ownership requires device SCF, final grid and integral forces")
    if gpu_history and not (gpu_initialization and population and integral_force and initial_populations):
        raise ValueError("GPU history qualification requires initialization, initial populations and integral forces")
    import io
    import json
    import os
    import subprocess
    import tarfile
    from siesta_remag.corpus import CASES, IO_POLICIES, prepare
    from siesta_remag.corpus import sha256
    from siesta_remag.runner import run_case
    from siesta_remag.compare import compare, compare_mae

    if not cases or any(c not in CASES for c in cases) or len(cases) != len(set(cases)):
        raise ValueError("Choose a nonempty, unique list of registered benchmark cases")
    if block_local_solver not in ("syev","jacobi") or (block_local_solver=="jacobi" and block_width!=32):
        raise ValueError("Batched Jacobi requires --block-width 32; other local solver is syev")
    if fused_matrix_rotation and (block_width!=32 or not (device_scf and vector_cache)):
        raise ValueError("Fused matrix rotation requires 32-state blocks, device SCF and vector cache")
    if fp64_vector_rotation and (block_width!=32 or not (device_scf and vector_cache)):
        raise ValueError("FP64 vector rotation requires 32-state blocks, device SCF and vector cache")
    if fp64_matrix_rotation and (block_width!=32 or not (device_scf and vector_cache and fused_matrix_rotation)):
        raise ValueError("FP64 matrix rotation requires 32-state blocks, device SCF, vector cache and fused matrix rotation")
    if block_width not in (32,64):
        raise ValueError("Experimental local eigensolver width must be 32 or 64")
    if solver_lanes not in (1,2,4,8) or (solver_lanes>1 and not (device_scf and vector_cache)):
        raise ValueError("Solver workspaces must be 1, 2, 4 or 8; parallel workspaces require device SCF and vector cache")
    if solver_warm_start and not (device_scf and vector_cache):
        raise ValueError("Eigensolver warm start requires device SCF and vector cache")
    if gpu_radial_tables and not integral_force:
        raise ValueError("GPU radial tables require the production integral-force path")
    if force_shadow_resident and not device_scf:
        raise ValueError("Resident force shadow requires the device cycle")
    if force_shadow_resident and device_cycle_ablation:
        raise ValueError("Device-cycle ablation cannot provide retained final fields; use the host-staged force shadow for that control")
    if (orbital_basis_shadow or orbital_basis) and (not grid_force or grid_force_ablation or (orbital_basis and orbital_basis_shadow)):
        raise ValueError("Choose device or shadow orbital derivatives with GPU forces and without force ablation")
    if (atomic_basis_shadow or atomic_basis) and (not atomic_grid_force or not grid_force or atomic_force_ablation or grid_force_ablation or (atomic_basis and atomic_basis_shadow)):raise ValueError("Atomic basis shadow requires authoritative GPU forces")
    if integral_force and (not (device_scf and final_grid and grid_force and atomic_grid_force and orbital_basis and atomic_basis)
            or any((two_center_shadow,pair_force_shadow,nonlocal_force_shadow,force_shadow,atomic_force_shadow,
                    orbital_basis_shadow,atomic_basis_shadow,final_grid_shadow,device_cycle_ablation,grid_force_ablation,
                    atomic_force_ablation,final_grid_ablation,orbital_basis_ablation,atomic_basis_ablation))):
        raise ValueError("Integral forces require the full authoritative device path without shadow/force ablation")
    if nonlocal_force_shadow and not pair_force_shadow:
        raise ValueError("Nonlocal shadow requires --pair-force-shadow")
    if pair_force_shadow and not (two_center_shadow and grid_force and device_scf):
        raise ValueError("Pair-force material shadow requires integral shadow, GPU grid forces and device SCF")
    if two_center_shadow and (not grid_force or grid_force_ablation or device_cycle_ablation):
        raise ValueError("Two-center material shadow requires GPU grid forces without cycle/force ablation")
    if atomic_basis_ablation and not atomic_basis:raise ValueError("Atomic basis ablation requires device derivatives")
    if orbital_basis_ablation and not orbital_basis:raise ValueError("Orbital basis ablation requires device derivatives")
    force_shadow = force_shadow or force_shadow_resident
    if atomic_grid_force and (not grid_force or atomic_force_shadow or grid_force_ablation):
        raise ValueError("Authoritative atomic forces require --grid-force without shadow or dfscf ablation")
    if (final_grid or final_grid_shadow) and (not atomic_grid_force or atomic_force_ablation or grid_force_ablation):
        raise ValueError('Final force-grid reuse requires shared atomic forces without other force ablations')
    if final_grid and final_grid_shadow:
        raise ValueError('Choose final-grid production or shadow mode')
    if (final_density or final_density_shadow) and (not final_grid or final_grid_ablation):
        raise ValueError('Final density reuse requires final-grid production without its ablation')
    if final_density and final_density_shadow:
        raise ValueError('Choose final-density production or shadow')
    if final_density_ablation and not final_density:
        raise ValueError('Final-density ablation requires production mode')
    if final_grid_ablation and not final_grid:
        raise ValueError('Final-grid ablation requires final-grid production mode')
    if atomic_force_ablation and not atomic_grid_force:
        raise ValueError("Atomic-force ablation requires --atomic-grid-force")
    if atomic_force_shadow and (not grid_force or grid_force_ablation):
        raise ValueError("Atomic-force shadows require --grid-force without its performance ablation")
    if grid_force and (not device_scf or force_shadow or device_cycle_ablation):
        raise ValueError("GPU grid forces require the device cycle without shadow or device-cycle ablation")
    if grid_force_ablation and not grid_force:
        raise ValueError("Grid-force ablation requires --grid-force")
    if force_shadow and not grid:
        raise ValueError("Force shadow requires the GPU grid topology")
    os.environ["REMAG_BLOCK_WIDTH"]=str(block_width)
    os.environ["REMAG_BLOCK_LOCAL_SOLVER"]=block_local_solver
    if gpu_model not in ("A100-40GB","A100-80GB","H100"):
        raise ValueError("Choose A100-40GB, A100-80GB or H100; the GPU count remains one")
    if io_policy not in IO_POLICIES:
        raise ValueError(f"Unknown I/O policy: {io_policy}")
    if candidate_repeats < 1 or candidate_repeats > 5 or (ablation and not dm):
        raise ValueError("Use 1–5 candidate repetitions; density-matrix ablation requires --dm")
    if (epoch_ablation or verify_epoch) and not grid:
        raise ValueError("Epoch controls require --grid")
    if normalize and not dm:
        raise ValueError("Device charge normalization requires --dm")
    if normalization_ablation and not normalize:
        raise ValueError("Normalization ablation requires --normalize")
    if occupations and not dm:
        raise ValueError("Device occupations require --dm")
    if occupation_ablation and not occupations:
        raise ValueError("Occupation ablation requires --occupations")
    if resident_spectra and not occupations:
        raise ValueError("Resident spectra require --occupations")
    if spectral_ablation and not resident_spectra:
        raise ValueError("Spectral ablation requires --resident-spectra")
    if matrix_assembly and not resident_spectra:
        raise ValueError("Device matrix assembly requires --resident-spectra")
    if matrix_ablation and not matrix_assembly:
        raise ValueError("Matrix ablation requires --matrix-assembly")
    if hartree and not grid:
        raise ValueError("Device Hartree requires --grid")
    if hartree_ablation and not hartree:
        raise ValueError("Hartree ablation requires --hartree")
    if resident_grid and not (grid and hartree):
        raise ValueError("Resident grid requires --grid --hartree")
    if grid_flow_ablation and not resident_grid:
        raise ValueError("Grid-flow ablation requires --resident-grid")
    if resident_grid and hartree_ablation:
        raise ValueError("Use --grid-flow-ablation for the combined resident Hartree/PBE path")
    if resident_scf and not (resident_grid and matrix_assembly and resident_spectra and normalize and dm and defer_scf_log):
        raise ValueError("Resident SCF requires resident grid, matrix assembly, spectra, normalization, DM and deferred logs")
    if resident_scf and any((ablation,epoch_ablation,normalization_ablation,occupation_ablation,spectral_ablation,
                             matrix_ablation,hartree_ablation,grid_flow_ablation,log_ablation)):
        raise ValueError("Use --sparse-state-ablation for the resident sparse-state comparison")
    if sparse_state_ablation and not resident_scf:
        raise ValueError("Sparse-state ablation requires --resident-scf")
    if block_solver and (not resident_scf or sparse_state_ablation):
        raise ValueError("Experimental block solver requires --resident-scf; use --block-solver-ablation for its control")
    if device_scf and (not resident_scf or sparse_state_ablation or block_solver or block_solver_ablation):
        raise ValueError("Device SCF requires --resident-scf without legacy solver/owner ablations")
    if (vector_cache and not device_scf) or (vector_cache_ablation and not vector_cache):
        raise ValueError("Vector cache requires device SCF; its ablation requires the cache")
    if population and not integral_force:
        raise ValueError("Population requires the authoritative integral-force route")
    if device_cycle_ablation and not device_scf:
        raise ValueError("Device-cycle ablation requires --device-scf")
    if device_scf and trace_cuda and not probe_scf_cycle:
        raise ValueError("Device-cycle CUDA tracing requires --probe-scf-cycle for conditional observer calibration")
    if block_solver_ablation and not block_solver:
        raise ValueError("Block-solver ablation requires --block-solver")
    if log_ablation and not defer_scf_log:
        raise ValueError("Log ablation requires --defer-scf-log")
    if reference_run and not reference_run.isalnum():
        raise ValueError("Reference reuse requires a saved run identifier")
    replay_sources=replay_block_failure.split(",") if replay_block_failure else []
    if len(replay_sources)>4 or len(set(replay_sources))!=len(replay_sources):
        raise ValueError("Choose up to four unique retained failures to replay")
    for source in replay_sources:
        parts=source.split("/")
        if len(parts)!=3 or not parts[0].isalnum() or parts[1] not in CASES or not parts[2].replace("-","").isalnum():
            raise ValueError("Failure replay requires RUN/CASE/ROLE within the results Volume")
    if not run_id.replace("-", "").isalnum():
        raise ValueError("Invalid run identifier")
    if len(subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True).splitlines()) != 1:
        raise RuntimeError("Exactly one visible GPU is required")
    from siesta_remag.attempts import prepare_attempt,complete_attempt,package_evidence,invocation_digest
    identity_files=[Path(__file__),Path("/opt/remag/modal_app.py"),Path("/opt/remag/patch_siesta.py"),
                    Path("/opt/remag/upstream.lock.json"),*Path("/opt/remag/native").glob("*"),
                    *Path("/opt/remag/siesta_remag").glob("*.py"),*Path("/opt/remag").glob("check_*.py")]
    identity_sources={str(p):sha256(p) for p in identity_files if p.is_file()}
    binary_files=[Path("/opt/reference/Src/siesta"),Path("/opt/candidate/Src/siesta"),
                  *Path("/opt/candidate/remag-native").glob("*.so"),*Path("/opt/candidate/remag-native").glob("*_check")]
    binary_hashes={str(p):sha256(p) for p in binary_files if p.is_file()}
    root = Path("/results") / run_id
    attempt,completed=prepare_attempt(root,invocation_digest(invocation_parameters,identity_sources,binary_hashes))
    if completed:
        print("Repackaging completed attempt "+attempt["attempt_id"],flush=True)
        return package_evidence(root)
    volume.commit()
    print("Scientific attempt "+attempt["attempt_id"]+"; preserved interrupted attempts: "+str(attempt["interrupted_attempts"]),flush=True)
    metadata = {"upstream": json.loads(Path("/opt/remag/upstream.lock.json").read_text()),
                "gpu": subprocess.check_output(["nvidia-smi", "-q"], text=True),
                "cpu": Path("/proc/cpuinfo").read_text(), "cases": cases,
                "policy": {"gpu_count": 1, "gpu_request":gpu_model+":1", "max_containers": 1, "sequential": True,
                           "batch_timeout_seconds":7200, "large_case_timeout_seconds":3600,
                           "other_case_timeout_seconds":900,"native_check_timeout_seconds":180,
                           "fermi_check_timeout_seconds":120,"hartree_check_timeout_seconds":120,
                           "pbe_check_timeout_seconds":180,"grid_potential_check_timeout_seconds":180,
                           "pulay_check_timeout_seconds":120,"scf_scalars_check_timeout_seconds":120,
                           "hamiltonian_seeds_check_timeout_seconds":120,"sparse_state_check_timeout_seconds":180,
                           "scf_convergence_check_timeout_seconds":120,"scf_control_check_timeout_seconds":360,"scf_cycle_check_timeout_seconds":480,
                           "small_eigensolver_check_timeout_seconds":180,"block_jacobi_check_timeout_seconds":900}}
    metadata["attempt"] = attempt
    metadata["native_binary_sha256"] = binary_hashes
    metadata["implementation_sha256"] = {str(p): sha256(p) for p in
        [Path("/opt/remag/patch_siesta.py"), *Path("/opt/remag/native").glob("*")]
        if p.is_file()}
    metadata["grid_enabled"] = grid
    metadata["density_matrix_enabled"] = dm
    metadata["ablation"] = ablation
    metadata["candidate_repeats"] = candidate_repeats
    metadata["io_policy"] = io_policy
    metadata["epoch_ablation"] = epoch_ablation
    metadata["verify_epoch"] = verify_epoch
    metadata["device_charge_normalization"] = normalize
    metadata["normalization_ablation"] = normalization_ablation
    metadata["device_occupations"] = occupations
    metadata["occupation_ablation"] = occupation_ablation
    metadata["resident_spectra"] = resident_spectra
    metadata["spectral_ablation"] = spectral_ablation
    metadata["matrix_assembly"] = matrix_assembly
    metadata["matrix_ablation"] = matrix_ablation
    metadata["device_hartree"] = hartree
    metadata["hartree_ablation"] = hartree_ablation
    metadata["resident_grid"] = resident_grid
    metadata["grid_flow_ablation"] = grid_flow_ablation
    metadata["resident_scf"] = resident_scf
    metadata["reference_run"] = reference_run or None
    metadata["replay_block_failure"] = replay_block_failure or None
    metadata["block_local_width"] = block_width
    metadata["block_local_solver"] = block_local_solver
    metadata["probe_grid_force"] = probe_grid_force
    metadata["probe_atomic_grid_force"] = probe_atomic_grid_force
    metadata["probe_orbital_basis"] = probe_orbital_basis
    metadata["orbital_basis_shadow"] = orbital_basis_shadow
    metadata["orbital_basis"] = orbital_basis
    metadata["orbital_basis_ablation"] = orbital_basis_ablation
    metadata["probe_two_center"] = probe_two_center
    metadata["two_center_shadow"] = two_center_shadow
    metadata["probe_pair_force"] = probe_pair_force
    metadata["probe_population"] = probe_population
    metadata["population"] = population
    metadata["initial_populations"] = initial_populations
    metadata["initial_state_check_sha256"] = sha256(Path("/opt/remag/check_initial_state.py"))
    metadata["solver_lanes"] = solver_lanes
    metadata["solver_warm_start"] = solver_warm_start
    metadata["gpu_radial_tables"] = gpu_radial_tables
    metadata["vector_cache"] = vector_cache
    metadata["vector_cache_ablation"] = vector_cache_ablation
    metadata["pair_force_shadow"] = pair_force_shadow
    metadata["pair_geometry_prescf"] = pair_force_shadow or integral_force
    metadata["integral_force"] = integral_force
    metadata["probe_soc_projector"] = probe_soc_projector
    metadata["probe_nonlocal_force"] = probe_nonlocal_force
    metadata["nonlocal_force_shadow"] = nonlocal_force_shadow
    metadata["probe_atomic_basis"] = probe_atomic_basis
    metadata["atomic_basis_shadow"] = atomic_basis_shadow
    metadata["atomic_basis"] = atomic_basis
    metadata["atomic_basis_ablation"] = atomic_basis_ablation
    metadata["atomic_force_shadow"] = atomic_force_shadow
    metadata["atomic_grid_force"] = atomic_grid_force
    metadata["atomic_force_ablation"] = atomic_force_ablation
    metadata["final_grid_shadow"] = final_grid_shadow
    metadata["final_grid"] = final_grid
    metadata["final_grid_ablation"] = final_grid_ablation
    metadata["final_density_shadow"] = final_density_shadow
    metadata["final_density"] = final_density
    metadata["final_density_ablation"] = final_density_ablation
    metadata["force_shadow"] = force_shadow
    metadata["force_shadow_resident"] = force_shadow_resident
    metadata["grid_force"] = grid_force
    metadata["grid_force_ablation"] = grid_force_ablation
    metadata["device_scf"] = device_scf
    metadata["gpu_initialization"] = gpu_initialization
    metadata["gpu_initial_mesh"] = gpu_initial_mesh
    metadata["shared_local_solver"] = shared_local_solver
    metadata["grid_potential_spins"] = grid_potential_spins
    metadata["fused_matrix_rotation"] = fused_matrix_rotation
    metadata["fp64_vector_rotation"] = fp64_vector_rotation
    metadata["fp64_matrix_rotation"] = fp64_matrix_rotation
    metadata["gpu_history"] = gpu_history
    metadata["final_spectra"] = final_spectra
    metadata["gate_h_export"] = gate_h_export
    metadata["nonlocal_h0_shadow"] = nonlocal_h0_shadow
    metadata["nonlocal_h0"] = nonlocal_h0
    metadata["compact_nonlocal_queries"] = nonlocal_h0
    metadata["poison_nonlocal_h0"] = poison_nonlocal_h0
    metadata["device_cycle_ablation"] = device_cycle_ablation
    metadata["sparse_state_ablation"] = sparse_state_ablation
    metadata["probe_small_solver"] = probe_small_solver
    metadata["probe_block_solver"] = probe_block_solver
    metadata["experimental_conditional_observer_mode"] = "memory_only"
    metadata["probe_scf_control"] = probe_scf_control
    metadata["probe_scf_cycle"] = probe_scf_cycle
    metadata["experimental_block_solver"] = block_solver
    metadata["block_solver_ablation"] = block_solver_ablation
    metadata["probe_control"] = probe_control
    metadata["trace_io"] = trace_io
    metadata["trace_cuda"] = trace_cuda
    metadata["defer_scf_log"] = defer_scf_log
    metadata["log_ablation"] = log_ablation
    metadata["harness_sha256"] = {p.name: sha256(p) for p in Path("/opt/remag/siesta_remag").glob("*.py")}
    metadata["native_check_sha256"] = sha256(Path("/opt/remag/check_gpu_solver.py"))
    metadata["fermi_check_sha256"] = sha256(Path("/opt/remag/check_fermi.py"))
    metadata["hartree_check_sha256"] = sha256(Path("/opt/remag/check_hartree.py"))
    metadata["pbe_oracle_sha256"] = sha256(Path("/opt/remag/check_pbe.py"))
    metadata["pbe_check_sha256"] = sha256(Path("/opt/remag/check_pbe_gpu.py"))
    metadata["grid_force_oracle_sha256"] = sha256(Path("/opt/remag/check_grid_force_formula.py"))
    metadata["grid_force_check_sha256"] = sha256(Path("/opt/remag/check_grid_force_gpu.py"))
    metadata["atomic_grid_force_oracle_sha256"] = sha256(Path("/opt/remag/check_atomic_grid_force_formula.py"))
    metadata["atomic_grid_force_check_sha256"] = sha256(Path("/opt/remag/check_atomic_grid_force_gpu.py"))
    metadata["atomic_comparison_check_sha256"] = sha256(Path("/opt/remag/check_atomic_comparison.py"))
    metadata["atomic_masked_force_check_sha256"] = sha256(Path("/opt/remag/check_atomic_masked_force.py"))
    metadata["two_center_comparison_check_sha256"] = sha256(Path("/opt/remag/check_two_center_comparison.py"))
    metadata["two_center_owner_check_sha256"] = sha256(Path("/opt/remag/check_two_center_owner.py"))
    metadata["pair_handoff_check_sha256"] = sha256(Path("/opt/remag/check_pair_handoff.py"))
    metadata["pair_force_oracle_sha256"] = sha256(Path("/opt/remag/check_pair_force.py"))
    metadata["pair_force_check_sha256"] = sha256(Path("/opt/remag/check_pair_force_gpu.py"))
    metadata["population_oracle_sha256"] = sha256(Path("/opt/remag/check_population.py"))
    metadata["population_check_sha256"] = sha256(Path("/opt/remag/check_population_gpu.py"))
    metadata["population_owner_check_sha256"] = sha256(Path("/opt/remag/check_population_owner.py"))
    metadata["population_observables_check_sha256"] = sha256(Path("/opt/remag/check_population_observables.py"))
    metadata["soc_projector_oracle_sha256"] = sha256(Path("/opt/remag/check_soc_projector.py"))
    metadata["soc_projector_check_sha256"] = sha256(Path("/opt/remag/check_soc_projector_gpu.py"))
    metadata["nonlocal_force_oracle_sha256"] = sha256(Path("/opt/remag/check_nonlocal_force.py"))
    metadata["nonlocal_force_check_sha256"] = sha256(Path("/opt/remag/check_nonlocal_force_gpu.py"))
    metadata["nonlocal_compact_oracle_sha256"] = sha256(Path("/opt/remag/check_nonlocal_compact.py"))
    metadata["nonlocal_compact_check_sha256"] = sha256(Path("/opt/remag/check_nonlocal_compact_gpu.py"))
    metadata["nonlocal_owner_check_sha256"] = sha256(Path("/opt/remag/check_nonlocal_owner.py"))
    metadata["pair_geometry_check_sha256"] = sha256(Path("/opt/remag/check_pair_geometry.py"))
    metadata["two_center_oracle_sha256"] = sha256(Path("/opt/remag/check_two_center.py"))
    metadata["two_center_check_sha256"] = sha256(Path("/opt/remag/check_two_center_gpu.py"))
    metadata["grid_owner_check_sha256"] = sha256(Path("/opt/remag/check_grid_owner.py"))
    metadata["atomic_owner_check_sha256"] = sha256(Path("/opt/remag/check_atomic_owner.py"))
    metadata["atomic_basis_oracle_sha256"] = sha256(Path("/opt/remag/check_atomic_basis.py"))
    metadata["atomic_basis_check_sha256"] = sha256(Path("/opt/remag/check_atomic_basis_gpu.py"))
    metadata["orbital_basis_oracle_sha256"] = sha256(Path("/opt/remag/check_orbital_basis.py"))
    metadata["orbital_owner_check_sha256"] = sha256(Path("/opt/remag/check_orbital_owner.py"))
    metadata["orbital_comparison_check_sha256"] = sha256(Path("/opt/remag/check_orbital_comparison.py"))
    metadata["orbital_basis_check_sha256"] = sha256(Path("/opt/remag/check_orbital_basis_gpu.py"))
    metadata["final_force_check_sha256"] = sha256(Path("/opt/remag/check_final_force.py"))
    metadata["integral_publication_check_sha256"] = sha256(Path("/opt/remag/check_integral_publication.py"))
    metadata["grid_potential_check_sha256"] = sha256(Path("/opt/remag/check_grid_potential.py"))
    metadata["pulay_check_sha256"] = sha256(Path("/opt/remag/check_mixing.py"))
    metadata["scf_scalars_oracle_sha256"] = sha256(Path("/opt/remag/check_scf_scalars.py"))
    metadata["scf_scalars_check_sha256"] = sha256(Path("/opt/remag/check_scf_scalars_gpu.py"))
    metadata["hamiltonian_seeds_oracle_sha256"] = sha256(Path("/opt/remag/check_hamiltonian_seeds.py"))
    metadata["hamiltonian_seeds_check_sha256"] = sha256(Path("/opt/remag/check_hamiltonian_seeds_gpu.py"))
    metadata["sparse_state_check_sha256"] = sha256(Path("/opt/remag/check_sparse_state.py"))
    metadata["scf_convergence_oracle_sha256"] = sha256(Path("/opt/remag/check_scf_convergence.py"))
    metadata["scf_convergence_check_sha256"] = sha256(Path("/opt/remag/check_scf_convergence_gpu.py"))
    metadata["scf_cycle_check_sha256"] = sha256(Path("/opt/remag/check_scf_cycle.py"))
    metadata["scf_control_check_sha256"] = sha256(Path("/opt/remag/check_scf_control_gpu.py"))
    metadata["small_eigensolver_check_sha256"] = sha256(Path("/opt/remag/check_small_eigensolver.py"))
    metadata["solver_regressions_sha256"] = {p.name:sha256(p) for p in Path("/opt/remag/solver-regressions").glob("*") if p.is_file()}
    metadata["block_jacobi_check_sha256"] = sha256(Path("/opt/remag/check_block_jacobi_gpu.py"))
    metadata["block_failure_replay_sha256"] = sha256(Path("/opt/remag/replay_block_failure.py"))
    metadata["cupti_observer_build_sha256"] = sha256(Path("/opt/remag/build_cupti_observer.py"))
    metadata["experimental_cupti_observer"] = json.loads(Path("/opt/remag-cupti128/build.json").read_text())
    metadata["experiment_driver_sha256"] = sha256(Path("/opt/remag/modal_app.py"))
    metadata["runtime_driver_sha256"] = sha256(Path(__file__))
    metadata["pulay_diagnostic_ptx_sha256"] = sha256(Path("/opt/candidate/pulay-diagnostic.ptx"))
    (root/"pulay-diagnostic.ptx").write_bytes(Path("/opt/candidate/pulay-diagnostic.ptx").read_bytes())
    metadata["build_cache"] = {role: Path(f"/opt/{role}/CMakeCache.txt").read_text()
                               for role in ("reference", "candidate")}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    # Preserve the exact implementation for each experiment, including failed ones.
    with tarfile.open(root / "implementation.tar.gz", "w:gz") as archive:
        files = [*Path("/opt/remag/native").glob("*"), *Path("/opt/remag/siesta_remag").glob("*.py"),
                 Path("/opt/remag/patch_siesta.py"), Path("/opt/remag/check_gpu_solver.py"),
                 Path("/opt/remag/check_fermi.py"),
                 Path("/opt/remag/check_hartree.py"),
                 Path("/opt/remag/check_pbe.py"), Path("/opt/remag/check_pbe_gpu.py"),
                 Path("/opt/remag/check_grid_potential.py"), Path("/opt/remag/check_mixing.py"),
                 Path("/opt/remag/check_grid_force_formula.py"),Path("/opt/remag/check_grid_force_gpu.py"),
                 Path("/opt/remag/check_atomic_grid_force_formula.py"),Path("/opt/remag/check_atomic_grid_force_gpu.py"),
                 Path("/opt/remag/check_atomic_comparison.py"),
                 Path("/opt/remag/check_atomic_masked_force.py"),
                 Path("/opt/remag/check_two_center_comparison.py"),
                 Path("/opt/remag/check_two_center_owner.py"),
                 Path("/opt/remag/check_pair_handoff.py"),
                 Path("/opt/remag/check_pair_force.py"),
                 Path("/opt/remag/check_pair_force_gpu.py"),
                 Path("/opt/remag/check_population.py"),
                 Path("/opt/remag/check_initial_state.py"),
                 Path("/opt/remag/check_population_gpu.py"),
                 Path("/opt/remag/check_population_owner.py"),
                 Path("/opt/remag/check_population_observables.py"),
                 Path("/opt/remag/check_soc_projector.py"),
                 Path("/opt/remag/check_soc_projector_gpu.py"),
                 Path("/opt/remag/check_nonlocal_force.py"),
                 Path("/opt/remag/check_nonlocal_force_gpu.py"),
                 Path("/opt/remag/check_nonlocal_compact.py"),
                 Path("/opt/remag/check_nonlocal_compact_gpu.py"),
                 Path("/opt/remag/check_nonlocal_owner.py"),
                 Path("/opt/remag/check_pair_geometry.py"),
                 Path("/opt/remag/check_two_center.py"), Path("/opt/remag/check_two_center_gpu.py"),
                 Path("/opt/remag/check_atomic_owner.py"), Path("/opt/remag/check_grid_owner.py"),
                 Path("/opt/remag/check_atomic_basis.py"),Path("/opt/remag/check_atomic_basis_gpu.py"),
                 Path("/opt/remag/check_orbital_owner.py"),
                 Path("/opt/remag/check_orbital_comparison.py"),
                 Path("/opt/remag/check_orbital_basis.py"),Path("/opt/remag/check_orbital_basis_gpu.py"),
                 Path("/opt/remag/check_final_force.py"), Path("/opt/remag/check_integral_publication.py"),
                 Path("/opt/remag/check_scf_scalars.py"), Path("/opt/remag/check_scf_scalars_gpu.py"),
                 Path("/opt/remag/check_hamiltonian_seeds.py"), Path("/opt/remag/check_hamiltonian_seeds_gpu.py"),
                 Path("/opt/remag/check_sparse_state.py"),
                 Path("/opt/remag/check_scf_convergence.py"), Path("/opt/remag/check_scf_convergence_gpu.py"),
                 Path("/opt/remag/check_scf_control_gpu.py"), Path("/opt/remag/check_scf_cycle.py"), Path("/opt/remag/check_initial_density.py"),
                 Path("/opt/remag/check_small_eigensolver.py"), Path("/opt/remag/check_block_jacobi_gpu.py"),
                 Path("/opt/remag/replay_block_failure.py"), Path("/opt/remag/build_cupti_observer.py"),
                 Path("/opt/remag/upstream.lock.json"), Path("/opt/remag/modal_app.py")]
        for path in files:
            if path.is_file():
                relative=path.relative_to("/opt/remag")
                if relative.parts[0]=="siesta_remag":
                    relative=Path("src")/relative
                elif path.name in ("patch_siesta.py","check_gpu_solver.py","check_fermi.py","check_hartree.py",
                                    "check_pbe.py","check_pbe_gpu.py","check_grid_potential.py","check_mixing.py","check_grid_force_formula.py","check_grid_force_gpu.py","check_atomic_grid_force_formula.py","check_atomic_grid_force_gpu.py","check_final_force.py","check_integral_publication.py","check_orbital_basis.py","check_orbital_basis_gpu.py","check_orbital_comparison.py","check_orbital_owner.py","check_atomic_basis.py","check_atomic_basis_gpu.py","check_atomic_owner.py","check_atomic_comparison.py","check_atomic_masked_force.py","check_grid_owner.py","check_two_center_comparison.py","check_two_center_owner.py","check_pair_handoff.py","check_pair_force.py","check_pair_force_gpu.py","check_population.py","check_initial_state.py","check_population_gpu.py","check_population_owner.py","check_population_observables.py","check_soc_projector.py","check_soc_projector_gpu.py","check_nonlocal_force.py","check_nonlocal_force_gpu.py","check_nonlocal_compact.py","check_nonlocal_compact_gpu.py","check_nonlocal_owner.py","check_pair_geometry.py","check_two_center.py","check_two_center_gpu.py",
                                    "check_scf_scalars.py","check_scf_scalars_gpu.py",
                                    "check_hamiltonian_seeds.py","check_hamiltonian_seeds_gpu.py","check_sparse_state.py",
                                    "check_scf_convergence.py","check_scf_convergence_gpu.py","check_scf_control_gpu.py","check_scf_cycle.py","check_initial_density.py",
                                    "check_small_eigensolver.py","check_block_jacobi_gpu.py","replay_block_failure.py","build_cupti_observer.py"):
                    relative=Path("scripts")/path.name
                archive.add(path, arcname=str(relative))
        for path in Path("/opt/remag/solver-regressions").glob("*"):
            if path.is_file():archive.add(path,arcname="benchmarks/solver-regressions/"+path.name)
        archive.add(Path(__file__),arcname="runtime/modal_app.py")
    check_log = root / "kernel-check.log"
    try:
        with check_log.open("w") as output:
            checked = subprocess.run(["python", "/opt/remag/check_gpu_solver.py",
                        "/opt/candidate/remag-native/libremag_gpu.so", str(root / "kernel-check.json")],
                        stdout=output, stderr=subprocess.STDOUT, cwd=root,timeout=180)
    finally:
        volume.commit()
    print(check_log.read_text(), flush=True)
    checked.check_returncode()
    # Check the FD kernel against unmodified upstream before any SCF run.
    fermi_log=root/"fermi-check.log"
    try:
        with fermi_log.open("w") as output:
            checked=subprocess.run(["python","/opt/remag/check_fermi.py","--upstream","/opt/siesta-src",
                "--output",str(root/"fermi-check.json"),"--library","/opt/candidate/remag-native/libremag_fermi_check.so"],
                stdout=output,stderr=subprocess.STDOUT,cwd=root,timeout=120)
    finally:
        volume.commit()
    print(fermi_log.read_text(),flush=True)
    checked.check_returncode()
    # Isolated periodic-Hartree reference checks before the optional SCF path.
    hartree_log=root/"hartree-check.log"
    try:
        with hartree_log.open("w") as output:
            checked=subprocess.run(["python","/opt/remag/check_hartree.py","--upstream","/opt/siesta-src",
                "--build","/opt/reference","--output",str(root/"hartree-check.json"),
                "--library","/opt/candidate/remag-native/libremag_hartree_check.so"],
                stdout=output,stderr=subprocess.STDOUT,cwd=root,timeout=120)
    finally:
        volume.commit()
    print(hartree_log.read_text(),flush=True)
    checked.check_returncode()
    # Validate each numerical component independently of its SCF adapter.
    pbe_log=root/"pbe-check.log"
    try:
        with pbe_log.open("w") as output:
            checked=subprocess.run(["python","/opt/remag/check_pbe_gpu.py","--upstream","/opt/siesta-src",
                "--build","/opt/reference","--output",str(root/"pbe-check.json"),
                "--library","/opt/candidate/remag-native/libremag_pbe_check.so"],
                stdout=output,stderr=subprocess.STDOUT,cwd=root,timeout=180)
    finally:
        volume.commit()
    print(pbe_log.read_text(),flush=True)
    checked.check_returncode()
    flow_log=root/"grid-potential-check.log"
    try:
        with flow_log.open("w") as output:
            checked=subprocess.run(["python","/opt/remag/check_grid_potential.py","--upstream","/opt/siesta-src",
                "--build","/opt/reference","--output",str(root/"grid-potential-check.json"),
                "--library","/opt/candidate/remag-native/libremag_grid_potential_check.so",
                "--bridge-library","/opt/candidate/remag-native/libremag_gpu.so"],
                stdout=output,stderr=subprocess.STDOUT,cwd=root,timeout=180)
    finally:
        volume.commit()
    print(flow_log.read_text(),flush=True)
    checked.check_returncode()
    scalar_log=root/"scf-scalars-check.log"
    try:
        with scalar_log.open("w") as output:
            checked=subprocess.run(["python","/opt/remag/check_scf_scalars_gpu.py","--upstream","/opt/siesta-src",
                "--build","/opt/reference","--output",str(root/"scf-scalars-check.json"),
                "--library","/opt/candidate/remag-native/libremag_scf_scalars_check.so"],
                stdout=output,stderr=subprocess.STDOUT,cwd=root,timeout=120)
    finally:
        volume.commit()
    print(scalar_log.read_text(),flush=True)
    checked.check_returncode()
    pulay_log=root/"pulay-check.log"
    try:
        with pulay_log.open("w") as output:
            checked=subprocess.run(["python","/opt/remag/check_mixing.py","--upstream","/opt/siesta-src",
                "--build","/opt/reference","--output",str(root/"pulay-check.json"),
                "--library","/opt/candidate/remag-native/libremag_pulay_check.so"],
                stdout=output,stderr=subprocess.STDOUT,cwd=root,timeout=120)
    finally:
        volume.commit()
    print(pulay_log.read_text(),flush=True)
    if checked.returncode:
        # Diagnose a failed component on this same instance before teardown.
        # This does not retry or qualify SCF, and never starts another GPU.
        import shutil
        sanitizer=shutil.which("compute-sanitizer")
        if sanitizer:
            diagnostic=root/"pulay-memcheck.log"
            try:
                with diagnostic.open("w") as output:
                    subprocess.run([sanitizer,"--tool","memcheck","--error-exitcode","1",
                        "python","/opt/remag/check_mixing.py","--upstream","/opt/siesta-src",
                        "--build","/opt/reference","--output",str(root/"pulay-memcheck.json"),
                        "--library","/opt/candidate/remag-native/libremag_pulay_check.so"],
                        stdout=output,stderr=subprocess.STDOUT,cwd=root,timeout=120)
            finally:
                volume.commit()
            print(diagnostic.read_text(),flush=True)
    standalone={"pulay":{"integrated_into_siesta":resident_scf,"passed":checked.returncode==0,
                         "returncode":checked.returncode,"log":"pulay-check.log"}}
    seeds_log=root/"hamiltonian-seeds-check.log"
    try:
        with seeds_log.open("w") as output:
            checked=subprocess.run(["python","/opt/remag/check_hamiltonian_seeds_gpu.py","--upstream","/opt/siesta-src",
                "--build","/opt/reference","--output",str(root/"hamiltonian-seeds-check.json"),
                "--library","/opt/candidate/remag-native/libremag_hamiltonian_seeds_check.so"],
                stdout=output,stderr=subprocess.STDOUT,cwd=root,timeout=120)
    finally:
        volume.commit()
    print(seeds_log.read_text(),flush=True)
    standalone["hamiltonian_seeds"]={"integrated_into_siesta":resident_scf,"passed":checked.returncode==0,
                                    "returncode":checked.returncode,"log":"hamiltonian-seeds-check.log"}
    if resident_scf:
        owner_log=root/"sparse-state-check.log"
        owner_dir=root/"sparse-state-check"
        owner_dir.mkdir()
        try:
            with owner_log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_sparse_state.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--output",str(owner_dir/"check.json"),
                    "--library","/opt/candidate/remag-native/libremag_sparse_state_check.so"],
                    stdout=output,stderr=subprocess.STDOUT,cwd=owner_dir,timeout=180)
        finally:
            volume.commit()
        print(owner_log.read_text(),flush=True)
        standalone["sparse_state"]={"integrated_into_siesta":True,"passed":checked.returncode==0,
                                    "returncode":checked.returncode,"log":"sparse-state-check.log"}
    convergence_log=root/"scf-convergence-check.log"
    try:
        with convergence_log.open("w") as output:
            checked=subprocess.run(["python","/opt/remag/check_scf_convergence_gpu.py","--upstream","/opt/siesta-src",
                "--build","/opt/reference","--output",str(root/"scf-convergence-check.json"),
                "--library","/opt/candidate/remag-native/libremag_scf_convergence_check.so"],
                stdout=output,stderr=subprocess.STDOUT,cwd=root,timeout=120)
    finally:
        volume.commit()
    print(convergence_log.read_text(),flush=True)
    standalone["scf_convergence"]={"integrated_into_siesta":False,"passed":checked.returncode==0,
                                   "returncode":checked.returncode,"log":"scf-convergence-check.log"}
    if probe_scf_control:
        controls=[("scf-control",[])]
        if trace_cuda:controls.append(("scf-control-memory",["--trace"]))
        for label,extra in controls:
            directory=root/(label+"-check");directory.mkdir();log=root/(label+"-check.log")
            try:
                with log.open("w") as output:
                    checked=subprocess.run(["python","/opt/remag/check_scf_control_gpu.py","--upstream","/opt/siesta-src",
                        "--build","/opt/reference","--output",str(directory/"check.json"),*extra,
                        "--library","/opt/candidate/remag-native/libremag_scf_control_check.so"],
                        stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=360)
            finally:
                volume.commit()
            print(log.read_text(),flush=True)
            standalone[label.replace('-','_')]={"integrated_into_siesta":False,"passed":checked.returncode==0,
                                               "returncode":checked.returncode,"log":log.name}
    if probe_two_center or two_center_shadow or integral_force:
        directory=root/"two-center-check";directory.mkdir();log=root/"two-center-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_two_center_gpu.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--library","/opt/candidate/remag-native/libremag_two_center_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=600)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["two_center"]={"integrated_into_siesta":two_center_shadow or integral_force,"passed":checked.returncode==0,
                                  "returncode":checked.returncode,"log":log.name}
    if probe_two_center or two_center_shadow or integral_force:
        directory=root/"two-center-comparison-check";directory.mkdir();log=root/"two-center-comparison-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_two_center_comparison.py","--library","/opt/candidate/remag-native/libremag_two_center_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=240)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["two_center_comparison"]={"integrated_into_siesta":two_center_shadow or integral_force,"passed":checked.returncode==0,
                                             "returncode":checked.returncode,"log":log.name}
    if probe_two_center or two_center_shadow or integral_force:
        directory=root/"two-center-owner-check";directory.mkdir();log=root/"two-center-owner-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_two_center_owner.py","--executable","/opt/candidate/remag-native/remag_two_center_owner_check",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=240)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["two_center_owner"]={"integrated_into_siesta":two_center_shadow or integral_force,"passed":checked.returncode==0,
                                             "returncode":checked.returncode,"log":log.name}
    if probe_nonlocal_force or nonlocal_force_shadow or integral_force:
        directory=root/"nonlocal-owner-check";directory.mkdir();log=root/"nonlocal-owner-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_nonlocal_owner.py",
                    "--executable","/opt/candidate/remag-native/remag_nonlocal_owner_check",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=240)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["nonlocal_owner"]={"integrated_into_siesta":nonlocal_force_shadow or integral_force,"passed":checked.returncode==0,
                                      "returncode":checked.returncode,"log":log.name}
    if probe_nonlocal_force or nonlocal_force_shadow or integral_force:
        directory=root/"nonlocal-compact-check";directory.mkdir();log=root/"nonlocal-compact-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_nonlocal_compact_gpu.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--library","/opt/candidate/remag-native/libremag_nonlocal_compact_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=900)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["nonlocal_compact"]={"integrated_into_siesta":nonlocal_force_shadow or integral_force,"passed":checked.returncode==0,
                                        "returncode":checked.returncode,"log":log.name}
    if probe_nonlocal_force or nonlocal_force_shadow or integral_force:
        directory=root/"nonlocal-force-check";directory.mkdir();log=root/"nonlocal-force-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_nonlocal_force_gpu.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--library","/opt/candidate/remag-native/libremag_nonlocal_force_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=600)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["nonlocal_force"]={"integrated_into_siesta":False,"passed":checked.returncode==0,
                                      "returncode":checked.returncode,"log":log.name}
    if probe_soc_projector or nonlocal_force_shadow or integral_force:
        directory=root/"soc-projector-check";directory.mkdir();log=root/"soc-projector-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_soc_projector_gpu.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--library","/opt/candidate/remag-native/libremag_soc_projector_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=600)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["soc_projector"]={"integrated_into_siesta":False,"passed":checked.returncode==0,
                                     "returncode":checked.returncode,"log":log.name}
    if pair_force_shadow or integral_force:
        directory=root/"pair-geometry-check";directory.mkdir();log=root/"pair-geometry-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_pair_geometry.py","--executable","/opt/candidate/remag-native/remag_pair_geometry_check",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=120)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["pair_geometry"]={"integrated_into_siesta":True,"passed":checked.returncode==0,
                                    "returncode":checked.returncode,"log":log.name}
    if pair_force_shadow or integral_force:
        directory=root/"pair-handoff-check";directory.mkdir();log=root/"pair-handoff-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_pair_handoff.py","--executable","/opt/candidate/remag-native/remag_pair_handoff_check",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=240)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["pair_handoff"]={"integrated_into_siesta":True,"passed":checked.returncode==0,
                                  "returncode":checked.returncode,"log":log.name}
    if gpu_initialization and population:
        directory=root/"initial-state-check";directory.mkdir();log=root/"initial-state-check.log"
        with log.open("w") as out:
            checked=subprocess.run(["python","/opt/remag/check_initial_state.py",
                "--executable","/opt/candidate/remag-native/remag_initial_state_check",
                "--output",str(directory/"check.json")],stdout=out,stderr=subprocess.STDOUT,timeout=300)
        standalone["initial_state"]={"integrated_into_siesta":True,"passed":checked.returncode==0,
            "exit_code":checked.returncode}
        volume.commit()
    if probe_population or population:
        directory=root/"population-check";directory.mkdir();log=root/"population-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_population_gpu.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--library","/opt/candidate/remag-native/libremag_population_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=600)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["population"]={"integrated_into_siesta":population,"passed":checked.returncode==0,
                                  "returncode":checked.returncode,"log":log.name}
    if population:
        directory=root/"population-observables-check";directory.mkdir();log=root/"population-observables-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_population_observables.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--library","/opt/candidate/remag-native/libremag_population_check.so",
                    "--reference-source","/opt/remag/native/population_observables_reference.f90",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=120)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["population_observables"]={"integrated_into_siesta":True,"passed":checked.returncode==0,
                                              "returncode":checked.returncode,"log":log.name}
    if population:
        directory=root/"population-owner-check";directory.mkdir();log=root/"population-owner-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_population_owner.py",
                    "--executable","/opt/candidate/remag-native/remag_population_owner_check",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=240)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["population_owner"]={"integrated_into_siesta":True,"passed":checked.returncode==0,
                                        "returncode":checked.returncode,"log":log.name}
    if probe_pair_force or pair_force_shadow or integral_force:
        directory=root/"pair-force-check";directory.mkdir();log=root/"pair-force-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_pair_force_gpu.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--library","/opt/candidate/remag-native/libremag_pair_force_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=600)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["pair_force"]={"integrated_into_siesta":pair_force_shadow or integral_force,"passed":checked.returncode==0,
                                  "returncode":checked.returncode,"log":log.name}
    if probe_atomic_basis or atomic_basis_shadow or atomic_basis:
        directory=root/"atomic-basis-check";directory.mkdir();log=root/"atomic-basis-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_atomic_basis_gpu.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--library","/opt/candidate/remag-native/libremag_atomic_basis_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=240)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["atomic_basis"]={"integrated_into_siesta":atomic_basis_shadow or atomic_basis,"passed":checked.returncode==0,
                                   "returncode":checked.returncode,"log":log.name}
    if atomic_basis_shadow or atomic_basis:
        directory=root/"atomic-comparison-check";directory.mkdir();log=root/"atomic-comparison-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_atomic_comparison.py",
                    "--library","/opt/candidate/remag-native/libremag_atomic_basis_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=240)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["atomic_comparison"]={"integrated_into_siesta":True,"passed":checked.returncode==0,
                                   "returncode":checked.returncode,"log":log.name}
    if atomic_basis_shadow or atomic_basis:
        directory=root/"atomic-owner-check";directory.mkdir();log=root/"atomic-owner-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_atomic_owner.py",
                    "--executable","/opt/candidate/remag-native/remag_atomic_owner_check",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=240)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["atomic_owner"]={"integrated_into_siesta":True,"passed":checked.returncode==0,
                                   "returncode":checked.returncode,"log":log.name}
    if probe_orbital_basis or orbital_basis_shadow or orbital_basis:
        directory=root/"orbital-basis-check";directory.mkdir();log=root/"orbital-basis-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_orbital_basis_gpu.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--library","/opt/candidate/remag-native/libremag_orbital_basis_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=180)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["orbital_basis"]={"integrated_into_siesta":orbital_basis_shadow or orbital_basis,"passed":checked.returncode==0,
                                    "returncode":checked.returncode,"log":log.name}
    if probe_orbital_basis or orbital_basis_shadow or orbital_basis:
        directory=root/"orbital-comparison-check";directory.mkdir();log=root/"orbital-comparison-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_orbital_comparison.py",
                    "--library","/opt/candidate/remag-native/libremag_orbital_basis_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=120)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["orbital_comparison"]={"integrated_into_siesta":orbital_basis_shadow or orbital_basis,"passed":checked.returncode==0,
                                         "returncode":checked.returncode,"log":log.name}
    if probe_orbital_basis or orbital_basis_shadow or orbital_basis:
        directory=root/"orbital-owner-check";directory.mkdir();log=root/"orbital-owner-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_orbital_owner.py",
                    "--executable","/opt/candidate/remag-native/remag_orbital_owner_check",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=240)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["orbital_owner"]={"integrated_into_siesta":orbital_basis_shadow or orbital_basis,"passed":checked.returncode==0,
                                    "returncode":checked.returncode,"log":log.name}
    if probe_atomic_grid_force or atomic_force_shadow or atomic_grid_force:
        directory=root/"atomic-grid-force-check";directory.mkdir();log=root/"atomic-grid-force-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_atomic_grid_force_gpu.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--library","/opt/candidate/remag-native/libremag_atomic_grid_force_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=180)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["atomic_grid_force"]={"integrated_into_siesta":atomic_force_shadow or atomic_grid_force,"passed":checked.returncode==0,
                                         "returncode":checked.returncode,"log":log.name}
    if atomic_basis:
        directory=root/"atomic-masked-force-check";directory.mkdir();log=root/"atomic-masked-force-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_atomic_masked_force.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--library","/opt/candidate/remag-native/libremag_atomic_grid_force_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=240)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["atomic_masked_force"]={"integrated_into_siesta":True,"passed":checked.returncode==0,
                                          "returncode":checked.returncode,"log":log.name}
    if integral_force:
        directory=root/"integral-publication-check";directory.mkdir();log=root/"integral-publication-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_integral_publication.py",
                    "--executable","/opt/candidate/remag-native/remag_integral_publication_check",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=240)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["integral_publication"]={"integrated_into_siesta":True,"passed":checked.returncode==0,
                                            "returncode":checked.returncode,"log":log.name}
    if atomic_grid_force:
        directory=root/"final-force-check";directory.mkdir();log=root/"final-force-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_final_force.py",
                    "--library","/opt/candidate/remag-native/libremag_final_force_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=180)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["final_force"]={"integrated_into_siesta":True,"passed":checked.returncode==0,
                                   "returncode":checked.returncode,"log":log.name}
    if force_shadow or grid_force:
        directory=root/"grid-owner-check";directory.mkdir();log=root/"grid-owner-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_grid_owner.py",
                    "--executable","/opt/candidate/remag-native/remag_grid_owner_check",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=180)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["grid_owner"]={"integrated_into_siesta":True,"passed":checked.returncode==0,
                                  "returncode":checked.returncode,"log":log.name}
    if probe_grid_force or force_shadow or grid_force:
        directory=root/"grid-force-check";directory.mkdir();log=root/"grid-force-check.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_grid_force_gpu.py","--upstream","/opt/siesta-src",
                    "--build","/opt/reference","--library","/opt/candidate/remag-native/libremag_grid_force_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=180)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone["grid_force"]={"integrated_into_siesta":force_shadow or grid_force,"passed":checked.returncode==0,
                                  "returncode":checked.returncode,"log":log.name}
    if probe_scf_cycle:
        cycle_checks=[("scf-cycle",[])]
        if trace_cuda:cycle_checks.append(("scf-cycle-memory",["--trace"]))
        if trace_io:cycle_checks.append(("scf-cycle-io",["--trace-io"]))
        for label,extra in cycle_checks:
            directory=root/(label+"-check");directory.mkdir();log=root/(label+"-check.log")
            try:
                with log.open("w") as output:
                    checked=subprocess.run(["python","/opt/remag/check_scf_cycle.py","--upstream","/opt/siesta-src",
                        "--build","/opt/reference","--output",str(directory/"check.json"),*extra,
                        "--library","/opt/candidate/remag-native/libremag_scf_cycle_check.so"],
                        stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=480)
            finally:
                volume.commit()
            print(log.read_text(),flush=True)
            standalone[label.replace('-','_')]={"integrated_into_siesta":False,"passed":checked.returncode==0,
                                               "returncode":checked.returncode,"log":log.name}
    # These plans are exercised by production only under --resident-scf.
    if probe_small_solver:
        small_dir=root/"small-eigensolver-check"
        small_dir.mkdir()
        small_log=root/"small-eigensolver-check.log"
        try:
            with small_log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/check_small_eigensolver.py",
                    "--output",str(small_dir/"check.json"),
                    "--library","/opt/candidate/remag-native/libremag_small_eigensolver_check.so"],
                    stdout=output,stderr=subprocess.STDOUT,cwd=small_dir,timeout=180)
        finally:
            volume.commit()
        print(small_log.read_text(),flush=True)
        standalone["small_eigensolver"]={"integrated_into_siesta":False,"passed":checked.returncode==0,
                                        "returncode":checked.returncode,"log":small_log.name}
    if probe_block_solver or block_solver or block_width!=64:
        # Keep numerical controls separate from profiling. Conditional kernel
        # instrumentation has failed with both CUPTI 12.6 and 12.8. The optional
        # transfer-only observer has its own explicit calibration and scope.
        # Fresh worker processes isolate each case and each observer choice.
        block_checks=[("block-jacobi-untraced",[])]
        if probe_block_solver:
            block_checks.append(("block-jacobi-memory",["--trace","--memory-only","--observer","/opt/candidate/remag-native/libremag_activity.so"]))
        for label,extra in block_checks:
            block_dir=root/(label+"-check")
            block_dir.mkdir()
            block_log=root/(label+"-check.log")
            try:
                with block_log.open("w") as output:
                    checked=subprocess.run(["python","/opt/remag/check_block_jacobi_gpu.py",
                        "--output",str(block_dir/"check.json"),*extra,"--regressions","/opt/remag/solver-regressions",
                        "--library","/opt/candidate/remag-native/libremag_block_jacobi_check.so"],
                        stdout=output,stderr=subprocess.STDOUT,cwd=block_dir,timeout=900)
            finally:
                volume.commit()
            print(block_log.read_text(),flush=True)
            standalone[label.replace('-','_')]={"integrated_into_siesta":(block_solver or (device_scf and block_width!=64)) and label=="block-jacobi-untraced",
                                                "passed":checked.returncode==0,
                                                "returncode":checked.returncode,"log":block_log.name}
    for index,source in enumerate(replay_sources):
        label="block-failure-replay"+(f"-{index+1}" if index else "")
        directory=root/label;directory.mkdir();log=directory/"stdout.log"
        try:
            with log.open("w") as output:
                checked=subprocess.run(["python","/opt/remag/replay_block_failure.py",
                    "--source",str(Path("/results")/source),
                    "--library","/opt/candidate/remag-native/libremag_block_jacobi_check.so",
                    "--output",str(directory/"check.json")],stdout=output,stderr=subprocess.STDOUT,cwd=directory,timeout=900)
        finally:
            volume.commit()
        print(log.read_text(),flush=True)
        standalone[label.replace("-","_")]={"integrated_into_siesta":False,"passed":checked.returncode==0,
                                            "returncode":checked.returncode,"log":str(log.relative_to(root))}
    # A failure is fatal before running that integration; independent grid
    # experiments may retain their own evidence before reporting the failure.
    (root/"standalone-component-status.json").write_text(json.dumps(standalone,indent=2)+"\n")
    volume.commit()
    if (resident_scf or force_shadow or grid_force) and not all(item['passed'] for item in standalone.values() if item['integrated_into_siesta']):
        raise RuntimeError("Resident SCF component preflight failed; evidence retained")
    if device_scf and trace_cuda and not standalone.get("scf_cycle_memory",{}).get("passed"):
        raise RuntimeError("Conditional-cycle observer preflight failed; production trace was not launched")
    if trace_cuda:
        from siesta_remag.cuda_trace import summarize_cuda
        activity_executable="/opt/candidate/remag-native/remag_activity_check"
        env={**os.environ,"REMAG_TRACE_CUDA":"1","REMAG_ACTIVITY_EXECUTABLE":activity_executable,
             "LD_PRELOAD":"/opt/candidate/remag-native/libremag_activity.so"}
        for graph_mode in (False,True):
            name="activity-graph-check" if graph_mode else "activity-check"
            activity_dir=root/name
            activity_dir.mkdir()
            try:
                with (activity_dir/"stdout.log").open("w") as output:
                    checked=subprocess.run([activity_executable]+(["--graph"] if graph_mode else []),
                        cwd=activity_dir,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=60)
            finally:
                volume.commit()
            print((activity_dir/"stdout.log").read_text(),flush=True)
            activity=summarize_cuda(json.loads((activity_dir/"remag-activity.json").read_text()),
                dict(total_explicit=dict(host_to_device_bytes=32768,device_to_host_bytes=32776),accounting_errors=[]),
                expected_scf_iterations=2,process_succeeded=checked.returncode==0)
            phases=activity.get("phases",{})
            checks={"coverage":activity["coverage_complete"],
                    "launch_attribution":activity.get("phase_attribution_verified") is True,
                    "initial_upload":phases.get("initialization",{}).get("host_to_device_bytes")==32768,
                    "final_export":phases.get("post_scf",{}).get("device_to_host_bytes")==32768,
                    "scf_scalar_bytes":phases.get("scf",{}).get("device_to_host_bytes")==8,
                    "steady_scalar_bytes":activity.get("steady_state",{}).get("device_to_host_bytes")==4,
                    "scf_array_copy":phases.get("scf",{}).get("device_to_device_bytes")==32768*(2 if graph_mode else 1),
                    "scf_kernels":phases.get("scf",{}).get("kernel_count")==2}
            activity.update(graph_replay=graph_mode,validation_checks=checks,passed=all(checks.values()))
            (root/(name+".json")).write_text(json.dumps(activity,indent=2)+"\n")
            volume.commit()
            print("CUPTI known-copy validation (graph="+str(graph_mode)+"): "+json.dumps(checks),flush=True)
            if not activity["passed"]:
                raise RuntimeError("CUDA activity observer failed its known-copy validation: "+name)
    if probe_control:
        probe_path=root/"graph-probe.json"
        probe=dict(scope="isolated solver graph capability; not an SCF residency trace",checks=[])
        for algorithm_id,algorithm in ((0,"generalized_divide_conquer"),(1,"generalized_jacobi"),
                                       (2,"reduced_jacobi"),(3,"reduced_batched")):
            for complex_flag,n in ((c,n) for c in (0,1) for n in (64,576,1152)):
                single=root/f"graph-probe-{algorithm}-{n}-{'complex' if complex_flag else 'real'}.json"
                try:
                    subprocess.run(["/opt/candidate/remag-native/remag_graph_probe",str(single),str(n),str(complex_flag),str(algorithm_id)],
                                   check=True,timeout=60,cwd=root)
                    result=json.loads(single.read_text())
                    probe["checks"].extend(result["checks"])
                    probe["cuda_build_version"]=result["cuda_build_version"]
                except (subprocess.SubprocessError,ValueError,OSError) as error:
                    probe["checks"].append(dict(n=n,complex=bool(complex_flag),algorithm=algorithm,isolated_process=True,
                                                error=str(error),verified=False))
                probe_path.write_text(json.dumps(probe,indent=2)+"\n")
                volume.commit()
        print("Solver conditional-graph capability: "+json.dumps(probe),flush=True)
    ledger, results, instrumentation, variants = [], {}, [], {}
    try:
        for case in cases:
            variants[case]={}
            prepared = prepare(case, Path("/opt/siesta-src"), root / "inputs" / case, io_policy=io_policy, split_scf_timings=not device_scf, initial_populations=initial_populations)
            atom_count = json.loads((prepared / "manifest.json").read_text())["atoms"]
            def execute(role, enable_dm, enable_epoch=True, enable_normalize=normalize, enable_defer=defer_scf_log,
                        enable_occupations=occupations, enable_spectra=resident_spectra,
                        enable_matrices=matrix_assembly, enable_hartree=hartree, enable_flow=resident_grid,
                        enable_state=resident_scf, enable_block=block_solver, enable_cycle=device_scf, enable_grid_force=grid_force, enable_atomic_force=atomic_grid_force, enable_final_grid=final_grid, enable_final_density=final_density, enable_orbital_basis=orbital_basis, enable_atomic_basis=atomic_basis, enable_vector_cache=vector_cache):
                binary = f"/opt/{'reference' if role == 'reference' else 'candidate'}/Src/siesta"
                if role=="reference" and reference_run:
                    import shutil
                    previous=Path("/results")/reference_run/case/"reference"
                    result=json.loads((previous/"result.json").read_text())
                    manifest=json.loads((prepared/"manifest.json").read_text())
                    for key in ("input_sha256","pseudopotentials","atoms","spin"):
                        if result["benchmark"][key]!=manifest[key]:
                            raise RuntimeError("Saved reference input mismatch: "+key)
                    if not result["valid"] or result["executable"]["sha256"]!=sha256(Path(binary)):
                        raise RuntimeError("Saved reference is invalid or has a different executable")
                    shutil.copytree(previous,root/case/role)
                    variants[case][role]=result
                    print(f"Reused {case} CPU reference from {reference_run}; cross-allocation performance is unqualified",flush=True)
                    volume.commit()
                    return result
                print(f"Running {case} {role}", flush=True)
                deferred=enable_defer and role != "reference"
                log_env=({"GFORTRAN_FORMATTED_BUFFER_SIZE":"1048576", "GFORTRAN_UNBUFFERED_ALL":"0",
                          "GFORTRAN_UNBUFFERED_PRECONNECTED":"0"} if deferred else {})
                result = run_case(prepared, Path(binary), root / case / role, threads=1, timeout=3600 if atom_count >= 100 else 900,
                                  extra_env={"REMAG_GRID": "1" if grid and role != "reference" else "0",
                                             "REMAG_DM": "1" if enable_dm else "0",
                                             "REMAG_GRID_POTENTIAL_SPINS": "1" if grid_potential_spins and role != "reference" else "0",
                                             "REMAG_GRID_EPOCH": "1" if enable_epoch else "0",
                                             "REMAG_GRID_VERIFY_EPOCH": "1" if verify_epoch else "0",
                                             "REMAG_NORMALIZE": "1" if enable_normalize and enable_dm else "0",
                                             "REMAG_OCCUPATIONS": "1" if enable_occupations and enable_dm else "0",
                                             "REMAG_RESIDENT_SPECTRA": "1" if enable_spectra and enable_occupations and enable_dm else "0",
                                             "REMAG_HS": "1" if enable_matrices and enable_spectra and enable_occupations and enable_dm else "0",
                                             "REMAG_HARTREE": "1" if enable_hartree and role != "reference" else "0",
                                             "REMAG_RESIDENT_GRID": "1" if enable_flow and role != "reference" else "0",
                                             "REMAG_RESIDENT_SCF": "1" if enable_state and role != "reference" else "0",
                                             "REMAG_FORCE_SHADOW_RESIDENT": "1" if force_shadow_resident and role != "reference" else "0",
                                             "REMAG_FORCE_SHADOW": "1" if force_shadow and role != "reference" else "0",
                                             "REMAG_ORBITAL_BASIS": "1" if enable_orbital_basis and role != "reference" else "0",
                                             "REMAG_ATOMIC_BASIS": "1" if enable_atomic_basis and role != "reference" else "0",
                                             "REMAG_INTEGRAL_FORCE": "1" if integral_force and role != "reference" else "0",
                                             "REMAG_POPULATION": "1" if population and role != "reference" else "0",
                                             "REMAG_TWO_CENTER_SHADOW": "1" if two_center_shadow and role != "reference" else "0",
                                             "REMAG_PAIR_FORCE_SHADOW": "1" if pair_force_shadow and role != "reference" else "0",
                                             "REMAG_NONLOCAL_H0_POISON_HOST": "1" if poison_nonlocal_h0 and role != "reference" else "0",
                                             "REMAG_NONLOCAL_H0": "1" if nonlocal_h0 and role != "reference" else "0",
                                             "REMAG_NONLOCAL_H0_SHADOW": "1" if nonlocal_h0_shadow and role != "reference" else "0",
                                             "REMAG_NONLOCAL_FORCE_SHADOW": "1" if nonlocal_force_shadow and role != "reference" else "0",
                                             "REMAG_ATOMIC_BASIS_SHADOW": "1" if atomic_basis_shadow and role != "reference" else "0",
                                             "REMAG_ORBITAL_BASIS_SHADOW": "1" if orbital_basis_shadow and role != "reference" else "0",
                                             "REMAG_GRID_FORCE": "1" if enable_grid_force and role != "reference" else "0",
                                             "REMAG_ATOMIC_FORCE_SHADOW": "1" if atomic_force_shadow and role != "reference" else "0",
                                             "REMAG_FINAL_GRID": "1" if enable_final_grid and role != "reference" else "0",
                                             "REMAG_FINAL_GRID_SHADOW": "1" if final_grid_shadow and role != "reference" else "0",
                                             "REMAG_FINAL_DENSITY": "1" if enable_final_density and role != "reference" else "0",
                                             "REMAG_FINAL_DENSITY_SHADOW": "1" if final_density_shadow and role != "reference" else "0",
                                             "REMAG_ATOMIC_GRID_FORCE": "1" if enable_atomic_force and role != "reference" else "0",
                                             "REMAG_BLOCK_WIDTH": str(block_width),
                                             "REMAG_BLOCK_LOCAL_SOLVER": block_local_solver,
                                             "REMAG_BLOCK_SHARED_JACOBI": "1" if shared_local_solver and role != "reference" else "0",
                                             "REMAG_BLOCK_FUSED_MATRIX": "1" if fused_matrix_rotation and enable_cycle and enable_vector_cache and role != "reference" else "0",
                                             "REMAG_BLOCK_FP64_VECTORS": "1" if fp64_vector_rotation and enable_cycle and enable_vector_cache and role != "reference" else "0",
                                             "REMAG_BLOCK_FP64_MATRIX": "1" if fp64_matrix_rotation and enable_cycle and enable_vector_cache and role != "reference" else "0",
                                             "REMAG_BLOCK_SOLVER": "1" if enable_block and role != "reference" else "0",
                                             "REMAG_DEVICE_SCF": "1" if enable_cycle and role != "reference" else "0",
                                             "REMAG_INITIAL_MESH": "1" if gpu_initial_mesh and enable_cycle and role != "reference" else "0",
                                             "REMAG_GPU_INITIALIZATION": "1" if gpu_initialization and enable_cycle and role != "reference" else "0",
                                             "REMAG_GPU_HISTORY": "1" if gpu_history and enable_cycle and role != "reference" else "0",
                                             "REMAG_FINAL_SPECTRA": "1" if final_spectra and enable_cycle and role != "reference" else "0",
                                             "REMAG_GATE_H_EXPORT": "1" if gate_h_export and enable_cycle and role != "reference" else "0",
                                             "REMAG_DEVICE_VECTOR_CACHE": "1" if enable_vector_cache and role != "reference" else "0",
                                             "REMAG_SCF_SOLVER_LANES": str(solver_lanes) if enable_cycle and enable_vector_cache and role != "reference" else "1",
                                             "REMAG_SCF_WARM_START": "1" if solver_warm_start and enable_cycle and enable_vector_cache and role != "reference" else "0",
                                             "REMAG_RADIAL_BATCH": "1" if gpu_radial_tables and role != "reference" else "0",
                                             "REMAG_ACTIVITY_MEMORY_ONLY": "1" if enable_cycle else "0",
                                             "REMAG_TRACE_IO": "1" if trace_io and role != "reference" else "0",
                                             "REMAG_TRACE_CUDA": "1" if trace_cuda and role != "reference" else "0",
                                             "REMAG_DEFER_SCF_LOG": "1" if deferred else "0", **log_env,
                                             "REMAG_HOST_ID": run_id+"-"+attempt["attempt_id"]})
                variants[case][role]=result
                for requested,key in ((trace_io,"io_trace"),(trace_cuda,"cuda_trace")):
                    if requested and role != "reference":
                        trace=result["telemetry"].get(key,{})
                        complete=trace.get("coverage_complete") is True
                        instrumentation.append(dict(case=case,role=role,kind=key,passed=complete,
                                                    errors=trace.get("errors",["Requested trace missing"])))
                        (root/"instrumentation.json").write_text(json.dumps(dict(
                            passed=all(item["passed"] for item in instrumentation),checks=instrumentation),indent=2)+"\n")
                        print(f"{case} {role} {key}: coverage_complete={complete}, errors={trace.get('errors')}",flush=True)
                volume.commit()
                print(f"{case} {role}: valid={result['valid']}, {result['wall_seconds']:.3f}s", flush=True)
                if role == "reference" and result["returncode"] < 0:
                    raise RuntimeError(f"Reference terminated by signal {-result['returncode']}; stopping the batch. "
                                       f"Recover its logs from run {run_id} before retrying.")
                return result
            pair = [execute("reference", False), execute("candidate", dm)]
            comparison = compare(*pair, reference_dir=root/case/"reference", candidate_dir=root/case/"candidate")
            results[case] = pair
            record = dict(optimization="device_atomic_derivatives" if atomic_basis else "device_orbital_derivatives" if orbital_basis else "requested_density_reuse" if final_density else "final_grid_reuse" if final_grid else "shared_grid_forces" if atomic_grid_force else "resident_grid_force" if grid_force else "device_controlled_scf" if device_scf else "experimental_block_solver" if block_solver else "resident_sparse_scf" if resident_scf else "resident_grid" if resident_grid else "device_hartree_and_matrix_assembly" if hartree and matrix_assembly else "device_hartree" if hartree else "device_matrix_assembly" if matrix_assembly else "resident_spectra" if resident_spectra else "device_occupations" if occupations else "device_charge_normalization" if normalize else "resident_eigensolver_density_matrix_grid" if dm else "resident_basis_grid_and_eigensolver" if grid else "generalized_eigensolver_fp64", benchmark=case, **comparison)
            ledger.append(record)
            (root / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
            print(f"Comparison: {case} passed={comparison['passed']} speedup={comparison['end_to_end_speedup']}", flush=True)
            if not comparison["passed"]:
                if resident_scf and not pair[1]["valid"]:
                    # A broken new owner/consumer lifecycle should be fixed
                    # before spending the remaining allocation on more cases.
                    (root/"early-stop.json").write_text(json.dumps(dict(
                        passed=False,failed_case=case,requested_cases=cases,
                        completed_cases=list(results),reason="Invalid resident-SCF candidate; retained failed pair"),
                        indent=2)+"\n")
                    print("Stopping diagnostic after invalid resident-SCF candidate",flush=True)
                    break
                continue  # Retain the failed pair; additional timings cannot qualify it.
            for repeat in range(2,candidate_repeats+1):
                role=f"candidate-repeat{repeat}"
                repeated=execute(role,dm)
                comparison=compare(pair[0],repeated,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization=record["optimization"],benchmark=f"{case}-repeat{repeat}",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                (root / "ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Repeat: {case} {repeat} passed={comparison['passed']} speedup={comparison['end_to_end_speedup']}",flush=True)
            if atomic_basis_ablation:
                role="candidate-without-atomic-basis"
                control=execute(role,dm,enable_atomic_basis=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="host_atomic_derivatives",benchmark=f"{case}-without-atomic-basis",
                    reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-atomic-basis-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root/"ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Atomic-basis ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if orbital_basis_ablation:
                role="candidate-without-orbital-basis"
                control=execute(role,dm,enable_orbital_basis=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="host_orbital_derivatives",benchmark=f"{case}-without-orbital-basis",
                    reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-orbital-basis-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root/"ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Orbital-basis ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if final_density_ablation:
                role="candidate-without-final-density"
                control=execute(role,dm,enable_final_density=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="final_grid_reuse",benchmark=f"{case}-without-final-density",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-final-density-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root/"ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Final-density ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if final_grid_ablation:
                role="candidate-without-final-grid"
                control=execute(role,dm,enable_final_grid=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="shared_grid_forces",benchmark=f"{case}-without-final-grid",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-final-grid-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root/"ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Final-grid ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if atomic_force_ablation:
                role="candidate-without-atomic-force"
                control=execute(role,dm,enable_atomic_force=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="resident_grid_force",benchmark=f"{case}-without-atomic-force",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-atomic-force-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root/"ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Atomic-force ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if grid_force_ablation:
                role="candidate-without-grid-force"
                control=execute(role,dm,enable_grid_force=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="device_controlled_scf",benchmark=f"{case}-without-grid-force",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-grid-force-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root/"ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Grid-force ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if block_solver_ablation:
                role="candidate-without-block-solver"
                control=execute(role,dm,enable_block=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="resident_sparse_scf",benchmark=f"{case}-without-block-solver",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-block-solver-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root/"ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Block solver ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if ablation:
                role="candidate-without-dm"
                control=execute(role,False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="resident_basis_grid_and_eigensolver",benchmark=f"{case}-without-dm",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root / "ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"DM ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if epoch_ablation:
                role="candidate-without-epoch"
                control=execute(role,dm,False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="content_validated_grid",benchmark=f"{case}-without-epoch",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-epoch-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root / "ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Epoch ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if normalization_ablation:
                role="candidate-without-normalization"
                control=execute(role,dm,enable_normalize=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="host_charge_normalization",benchmark=f"{case}-without-normalization",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-normalization-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root / "ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Normalization ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if occupation_ablation:
                role="candidate-without-occupations"
                control=execute(role,dm,enable_occupations=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="host_occupations",benchmark=f"{case}-without-occupations",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-occupations-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root / "ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Occupation ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if spectral_ablation:
                role="candidate-without-resident-spectra"
                control=execute(role,dm,enable_spectra=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="per_iteration_spectral_export",benchmark=f"{case}-without-resident-spectra",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-spectra-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root / "ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Spectral ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if matrix_ablation:
                role="candidate-without-matrix-assembly"
                control=execute(role,dm,enable_matrices=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="host_matrix_assembly",benchmark=f"{case}-without-matrix-assembly",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-matrices-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root / "ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Matrix ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if grid_flow_ablation:
                role="candidate-without-resident-grid"
                control=execute(role,dm,enable_flow=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="host_grid_consumers",benchmark=f"{case}-without-resident-grid",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-grid-flow-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root/"ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Grid-flow ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if vector_cache_ablation:
                role="candidate-without-vector-cache"
                control=execute(role,dm,enable_vector_cache=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="recomputed_device_eigenvectors",benchmark=f"{case}-without-vector-cache",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-vector-cache-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root/"ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Vector-cache ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if device_cycle_ablation:
                role="candidate-without-device-cycle"
                control=execute(role,dm,enable_cycle=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="host_controlled_resident_gpu",benchmark=f"{case}-without-device-cycle",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-device-cycle-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root/"ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Device-cycle ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if sparse_state_ablation:
                role="candidate-without-resident-scf"
                control=execute(role,dm,enable_state=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="host_sparse_scf_consumers",benchmark=f"{case}-without-resident-scf",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-sparse-state-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root/"ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Sparse-state ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if hartree_ablation:
                role="candidate-without-hartree"
                control=execute(role,dm,enable_hartree=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="host_hartree",benchmark=f"{case}-without-hartree",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-hartree-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root / "ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Hartree ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
            if log_ablation:
                role="candidate-without-deferred-log"
                control=execute(role,dm,enable_defer=False)
                comparison=compare(pair[0],control,reference_dir=root/case/"reference",candidate_dir=root/case/role)
                ledger.append(dict(optimization="per_iteration_log_flush",benchmark=f"{case}-without-deferred-log",
                                   reference_path=f"{case}/reference",candidate_path=f"{case}/{role}",**comparison))
                delta=compare(control,pair[1],reference_dir=root/case/role,candidate_dir=root/case/"candidate")
                (root/f"ablation-log-{case}.json").write_text(json.dumps(delta,indent=2)+"\n")
                (root / "ledger.json").write_text(json.dumps(ledger,indent=2)+"\n")
                print(f"Log ablation: {case} passed={delta['passed']} incremental_speedup={delta['end_to_end_speedup']}",flush=True)
        for family, label in (("fept2_soc","fept2"),("fept16_soc","fept16"),("fept2_soc_gamma","fept2-gamma")):
            x, z = f"{family}_x", f"{family}_z"
            if x in results and z in results:
                # Every repeated/control directional pair must preserve MAE;
                # passing two absolute-energy gates alone is insufficient.
                for role in sorted((set(variants[x])|set(variants[z]))-{"reference"}):
                    if role not in variants[x] or role not in variants[z]:
                        mae=dict(passed=False,reason="Missing one orientation for this variant")
                    else:
                        mae=compare_mae(variants[x]["reference"],variants[z]["reference"],variants[x][role],variants[z][role],
                            directories=[root/x/"reference",root/z/"reference",root/x/role,root/z/role])
                    mae["variant"]=role
                    suffix="" if role=="candidate" else f"-{role}"
                    (root/f"mae-{label}{suffix}.json").write_text(json.dumps(mae,indent=2)+"\n")
    finally:
        volume.commit()
    # Seal only after all execution and result files have committed. A lost
    # response can then be repackaged without rerunning or mixing allocations.
    complete_attempt(root)
    volume.commit()
    return package_evidence(root)

@app.local_entrypoint()
def main(cases: str = "fe16_collinear,fe2_noncollinear,fept2_soc_z,fept2_soc_x,fe54_collinear",
         grid: bool = False, dm: bool = False, ablation: bool = False, candidate_repeats: int = 1,
         io_policy: str = "end-of-cycle", epoch_ablation: bool = False, verify_epoch: bool = False,
         normalize: bool = False, normalization_ablation: bool = False, probe_control: bool = False, trace_io: bool = False,
         defer_scf_log: bool = False, log_ablation: bool = False, trace_cuda: bool = False,
         occupations: bool = False, occupation_ablation: bool = False,
         resident_spectra: bool = False, spectral_ablation: bool = False,
         matrix_assembly: bool = False, matrix_ablation: bool = False,
         hartree: bool = False, hartree_ablation: bool = False,
         resident_grid: bool = False, grid_flow_ablation: bool = False,
         resident_scf: bool = False, sparse_state_ablation: bool = False, probe_small_solver: bool = False, probe_block_solver: bool = False,
         block_solver: bool = False, block_solver_ablation: bool = False, probe_scf_control: bool = False, probe_scf_cycle: bool = False,
         gpu_model: str = "A100-40GB", device_scf: bool = False, device_cycle_ablation: bool = False, reference_run: str = "", replay_block_failure: str = "", block_width: int = 64, block_local_solver: str = "syev", probe_grid_force: bool = False, force_shadow: bool = False, force_shadow_resident: bool = False, grid_force: bool = False, grid_force_ablation: bool = False, probe_atomic_grid_force: bool = False, atomic_force_shadow: bool = False, atomic_grid_force: bool = False, atomic_force_ablation: bool = False, final_grid_shadow: bool = False, final_grid: bool = False, final_grid_ablation: bool = False, final_density_shadow: bool = False, final_density: bool = False, final_density_ablation: bool = False, probe_orbital_basis: bool = False, orbital_basis_shadow: bool = False, orbital_basis: bool = False, orbital_basis_ablation: bool = False, probe_atomic_basis: bool = False, atomic_basis_shadow: bool = False, atomic_basis: bool = False, atomic_basis_ablation: bool = False, probe_two_center: bool = False, two_center_shadow: bool = False, probe_pair_force: bool = False, pair_force_shadow: bool = False, probe_soc_projector: bool = False, probe_nonlocal_force: bool = False, nonlocal_force_shadow: bool = False, integral_force: bool = False, probe_population: bool = False, vector_cache: bool = False, vector_cache_ablation: bool = False, population: bool = False, probe_density_history_only: bool = False, probe_initialization_only: bool = False, gpu_initialization: bool = False, initial_populations: bool = False, gpu_history: bool = False, final_spectra: bool = False, gate_h_export: bool = False, probe_nonlocal_h0_only: bool = False, nonlocal_h0_shadow: bool = False, nonlocal_h0: bool = False, poison_nonlocal_h0: bool = False, probe_memory_observer_only: bool = False, solver_lanes: int = 1, solver_warm_start: bool = False, gpu_radial_tables: bool = False, gpu_initial_mesh: bool = False, shared_local_solver: bool = False, grid_potential_spins: bool = False, fused_matrix_rotation: bool = False, fp64_vector_rotation: bool = False, fp64_matrix_rotation: bool = False):

    import datetime
    import json
    import tarfile
    import io
    if block_local_solver not in ("syev","jacobi") or (block_local_solver=="jacobi" and block_width!=32):
        raise ValueError("Batched Jacobi requires --block-width 32; other local solver is syev")
    if fused_matrix_rotation and (block_width!=32 or not (device_scf and vector_cache)):
        raise ValueError("Fused matrix rotation requires 32-state blocks, device SCF and vector cache")
    if fp64_vector_rotation and (block_width!=32 or not (device_scf and vector_cache)):
        raise ValueError("FP64 vector rotation requires 32-state blocks, device SCF and vector cache")
    if fp64_matrix_rotation and (block_width!=32 or not (device_scf and vector_cache and fused_matrix_rotation)):
        raise ValueError("FP64 matrix rotation requires 32-state blocks, device SCF, vector cache and fused matrix rotation")
    if block_width not in (32,64):
        raise ValueError("Experimental local eigensolver width must be 32 or 64")
    if solver_lanes not in (1,2,4,8) or (solver_lanes>1 and not (device_scf and vector_cache)):
        raise ValueError("Solver workspaces must be 1, 2, 4 or 8; parallel workspaces require device SCF and vector cache")
    if solver_warm_start and not (device_scf and vector_cache):
        raise ValueError("Eigensolver warm start requires device SCF and vector cache")
    if gpu_radial_tables and not integral_force:
        raise ValueError("GPU radial tables require the production integral-force path")
    if gpu_model not in ("A100-40GB","A100-80GB","H100"):
        raise ValueError("Choose A100-40GB, A100-80GB or H100; the GPU count remains one")
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"Run ID: {run_id}; full artifacts: siesta-remag-results/{run_id}", flush=True)
    function=benchmark if gpu_model=="A100-40GB" else benchmark.with_options(gpu=gpu_model+":1",max_containers=1)
    if (probe_density_history_only or probe_initialization_only) != (os.environ.get("REMAG_HISTORY_COMPONENT_ONLY")=="1"):
        raise ValueError("History-only flag requires REMAG_HISTORY_COMPONENT_ONLY=1; unset it for material runs")
    if probe_memory_observer_only != (os.environ.get("REMAG_MEMORY_COMPONENT_ONLY")=="1"):
        raise ValueError("Memory observer probe requires REMAG_MEMORY_COMPONENT_ONLY=1; unset it for other runs")
    if probe_nonlocal_h0_only != (os.environ.get("REMAG_H0_COMPONENT_ONLY")=="1"):
        raise ValueError("H0-only flag requires REMAG_H0_COMPONENT_ONLY=1; unset it for other runs")
    if sum((probe_density_history_only,probe_initialization_only,probe_nonlocal_h0_only,probe_memory_observer_only))>1:
        raise ValueError("Choose one component probe")
    # Submit exactly one bounded call. Waiting for an async call does not give
    # a disconnected synchronous RPC ownership of the remote computation.
    call = function.spawn(cases.split(","), run_id, grid, dm, ablation, candidate_repeats, io_policy,
                               epoch_ablation, verify_epoch, normalize, normalization_ablation, probe_control, trace_io,
                               defer_scf_log, log_ablation, trace_cuda, occupations, occupation_ablation,
                               resident_spectra, spectral_ablation, matrix_assembly, matrix_ablation,
                               hartree, hartree_ablation, resident_grid, grid_flow_ablation,resident_scf,sparse_state_ablation,probe_small_solver,probe_block_solver,
                               block_solver,block_solver_ablation,probe_scf_control,probe_scf_cycle,gpu_model,device_scf,device_cycle_ablation,reference_run,replay_block_failure,block_width,block_local_solver,probe_grid_force,force_shadow,force_shadow_resident,grid_force,grid_force_ablation,probe_atomic_grid_force,atomic_force_shadow,atomic_grid_force,atomic_force_ablation,final_grid_shadow,final_grid,final_grid_ablation,final_density_shadow,final_density,final_density_ablation,probe_orbital_basis,orbital_basis_shadow,orbital_basis,orbital_basis_ablation,probe_atomic_basis,atomic_basis_shadow,atomic_basis,atomic_basis_ablation,probe_two_center,two_center_shadow,probe_pair_force,pair_force_shadow,probe_soc_projector,probe_nonlocal_force,nonlocal_force_shadow,integral_force,probe_population,vector_cache,vector_cache_ablation,population,probe_density_history_only,probe_initialization_only,gpu_initialization,initial_populations,gpu_history,final_spectra,gate_h_export,probe_nonlocal_h0_only,nonlocal_h0_shadow,nonlocal_h0,poison_nonlocal_h0,probe_memory_observer_only,solver_lanes,solver_warm_start,gpu_radial_tables,gpu_initial_mesh,shared_local_solver,grid_potential_spins,fused_matrix_rotation,fp64_vector_rotation,fp64_matrix_rotation)
    print(f"Function call: {call.object_id}; recover artifacts by run ID if this client disconnects", flush=True)
    payload = call.get()
    target = ROOT / "results" / run_id
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(target, filter="data")
    print(f"Saved {target}")
    ledger = json.loads((target / "ledger.json").read_text())
    print(json.dumps(ledger, indent=2))
    extra_checks = [json.loads(p.read_text()) for pattern in ("mae-*.json","ablation-*.json") for p in target.glob(pattern)]
    if not all(r["passed"] for r in [*ledger, *extra_checks]):
        raise SystemExit("Scientific regression failed; saved all available evidence above")
    instrumentation=target/"instrumentation.json"
    if instrumentation.exists() and not json.loads(instrumentation.read_text())["passed"]:
        raise SystemExit("Instrumentation coverage incomplete; scientific and trace evidence saved separately above")
    standalone=target/"standalone-component-status.json"
    if standalone.exists() and not all(item['passed'] for item in json.loads(standalone.read_text()).values()):
        raise SystemExit("An isolated component failed; its evidence and independent SIESTA comparisons are retained separately")
