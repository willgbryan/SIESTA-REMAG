#!/usr/bin/env python3
"""Timing campaign for the REMAG paper figures, hcp Co with off-site SOC.

Every run is a fresh process in a fresh directory, timed from launch to exit.
Repetitions are interleaved (all implementations for repetition 1, then 2, ...).
One JSON record per run is appended to ~/bench/records.jsonl.
Usage: bench.py GROUP [GROUP ...]   (groups: profile lanes scaling-gpu methods-gpu
       threads mpi-ranks scaling-cpu methods-host scaling-serial)
"""
import json, os, re, shutil, subprocess, sys, threading, time
from pathlib import Path

HOME = Path.home()
ROOT = HOME / "bench"
PSML = HOME / "pseudo/nc-fr-04_pbe_standard/Co.psml"
MAKE = HOME / "co/make_siesta.py"
PY = HOME / "venv/bin/python"
REMAG = HOME / "candidate-kl/Src/siesta"
PRISTINE = HOME / "reference/Src/siesta"
PRISTINE_MPI = HOME / "reference-mpi/Src/siesta"
CORES = 28
MESHES = [(6, 6, 3), (10, 10, 6), (14, 14, 8), (20, 20, 11)]
K20 = (20, 20, 11)

COMMON = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
DEVICE = dict(REMAG_GPU="1", REMAG_GRID="1", REMAG_DM="1", REMAG_GRID_EPOCH="1", REMAG_NORMALIZE="1",
              REMAG_OCCUPATIONS="1", REMAG_RESIDENT_SPECTRA="1", REMAG_HS="1", REMAG_HARTREE="1",
              REMAG_RESIDENT_GRID="1", REMAG_RESIDENT_SCF="1", REMAG_DEVICE_SCF="1", REMAG_SCF_K_LOOP="1",
              REMAG_DEVICE_VECTOR_CACHE="1", REMAG_GRID_FORCE="1", REMAG_ATOMIC_GRID_FORCE="1",
              REMAG_FINAL_GRID="1", REMAG_FINAL_DENSITY="1", REMAG_ORBITAL_BASIS="1", REMAG_ATOMIC_BASIS="1",
              REMAG_INTEGRAL_FORCE="1", REMAG_POPULATION="1", REMAG_GPU_INITIALIZATION="1",
              REMAG_GPU_HISTORY="1", REMAG_FINAL_SPECTRA="1", REMAG_GATE_H_EXPORT="1",
              REMAG_NONLOCAL_H0="1", REMAG_INITIAL_MESH="1", REMAG_GRID_POTENTIAL_SPINS="1")


def bj(lanes):
    return dict(DEVICE, REMAG_BLOCK_WIDTH="32", REMAG_BLOCK_LOCAL_SOLVER="jacobi", REMAG_BLOCK_SHARED_JACOBI="1",
                REMAG_SCF_SOLVER_LANES=str(lanes), REMAG_SCF_WARM_START="1")


def batched(lanes):
    return dict(DEVICE, REMAG_SCF_BATCHED_SOLVER="1", REMAG_SCF_SOLVER_LANES=str(lanes))


def impl(name):
    """name -> (argv, env, extra fdf lines, uses_gpu)."""
    if name.startswith("gpu-batched-"):
        return [str(REMAG)], batched(int(name.rsplit("-", 1)[1])), [], True
    if name.startswith("gpu-bj-"):
        return [str(REMAG)], bj(int(name.rsplit("-", 1)[1])), [], True
    if name.startswith("host-"):
        return [str(REMAG)], dict(REMAG_GPU="0"), [f"REMAG.HostK.Threads {int(name[5:])}"], False
    if name == "pristine-serial":
        return [str(PRISTINE)], {}, [], False
    if name.startswith("pristine-mpi-"):
        n = int(name.rsplit("-", 1)[1])
        return (["mpirun", "-np", str(n), "--bind-to", "none", str(PRISTINE_MPI)], {},
                ["Diag.ParallelOverK true"], False)
    raise ValueError(name)


def deck(kind, mesh):
    """Generated input directory. kind: legacy (pristine-compatible: PBE legacy, unconstrained),
    lf (REMAG: local-frame PBE, constrained 45 deg), mc (multicollinear), tet (lf + optimized tetrahedra)."""
    tag = f"{kind}-{mesh[0]}x{mesh[1]}x{mesh[2]}"
    out = ROOT / "decks" / tag
    if not out.exists():
        args = [str(PY), str(MAKE), str(out), "--psml", str(PSML), "--kmesh", ",".join(map(str, mesh)),
                "--thetas", "45", "--soc", "offsite", "--h-tolerance", "1.e-4", "--shift", "0.005",
                "--mesh", "1200", "--temperature", "2000"]
        args += {"legacy": ["--nc-gga", "legacy", "--unconstrained"], "lf": ["--nc-gga", "local-frame"],
                 "mc": ["--nc-gga", "multicollinear"],
                 "tet": ["--nc-gga", "local-frame", "--occupation", "Tetrahedra.Optimized"]}[kind]
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
    return tag, next(p for p in out.iterdir() if p.is_dir())


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                self.peak = max(self.peak, int(out.split()[0]))
            except Exception:
                pass
            self.stop.wait(0.25)


def last(pattern, text, group=1):
    m = re.findall(pattern, text, re.M)
    return float(m[-1]) if m else None


def run(group, name, kind, mesh, rep, extra=(), env_extra=None, label=None):
    tag, src = deck(kind, mesh)
    label = label or name
    work = ROOT / "runs" / group / f"{tag}-{label}-r{rep}"
    if (work / "record.json").exists():
        return json.loads((work / "record.json").read_text())
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(src, work)
    argv, env_impl, fdf, gpu = impl(name)
    # fdf keeps the first occurrence of a key, so an override replaces the deck's line in place
    lines = (work / "input.fdf").read_text().splitlines()
    for line in list(fdf) + list(extra):
        key = line.split()[0].lower()
        hits = [i for i, l in enumerate(lines) if l.split() and l.split()[0].lower() == key]
        if hits:
            lines[hits[0]] = line
        else:
            lines.append(line)
    (work / "input.fdf").write_text("\n".join(lines) + "\n")
    env = dict(os.environ, **COMMON, **env_impl, **(env_extra or {}))
    sampler = GpuSampler()
    sampler.start()
    t0 = time.perf_counter()
    if argv[0] == "mpirun":
        with open(work / "output.out", "w") as out, open(work / "error.log", "w") as err:
            rc = subprocess.run(argv + ["input.fdf"], cwd=work, env=env, stdout=out, stderr=err).returncode
    else:
        with open(work / "input.fdf") as inp, open(work / "output.out", "w") as out, open(work / "error.log", "w") as err:
            rc = subprocess.run(argv, cwd=work, env=env, stdin=inp, stdout=out, stderr=err).returncode
    wall = time.perf_counter() - t0
    sampler.stop.set()
    sampler.join()
    text = (work / "output.out").read_text(errors="replace")
    rec = dict(group=group, impl=name, label=label, deck=kind, kmesh=list(mesh), nk=mesh[0] * mesh[1] * mesh[2],
               rep=rep, rc=rc, wall_s=wall, extra=list(extra), env_extra=env_extra or {},
               iterations=len(re.findall(r"^ *scf: +\d+", text, re.M)),
               etot_ev=last(r"siesta: +Total = +(-?[\d.]+)", text),
               free_ev=last(r"siesta: +FreeEng = +(-?[\d.]+)", text),
               fermi_ev=last(r"siesta: +Fermi = +(-?[\d.]+)", text),
               moment_mub=last(r"spin moment: \{S\} , \|S\| = \{.*\} +([\d.]+)", text),
               normal_exit=(work / "0_NORMAL_EXIT").exists(), gpu_peak_mib=sampler.peak if gpu else None,
               finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    for p in work.iterdir():  # keep the record small on disk; outputs we need stay
        if p.suffix in (".DM", ".HSX", ".psml") or p.name.endswith((".ion", ".ion.nc", ".ion.xml")) or "RHO" in p.name:
            p.unlink()
    (work / "record.json").write_text(json.dumps(rec, indent=1))
    with open(ROOT / "records.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{group:15s} {tag:18s} {label:18s} r{rep} rc={rc} wall={wall:8.2f} it={rec['iterations']} "
          f"E={rec['etot_ev']} F={rec['free_ev']} Ef={rec['fermi_ev']} M={rec['moment_mub']}", flush=True)
    return rec


FIXED4 = ["MaxSCFIterations 4", "SCF.MustConverge false"]
GROUPS = {}


def grp(f):
    GROUPS[f.__name__.replace("_", "-")] = f
    return f


@grp
def profile():
    for name in ("gpu-batched-256", "gpu-bj-64"):
        run("profile", name, "legacy", K20, 1, env_extra=dict(REMAG_SCF_STAGE_PROFILE="1"))


@grp
def lanes():
    names = [f"gpu-bj-{n}" for n in (16, 32, 128)] + [f"gpu-batched-{n}" for n in (32, 64, 128, 512, 1024)]
    for rep in (1, 2, 3):
        for name in names:
            run("lanes", name, "legacy", K20, rep)


@grp
def scaling_gpu():
    for rep in (1, 2, 3):
        for mesh in MESHES:
            for name in ("gpu-batched-256", "gpu-bj-64"):
                run("scaling", name, "legacy", mesh, rep)


@grp
def methods_gpu():
    for rep in (1, 2, 3):
        for kind in ("lf", "mc", "tet"):
            run("methods", "gpu-batched-256", kind, K20, rep)


@grp
def threads():
    for rep in (1, 2):
        for n in (1, 2, 4, 8, 16, 28):
            run("threads", f"host-{n}", "legacy", K20, rep, extra=FIXED4)
    run("threads", "pristine-serial", "legacy", K20, 1, extra=FIXED4)


@grp
def mpi_ranks():
    for n in (2, 4, 8, 16, 28):
        run("threads", f"pristine-mpi-{n}", "legacy", K20, 1, extra=FIXED4)


@grp
def scaling_cpu():
    for rep in (1, 2, 3):
        for mesh in MESHES:
            for name in (f"host-{CORES}", f"pristine-mpi-{CORES}"):
                run("scaling", name, "legacy", mesh, rep)


@grp
def methods_host():
    for kind in ("lf", "mc", "tet"):
        run("methods", f"host-{CORES}", kind, K20, 1)


@grp
def scaling_serial():
    for mesh in MESHES:
        for rep in ((1, 2, 3) if mesh[0] <= 10 else (1,)):
            run("scaling", "pristine-serial", "legacy", mesh, rep)


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    for g in sys.argv[1:]:
        print(f"== {g} {time.strftime('%H:%M:%S')}", flush=True)
        GROUPS[g]()
    print("BENCH_DONE", flush=True)
