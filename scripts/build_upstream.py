#!/usr/bin/env python3
"""Build a pinned, serial FP64 reference; all products stay in the workspace."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(*args):
    subprocess.run([str(a) for a in args], check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--source", type=Path, default=ROOT / "upstream/siesta")
    parser.add_argument("--build", type=Path, default=ROOT / "build/upstream")
    args = parser.parse_args()
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    source, build = args.source.resolve(), args.build.resolve()
    if not source.exists():
        run("git", "clone", "--depth", "1", "--branch", lock["tag"], lock["url"], source)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != lock["commit"]:
        raise SystemExit(f"Upstream commit mismatch: {commit}")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise SystemExit("Reference source has tracked modifications; refusing an unverified baseline")
    run("git", "-C", source, "submodule", "update", "--init", "--recursive", *lock["submodules"])
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DFortran_FLAGS_RELEASE=-O3", "-DC_FLAGS_RELEASE=-O3"]
    flags += [f"-DSIESTA_WITH_{name}=OFF" for name in
              ("MPI", "ELSI", "DFTD3", "FLOOK", "NETCDF", "OPENMP", "FFTW", "HOST_OPTIMIZATION")]
    openblas = Path("/opt/homebrew/opt/openblas/lib/libopenblas.dylib")
    if openblas.exists():
        flags += [f"-DBLAS_LIBRARY={openblas}", f"-DLAPACK_LIBRARY={openblas}"]
    run("cmake", "-S", source, "-B", build, *flags)
    run("cmake", "--build", build, "--target", "SIESTA.siesta", "--parallel", args.jobs)
    (build / "provenance.json").write_text(json.dumps({**lock, "cmake_flags": flags}, indent=2) + "\n")

if __name__ == "__main__":
    main()
