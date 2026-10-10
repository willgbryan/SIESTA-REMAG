#!/usr/bin/env python3
"""Build an isolated CUPTI 12.8 observer without replacing CUDA 12.6 math libraries."""
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


package = distribution("nvidia-cuda-cupti-cu12")
assert package.version == "12.8.90", package.version
source = Path(package.locate_file("nvidia/cuda_cupti"))
target = Path("/opt/remag-cupti128")
target.mkdir()
library = source / "lib/libcupti.so.12"
shutil.copy2(library, target / library.name)
command = ["g++", "-std=c++17", "-O2", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror",
           "-I" + str(source / "include"), "-I/usr/local/cuda/include",
           "/opt/remag/native/activity.cpp", "-o", str(target / "libremag_activity.so"),
           "-L" + str(target), "-Wl,-rpath," + str(target), "-l:libcupti.so.12",
           "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart"]
subprocess.run(command, check=True)
(target / "build.json").write_text(json.dumps(dict(
    scope="Diagnostic observer only; production CUDA/cuBLAS/cuSOLVER remain 12.6.3",
    distribution=package.metadata["Name"], version=package.version, command=command,
    source_sha256=sha(Path("/opt/remag/native/activity.cpp")),
    build_script_sha256=sha(Path(__file__)), library_sha256=sha(library),
    observer_sha256=sha(target / "libremag_activity.so"),
    headers_sha256={p.name: sha(p) for p in sorted((source / "include").glob("*.h"))}), indent=2) + "\n")
