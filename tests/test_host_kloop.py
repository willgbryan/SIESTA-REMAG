"""CPU check of the threaded spin-orbit k loop (native/host_kloop.cpp) against a numpy port of diag3k."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++17 compiler and LAPACK")
def test_host_kloop_matches_diag3k(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/"scripts/check_host_kloop.py"), "--output", str(tmp_path)],
                            capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    report = json.loads((tmp_path/"check.json").read_text())
    assert report["passed"]
    cases = report["cases"]
    # Every solver, kept and recomputed vectors, several thread counts, and the unfactored phases.
    assert {c["solver"] for c in cases} == {"divide", "mrrr", "qr"}
    assert {c["kept"] for c in cases} == {True, False} and {c["factored"] for c in cases} == {True, False}
    assert len({c["threads"] for c in cases}) >= 3
