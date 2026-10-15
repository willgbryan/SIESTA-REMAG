"""CPU checks of tetrahedron occupations: exact single-tetrahedron integrals, grids, QE's code and QE's runs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++17 compiler")
def test_tetrahedra_cpu_checks(tmp_path):
    script = ROOT/"scripts/check_tetrahedra.py"
    result = subprocess.run([sys.executable, str(script), "--output", str(tmp_path)], capture_output=True, text=True,
                            timeout=1800)
    report = json.loads((tmp_path/"check.json").read_text())
    failed = [name for name, part in report.items() if isinstance(part, dict) and not part["passed"]]
    assert result.returncode == 0 and report["passed"], (failed, result.stderr[-2000:])
    # QE's own runs must be present, and cover every method and spin layout.
    runs = report["qe_runs"]["runs"]
    assert {(r["system"], r["method"]) for r in runs} >= {(s, m) for s in ("al", "co", "co-soc")
                                                          for m in ("linear", "bloechl", "optimized")}
    assert max(r["max_occupation_difference_at_qe_fermi"] for r in runs) < 1.e-10
    rows = report["convergence"]["rows"]
    assert abs(rows[-1]["optimized"]["band_energy_error"]) < abs(rows[-1]["linear"]["band_energy_error"])
