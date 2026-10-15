"""CPU qualification of site spin constraints: exact packing, host ABI and outer search."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++17 compiler")
def test_spin_constraint_cpu_oracle(tmp_path):
    pytest.importorskip("scipy")
    script = Path(__file__).resolve().parents[1]/"scripts/check_spin_constraint_math.py"
    result = subprocess.run([sys.executable, str(script), "--output", str(tmp_path)], capture_output=True, text=True,
                            timeout=1800)
    report = json.loads((tmp_path/"check.json").read_text())
    failed = [group for group, records in report.items() if isinstance(records, list)
              for r in records if isinstance(r, dict) and not r.get("passed", True)]
    assert result.returncode == 0 and report["passed"], (failed, result.stderr[-2000:])
    # Exact dyadic fixtures cover both spinor storages, with and without images.
    assert {(r["spins"], r["images"]) for r in report["exact"] if "spins" in r} == {(8, False), (8, True), (4, False), (4, True)}
    assert any(r.get("case") == "penalized_field" for r in report["exact"])
    assert report["soft_mode"]["passed"]
    modes = {(r["case"], r["mode"]) for r in report["host_abi_searches"]}
    assert ("noncollinear", "steepest_golden") in modes and ("cluster", "bfgs_secant_coarse") in modes
