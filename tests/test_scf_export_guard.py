"""Check the actual final-grid guard on host-resident residual history."""
from pathlib import Path
import shutil
import signal
import subprocess

import pytest


def test_residuals_survive_abort_and_success_is_silent(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    repo = Path(__file__).resolve().parents[1]
    exe = tmp_path / 'guard'
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I'+str(repo/'native'), str(repo/'tests/fixtures/scf_export_guard.cpp'),
                    '-o', str(exe)], check=True)
    success = subprocess.run([exe, 'accepted'], cwd=tmp_path, check=True, capture_output=True)
    assert not success.stderr
    failed = subprocess.run([exe, 'unconverged'], cwd=tmp_path, capture_output=True, text=True)
    assert failed.returncode == -signal.SIGABRT
    assert failed.stderr.splitlines() == [
        'REMAG rejected final grid: geometry=7 iterations=2 exhausted=1',
        'REMAG rejected SCF step 1: dDmax=0.5 dHmax_Ry=0.25 dEmax_Ry=0.125',
        'REMAG rejected SCF step 2: dDmax=0.75 dHmax_Ry=0.5 dEmax_Ry=0.25',
    ]
