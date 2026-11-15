"""Compile the real C++ owner with a CPU-only copy double; no GPU jobs."""
from pathlib import Path
import shutil
import subprocess
import pytest


@pytest.fixture(scope='module')
def owner(tmp_path_factory):
    compiler=shutil.which('c++')
    if compiler is None:pytest.skip('C++ compiler required')
    root=Path(__file__).resolve().parents[1];out=tmp_path_factory.mktemp('soc-owner')/'check'
    subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-I'+str(root/'tests/fixtures/soc_owner_cuda'),
        '-I'+str(root/'native'),str(root/'tests/fixtures/soc_owner_driver.cpp'),'-o',str(out)],
        check=True,capture_output=True,text=True)
    return out


@pytest.mark.parametrize('mode',['valid','poison','repeat','stale','scalar','unbound','shadow','failed',
    'unleased','leased','forces','complete','scf','copy_failure','nonfinite','missing','topology'])
def test_real_owner_copy_lifetime_and_accounting(owner,tmp_path,mode):
    result=subprocess.run([str(owner),mode],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
