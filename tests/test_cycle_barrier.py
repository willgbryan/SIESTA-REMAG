"""Compile and execute the actual diagnostic stop, without CUDA or a solver."""
import json
from pathlib import Path
import shutil
import subprocess
import pytest


def test_diagnostic_barrier_stops_before_any_cycle_and_never_returns(tmp_path):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('C++ compiler unavailable')
    source=Path(__file__).parents[1]/'native/probe_cycle_barrier.cpp'
    caller=tmp_path/'caller.cpp'
    caller.write_text('''#include <cstdio>
extern "C" int remag_cycle_execute(int,int*,int*);
int main(){int n=0,c=0;remag_cycle_execute(7,&n,&c);auto* f=std::fopen("should-not-exist","w");if(f)std::fclose(f);return 0;}
''')
    binary=tmp_path/'barrier-check'
    subprocess.run([compiler,'-std=c++17',str(source),str(caller),'-o',str(binary)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(binary)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==90
    assert not (tmp_path/'should-not-exist').exists()
    assert json.loads((tmp_path/'cycle-barrier.json').read_text())=={
        'format':'REMAG-CYCLE-BARRIER/1','geometry':7,'SCF_cycle_launched':False,'intentional_exit':90}


def test_barrier_receipt_failure_is_not_success(tmp_path):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('C++ compiler unavailable')
    source=Path(__file__).parents[1]/'native/probe_cycle_barrier.cpp'
    caller=tmp_path/'caller.cpp';caller.write_text('extern "C" int remag_cycle_execute(int,int*,int*);int main(){return remag_cycle_execute(1,nullptr,nullptr);}')
    binary=tmp_path/'barrier-check'
    subprocess.run([compiler,'-std=c++17',str(source),str(caller),'-o',str(binary)],check=True,capture_output=True,text=True)
    (tmp_path/'cycle-barrier.json').mkdir()
    assert subprocess.run([str(binary)],cwd=tmp_path).returncode==91
