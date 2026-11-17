import json
from pathlib import Path
import shutil
import subprocess
import pytest
from siesta_remag.research.native_setup_probe import INPUTS, validate_policy, barrier_reached, summarize


def policy():
    return dict(format='REMAG-NATIVE-SETUP-PROBE/1', timeout_seconds=180,
        reservation_cents=23, gpu_count=1, retries=0, cycle_launch=False,
        image='im-test', previous_app='ap-prior', held_app='ap-held', volume='saved-volume',
        files={name: 'a'*64 for name in INPUTS},
        **{key: 'b'*64 for key in ('helper_sha256', 'module_sha256', 'held_owner_sha256', 'driver_sha256')})


@pytest.mark.parametrize('key,value', [('timeout_seconds', 181), ('reservation_cents', 24),
    ('retries', 1), ('gpu_count', 2), ('cycle_launch', True), ('previous_app', ''),
    ('held_app', ''), ('files', {}), ('driver_sha256', '')])
def test_changed_diagnostic_is_rejected(key, value):
    p = policy(); p[key] = value
    with pytest.raises(ValueError): validate_policy(p)


def test_diagnostic_summary_never_promotes_to_native_acceptance(tmp_path):
    validate_policy(policy())
    marker = dict(format='REMAG-CYCLE-BARRIER/1', geometry=1, SCF_cycle_launched=False, intentional_exit=90)
    (tmp_path/'cycle-barrier.json').write_text(json.dumps(marker))
    (tmp_path/'graph-interpose.log').write_text('{"phase":"instantiate_end","status":0}\n')
    result = summarize(tmp_path, 90, False)
    assert result['barrier_reached'] and not result['native_runtime_repaired'] and not result['scientific_acceptance']
    assert not barrier_reached(-11, marker)
    assert not barrier_reached(90, marker, timed_out=True)
    assert not barrier_reached(90, dict(marker, SCF_cycle_launched=True))
    assert not barrier_reached(90, dict(marker, geometry=True))


def test_partial_crash_summary_is_retained(tmp_path):
    (tmp_path/'graph-interpose.log').write_text('{"phase":"instantiate_begin"}\npartial')
    r = summarize(tmp_path, -11, False)
    assert r['events'] == [{'phase': 'instantiate_begin'}]
    assert r['parse_errors'] and not r['barrier_reached']


@pytest.fixture
def interpose_binary(tmp_path):
    compiler = shutil.which('c++')
    if not compiler: pytest.skip('C++ compiler unavailable')
    (tmp_path/'cuda_runtime_api.h').write_text('''#pragma once
using cudaError_t=int; using cudaGraph_t=void*; using cudaGraphExec_t=void*;
constexpr unsigned cudaGraphDebugDotFlagsVerbose=1;
extern "C" int cudaGraphDebugDotPrint(void*,const char*,unsigned);
extern "C" int cudaRuntimeGetVersion(int*);
extern "C" int cudaDriverGetVersion(int*);
extern "C" int cudaGraphInstantiate(void**,void*,unsigned long long);
''')
    stub = tmp_path/'stub.cpp'
    stub.write_text('''#include "cuda_runtime_api.h"
#include <cstdlib>
#include <cstring>
static int calls=0;
extern "C" int cudaGraphDebugDotPrint(void*,const char*,unsigned){return 19;}
extern "C" int cudaRuntimeGetVersion(int* v){*v=12060;return 0;}
extern "C" int cudaDriverGetVersion(int* v){*v=13050;return 0;}
int real_call(void** out,void* graph,unsigned long long flags){++calls;if(graph!=(void*)0x5678||flags!=17)std::_Exit(99);*out=(void*)0x1234;return 27;}
extern "C" void* probe_lookup(void*,const char* name) noexcept {if(std::getenv("NO_SYMBOL"))return nullptr;if(std::strcmp(name,"cudaGraphInstantiate"))std::_Exit(98);return (void*)&real_call;}
int main(int argc,char** argv){int n=argc>1?std::atoi(argv[1]):1;for(int i=0;i<n;++i){void* out=nullptr;if(cudaGraphInstantiate(&out,(void*)0x5678,17)!=27||out!=(void*)0x1234)return 97;}return calls==n?0:96;}
''')
    binary = tmp_path/'interpose-test'
    source = Path(__file__).parents[1]/'native/probe_graph_interpose.cpp'
    subprocess.run([compiler, '-std=c++17', '-Ddlsym=probe_lookup', '-I'+str(tmp_path), str(source), str(stub), '-o', str(binary)], check=True, capture_output=True)
    return binary


def test_actual_interposer_forwards_once_without_altering_flags_or_result(interpose_binary, tmp_path):
    assert subprocess.run([str(interpose_binary)], cwd=tmp_path).returncode == 0
    rows = [json.loads(x) for x in (tmp_path/'graph-interpose.log').read_text().splitlines()]
    assert [r['phase'] for r in rows] == ['instantiate_begin', 'instantiate_end']
    assert [r['status'] for r in rows] == [-1, 27]
    assert all(r['dump_status'] == 19 and r['flags'] == 17 and r['driver'] == 13050 for r in rows)


def test_actual_interposer_bounds_number_of_dumps(interpose_binary, tmp_path):
    assert subprocess.run([str(interpose_binary), '65'], cwd=tmp_path).returncode == 92
    assert len((tmp_path/'graph-interpose.log').read_text().splitlines()) == 128


def test_actual_interposer_fails_closed_without_forward_symbol(interpose_binary, tmp_path):
    import os
    assert subprocess.run([str(interpose_binary)], cwd=tmp_path, env=dict(os.environ, NO_SYMBOL='1')).returncode == 93
    assert not (tmp_path/'graph-interpose.log').exists()
