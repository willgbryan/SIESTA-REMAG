"""Exercise actual geometry-owner guards without executing device algorithms.

The fixtures seed already prepared owners and vary publication/lease state.
CUDA declarations permit host compilation; unused numerical code is discarded.
This checks lifetime transitions, not CUDA execution or scientific equivalence.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
OWNERS = ('pair_geometry_state', 'nonlocal_force_state', 'nonlocal_h0_state', 'initial_mesh_production')
RECEIPTS = ('remag-pair-geometry.json', 'remag-nonlocal-geometry.json',
            'remag-nonlocal-h0.json', 'remag-initial-mesh-production.json')

CUDA = r'''
#pragma once
#include <cstddef>
#include <stdexcept>
using cudaError_t=int;using cudaStream_t=void*;using cudaGraph_t=void*;
using cudaGraphExec_t=void*;using cudaGraphNode_t=void*;using cudaGraphNodeType=int;
constexpr int cudaSuccess=0,cudaStreamNonBlocking=1,cudaMemcpyHostToDevice=1,
    cudaMemcpyDeviceToHost=2,cudaMemcpyDeviceToDevice=3,
    cudaStreamCaptureModeGlobal=0,cudaGraphNodeTypeKernel=0;
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* p,unsigned){*p=nullptr;return 0;}
inline cudaError_t cudaStreamSynchronize(cudaStream_t){return 0;}
inline cudaError_t cudaStreamDestroy(cudaStream_t){return 0;}
inline cudaError_t cudaFree(void*){throw std::runtime_error("Unexpected device storage");}
inline const char* cudaGetErrorString(cudaError_t){return "Host lifetime test";}
cudaError_t cudaMalloc(void**,size_t);
cudaError_t cudaMemcpy(void*,const void*,size_t,int);
cudaError_t cudaMemcpyAsync(void*,const void*,size_t,int,cudaStream_t);
cudaError_t cudaMemsetAsync(void*,int,size_t,cudaStream_t);
cudaError_t cudaStreamBeginCapture(cudaStream_t,int);
inline cudaError_t cudaStreamEndCapture(cudaStream_t,cudaGraph_t*){throw std::runtime_error("Unexpected capture");}
inline cudaError_t cudaGraphDestroy(cudaGraph_t){throw std::runtime_error("Unexpected graph");}
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t){throw std::runtime_error("Unexpected graph executable");}
cudaError_t cudaGraphGetNodes(cudaGraph_t,cudaGraphNode_t*,size_t*);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t,cudaGraphNodeType*);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*,cudaGraph_t,unsigned long long);
cudaError_t cudaGraphLaunch(cudaGraphExec_t,cudaStream_t);
'''

CHECK = r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <string>
#include OWNER_SOURCE
static uint64_t test_epoch=7;
static bool test_accepted=false,test_energy=true,test_retained=true,test_scf=false;
extern "C" int remag_orientation_retain_geometry(){return test_retained;}
extern "C" int remag_two_center_active(){return 1;}
namespace remag {
uint64_t geometry_epoch(){return test_epoch;}
bool in_scf(){return test_scf;}
bool energy_only_enabled(){return test_energy;}
bool integral_force_enabled(){return true;}
bool pair_force_active(){return true;}
bool final_electronic_state_complete(uint64_t epoch){return test_accepted&&epoch==test_epoch&&!test_scf;}
void validate_grid_sparse_topology(int,int,int,const int*,const int*,const int*) {
    std::abort(); // A rejection must occur in the owner, before this consumer.
}
GridContractionView grid_contraction_view(){std::abort();}
#if OWNER == 3
HartreePlan::~HartreePlan()=default;
cudaError_t bind_initial_harris_device(const HartreeResult*,GridPotentialProblem*,int*,cudaStream_t){std::abort();}
#endif
}
template<class F>void rejects(F action){bool caught=false;try{action();}catch(const std::exception&){caught=true;}assert(caught);}
void bridge(int status){if(status)throw std::runtime_error("Rejected reuse");}

#if OWNER == 0
void setup(){
    Record r;r.epoch=test_epoch;r.staged=123;r.allocated=456;
    records().rows.push_back(r);geometry=std::make_unique<Geometry>();
    geometry->ready=true;geometry->bound=true;geometry->integral=true;
}
auto& row(){return records().rows.at(geometry->record);}
void complete(uint64_t epoch){remag::complete_pair_geometry_energy_only(epoch);}
int reuse(){return remag_reuse_pair_geometry();}
void force_complete(){row().complete=true;row().consumed=5;}
void force_access(){remag::begin_pair_geometry(1);}
void partial_rejected(){geometry->active=1;rejects([]{complete(test_epoch);});geometry->active=0;}
void no_forces(){assert(!row().complete&&row().consumed==0);}
#elif OWNER == 1
void setup(){
    Record r;r.epoch=test_epoch;r.integral=true;r.geometry_bytes=123;r.allocated=456;
    records().rows.push_back(r);geometry=std::make_unique<Geometry>();
    geometry->ready=true;geometry->bound=true;
}
auto& row(){return records().rows.at(geometry->record);}
void complete(uint64_t epoch){remag::complete_nonlocal_geometry_energy_only(epoch);}
int reuse(){return remag_reuse_nonlocal_geometry();}
void force_complete(){row().complete=true;row().passed=true;row().status=0;geometry->mode=2;geometry->staged=true;geometry->applied=true;}
void force_access(){remag::FinalGridStateView fields{};fields.epoch=test_epoch;
    remag::borrow_integral_nonlocal(0,0,fields,reinterpret_cast<void*>(1));}
void partial_rejected(){geometry->mode=2;rejects([]{complete(test_epoch);});geometry->mode=0;}
void no_forces(){assert(!row().complete&&!row().passed&&row().status==-1&&geometry->mode==0);}
#elif OWNER == 2
using namespace remag;
void setup(){
    setenv("REMAG_NONLOCAL_H0","1",1);
    Record r;r.epoch=test_epoch;r.production=true;r.bound=true;r.passed=true;r.status=0;
    r.scf_borrowed=true;r.scf_released=true;r.staged=123;r.allocated=456;
    evidence().rows.push_back(r);state=std::make_unique<State>();
    state->integrals=reinterpret_cast<double*>(1);
}
auto& row(){return evidence().rows.at(state->record);}
void complete(uint64_t epoch){complete_nonlocal_h0_energy_only(epoch);}
int reuse(){return remag_reuse_nonlocal_h0();}
void force_complete(){row().complete=true;row().force_borrowed=true;}
void force_access(){borrow_nonlocal_h0_integrals(test_epoch,0,0);}
void partial_rejected(){row().scf_released=false;rejects([]{complete(test_epoch);});row().scf_released=true;}
void no_forces(){assert(!row().complete&&!row().force_borrowed);}
#elif OWNER == 3
void setup(){
    setenv("REMAG_INITIAL_MESH","1",1);
    Record r;r.epoch=test_epoch;r.launched=true;r.phi_bound=true;r.density_borrowed=true;
    r.density_released=true;r.passed=true;r.converged=1;r.cycle_status=0;r.producer_status=0;
    r.staged=123;r.allocated=456;records().rows.push_back(r);
    owner=std::make_unique<Owner>();owner->epoch=test_epoch;
}
auto& row(){return records().rows.at(owner->row);}
void complete(uint64_t epoch){remag::complete_initial_mesh_energy_only(epoch);}
int reuse(){return remag_reuse_initial_mesh();}
void force_complete(){row().forces_complete=true;}
void force_access(){remag::GridPotentialProblem p{},device{};
    remag::borrow_initial_mesh_density(test_epoch,p,&device,reinterpret_cast<void*>(1));}
void partial_rejected(){row().density_released=false;rejects([]{complete(test_epoch);});row().density_released=true;}
void no_forces(){assert(!row().forces_complete);}
#endif

int main(int argc,char** argv){
    assert(argc==2);setup();const bool force=std::string(argv[1])=="force";
    if(force){test_energy=false;force_complete();}
    else {
        rejects([]{complete(test_epoch);}); // SCF may be done, publication is not.
        rejects([]{bridge(reuse());});
        test_accepted=true;
        test_energy=false;rejects([]{complete(test_epoch);});test_energy=true;
        test_retained=false;rejects([]{complete(test_epoch);});test_retained=true;
        test_scf=true;rejects([]{complete(test_epoch);});test_scf=false;
        rejects([]{complete(test_epoch+1);});
        test_epoch++;rejects([]{complete(test_epoch);});test_epoch--;
        partial_rejected();
        complete(test_epoch);no_forces();
        assert(row().energy_only_complete);
        rejects([]{complete(test_epoch);});rejects([]{force_access();});
    }
    rejects([]{bridge(reuse());}); // Even completed owners need next epoch.
    test_epoch+=2;rejects([]{bridge(reuse());});test_epoch--;
    test_retained=false;rejects([]{bridge(reuse());});test_retained=true;
    bridge(reuse());
    assert(row().epoch==test_epoch&&!row().energy_only_complete&&row().allocated==0&&row().staged==0);
    no_forces();test_accepted=false;
    rejects([]{complete(test_epoch);});rejects([]{bridge(reuse());});
    test_epoch++;rejects([]{bridge(reuse());}); // No inherited acceptance.
}
'''


@pytest.fixture(scope='module', params=range(len(OWNERS)), ids=OWNERS)
def executable(request, tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler needed for geometry lifetime checks')
    folder = tmp_path_factory.mktemp('geometry_lifetime')
    (folder/'cuda_runtime_api.h').write_text(CUDA)
    (folder/'cufft.h').write_text('using cufftHandle=int; struct cufftDoubleComplex {double x,y;};\n')
    source = folder/'check.cpp';source.write_text(CHECK)
    target = folder/'check'
    linker = '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections'
    result = subprocess.run([compiler, '-std=c++17', '-O1', '-ffunction-sections', '-fdata-sections',
                            '-Wall', '-Wextra', '-Werror', linker, '-I'+str(folder), '-I'+str(ROOT/'native'),
                            '-DOWNER='+str(request.param), '-DOWNER_SOURCE="'+OWNERS[request.param]+'.cpp"',
                            str(source), '-o', str(target)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return target, RECEIPTS[request.param]


@pytest.mark.parametrize('mode', ['energy', 'force'])
def test_publication_then_reuse_preserves_distinct_force_state(executable, mode, tmp_path):
    program, receipt = executable
    result = subprocess.run([str(program), mode], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = json.loads((tmp_path/receipt).read_text())['cycles']
    assert len(rows) == 2 and [r['epoch'] for r in rows] == [7, 8]
    assert rows[0]['energy_only_complete'] is (mode == 'energy')
    assert rows[0].get('complete', rows[0].get('forces_complete')) is (mode == 'force')
    assert rows[1]['energy_only_complete'] is False
    assert rows[1].get('complete', rows[1].get('forces_complete')) is False
