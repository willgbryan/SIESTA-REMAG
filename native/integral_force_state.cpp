// SPDX-License-Identifier: GPL-3.0-or-later
#include "integral_force_state.hpp"
#include "final_grid_state.hpp"
#include "pair_geometry_state.hpp"
#include "pair_force.hpp"
#include "nonlocal_force_state.hpp"
#include "nonlocal_h0_state.hpp"
#include "two_center_state.hpp"
#include "orbital_basis_state.hpp"
#include "atomic_basis_state.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
namespace {
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Cycle {
    uint64_t epoch=0;int atoms=0,components=0;bool started=false,finished=false,terminal=false;
    cudaStream_t stream=nullptr;std::vector<void*> memory;
    std::vector<cudaGraph_t> graphs;std::vector<cudaGraphExec_t> executables;
    ~Cycle(){
        if(stream)cudaStreamSynchronize(stream);
        for(auto e:executables)cudaGraphExecDestroy(e);
        for(auto g:graphs)cudaGraphDestroy(g);
        for(auto p:memory)cudaFree(p);
    }
    template<class T>T* allocate(size_t n){T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),std::max(size_t(1),n)*sizeof(T)));memory.push_back(p);return p;}
    template<class F>int captured(int expected,F fn){
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        try{fn();}catch(...){cudaGraph_t failed=nullptr;cudaStreamEndCapture(stream,&failed);if(failed)cudaGraphDestroy(failed);throw;}
        cudaGraph_t graph=nullptr;check(cudaStreamEndCapture(stream,&graph));graphs.push_back(graph);
        size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));require(count==size_t(expected),"Integral force graph size differs");
        std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
        for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));require(type==cudaGraphNodeTypeKernel,"Forbidden integral force graph node");}
        cudaGraphExec_t executable=nullptr;check(cudaGraphInstantiate(&executable,graph,0));executables.push_back(executable);
        check(cudaGraphLaunch(executable,stream));return int(count);
    }
};
std::unique_ptr<Cycle> cycle;
void run_pair(int consumer,const double* initial_force,const double* initial_stress){
    auto& c=*cycle;const auto fields=remag::borrow_final_pair_state(c.epoch,consumer);
    if(consumer==1)c.components=fields.h_spins;
    else require(c.components==fields.h_spins,"Integral spin layout changed");
    const auto pool=remag::final_force_accumulator(c.epoch,c.atoms,1,1,initial_force,initial_stress);c.stream=pool.stream;
    const auto geometry=remag::borrow_integral_pair_geometry(consumer,c.atoms,fields);
    const auto table=remag::borrow_integral_two_center(consumer);const int n=geometry.queries;
    auto integrals=c.allocate<double>(4*size_t(n));const remag::PairForceProblem problem{fields.volume,consumer,0};auto device_problem=c.allocate<remag::PairForceProblem>(1);
    check(cudaMemcpyAsync(device_problem,&problem,sizeof(problem),cudaMemcpyHostToDevice,c.stream));
    remag::transfer(remag::Transfer::grid_flow_geometry,true,sizeof(problem));
    // The only host input is immutable geometry/configuration. Keep the stack
    // problem alive through its initial staging; there is no electronic readback.
    check(cudaStreamSynchronize(c.stream));
    remag::PairForceView view{n,c.atoms,fields.shape.nuo,fields.shape.no,fields.shape.nnz,std::min(2,fields.h_spins),
        geometry.row,geometry.column,geometry.atoms,geometry.num,geometry.ptr,geometry.list,integrals,geometry.xyz};
    const int nodes=c.captured(n?2:1,[&]{
        if(n)check(remag::evaluate_two_center_device(table,n,geometry.pairs,geometry.xyz,integrals,pool.status,c.stream));
        check(remag::add_pair_force_device(view,device_problem,consumer==1?fields.density:fields.energy_density,pool.energy,pool.force,pool.stress,pool.status,c.stream));
    });
    remag::record_final_integral_force(c.epoch,{consumer,n,nodes,sizeof(problem),std::max(size_t(1),4*size_t(n))*8+sizeof(problem)});
    remag::complete_pair_geometry(consumer);remag::complete_final_pair_shadow(c.epoch,consumer);
}
void run_nonlocal(const double* force,const double* stress){
    auto& c=*cycle;const int consumer=c.components==8?3:2;
    const auto fields=remag::borrow_final_pair_state(c.epoch,consumer);
    const auto pool=remag::final_force_accumulator(c.epoch,c.atoms,1,1,force,stress);
    require(pool.stream==c.stream,"Integral accumulator stream changed");
    const auto geometry=remag::borrow_integral_nonlocal(consumer,c.atoms,fields,c.stream);
    const int n=geometry.view.queries;const bool retained=remag::nonlocal_h0_active();
    remag::TwoCenterView table{};if(!retained)table=remag::borrow_integral_two_center(consumer);
    auto generated=retained?nullptr:c.allocate<double>(4*size_t(n));
    const double* integrals=retained?remag::borrow_nonlocal_h0_integrals(c.epoch,consumer,n):generated;
    auto weights=c.allocate<double>(fields.shape.nnz);
    const int nodes=c.captured(retained?2:(n?3:2),[&]{
        if(n&&!retained)check(remag::evaluate_two_center_device(table,n,geometry.pairs,geometry.view.xyz,generated,pool.status,c.stream));
        remag::apply_integral_nonlocal(integrals,weights,pool);
    });
    remag::record_final_integral_force(c.epoch,{consumer,n,nodes,0,8*((retained?0:std::max(size_t(1),4*size_t(n)))+size_t(fields.shape.nnz)),retained});
    remag::complete_final_pair_shadow(c.epoch,consumer);
}
int failed(const std::exception& e){if(cycle)cycle->terminal=true;std::fprintf(stderr,"REMAG integral forces: %s\n",e.what());return -290;}
}
extern "C" int remag_integral_force_enabled(){return remag::integral_force_enabled();}
extern "C" int remag_integral_force_begin(int atoms,const double* force,const double* stress){
    try{
        require(remag::integral_force_enabled()&&!remag::in_scf()&&!cycle&&atoms>0&&force&&stress&&
            remag::orbital_basis_enabled()&&remag::atomic_basis_device_enabled()&&!remag_two_center_shadow_enabled(),"Invalid production integral force mode/start");
        cycle=std::make_unique<Cycle>();cycle->epoch=remag::geometry_epoch();cycle->atoms=atoms;
        run_pair(1,force,stress);run_nonlocal(force,stress);cycle->started=true;return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_integral_hartree_stress(){
    try{require(cycle&&cycle->started&&!cycle->finished&&!cycle->terminal,"Hartree stress outside integral force cycle");remag::add_final_hartree_stress(cycle->epoch);return 0;}
    catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_integral_force_finish(int atoms,double* force,double* stress,double* energy){
    try{
        require(cycle&&cycle->started&&!cycle->finished&&!cycle->terminal&&atoms==cycle->atoms&&force&&stress&&energy,"Invalid production integral force finish");
        auto& c=*cycle;run_pair(4,force,stress);c.finished=true;
        const auto result=remag::collect_final_forces(c.epoch,atoms,force,stress);
        remag::finish_integral_grid_force(c.epoch,result.status);
        remag::finish_integral_nonlocal(result.status);remag::finish_integral_pair_geometry(c.epoch,result.status);
        static int sequence=0;char name[80];std::snprintf(name,sizeof(name),"remag-integral-force-%04d.json",++sequence);
        FILE* f=std::fopen(name,"w");require(f,"Cannot retain integral force publication evidence");
        std::fprintf(f,"{\"scope\":\"authoritative GPU kinetic/nonlocal/grid/overlap forces; initial ionic terms remain on CPU\",\"epoch\":%llu,\"atoms\":%d,\"status\":%d,\"export_bytes\":%zu,\"intermediate_export_bytes\":0,\"derivative_reference_upload_bytes\":0,\"consumers\":[",
            static_cast<unsigned long long>(c.epoch),atoms,result.status,result.exported_bytes);
        const auto& records=remag::final_integral_force_records(c.epoch);
        for(size_t i=0;i<records.size();i++){const auto& r=records[i];std::fprintf(f,"%s{\"consumer\":%d,\"queries\":%d,\"graph_nodes\":%d,\"setup_bytes\":%zu,\"workspace_bytes\":%zu",i?",":"",r.consumer,r.queries,r.graph_nodes,r.setup_bytes,r.workspace_bytes);if(r.retained_integrals)std::fputs(",\"retained_integrals\":true",f);std::fputc('}',f);}
        std::fputs("]",f);const char* names[]={"energy","force","stress"};const std::vector<double>* arrays[]={&result.energy,&result.force,&result.stress};
        for(int i=0;i<3;i++){std::fprintf(f,",\"%s\":[",names[i]);for(size_t j=0;j<arrays[i]->size();j++)std::fprintf(f,"%s%.17g",j?",":"",(*arrays[i])[j]);std::fputc(']',f);}
        std::fputs("}\n",f);std::fclose(f);
        remag::complete_final_force_publication(c.epoch);
        std::copy(result.energy.begin(),result.energy.end(),energy);std::copy(result.force.begin(),result.force.end(),force);std::copy(result.stress.begin(),result.stress.end(),stress);
        cycle.reset();return 0;
    }catch(const std::exception& e){return failed(e);}
}
