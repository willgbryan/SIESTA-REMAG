// SPDX-License-Identifier: GPL-3.0-or-later
// Material shadow for exact upstream radial derivatives and retained fields.
#include "atomic_grid_force.hpp"
#include "atomic_basis_state.hpp"
#include "final_grid_state.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
void require(bool v,const char* message) {if(!v) throw std::runtime_error(message);}
struct Shadow {
    remag::FinalGridStateView final{};
    remag::AtomicGridForceProblem problem{};
    int points=0,atoms=0;
    bool authoritative=false,basis_shadow=false,basis_device=false;
    uint64_t epoch=0;
    std::vector<int> point,atom;
    std::vector<double> gradient,displacement,force,stress;
};
std::unique_ptr<Shadow> shadow;
struct Device {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> memory;size_t staged=0,exported=0;bool owns_stream;
    explicit Device(cudaStream_t borrowed=nullptr):stream(borrowed),owns_stream(!borrowed) {
        if(owns_stream) check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    }
    ~Device() {
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);
        if(graph) cudaGraphDestroy(graph);
        for(void* p:memory) cudaFree(p);
        if(stream&&owns_stream) cudaStreamDestroy(stream);
    }
    template<class T> T* upload(size_t n,const T* source) {
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),(n?n:1)*sizeof(T)));memory.push_back(p);
        if(n) {check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyHostToDevice,stream));
            staged+=n*sizeof(T);remag::transfer(remag::Transfer::grid_flow_geometry,true,n*sizeof(T));}
        return p;
    }
    template<class T> void download(T* host,const T* device,size_t n) {
        check(cudaMemcpy(host,device,n*sizeof(T),cudaMemcpyDeviceToHost));exported+=n*sizeof(T);
        remag::transfer(remag::Transfer::grid_output,false,n*sizeof(T));
    }
};
int failed(const std::exception& e) {
    std::fprintf(stderr,"REMAG atomic-grid force shadow: %s\n",e.what());shadow.reset();return -231;
}
void array(FILE* file,const char* name,const double* values,size_t count) {
    std::fprintf(file,"\"%s\":[",name);
    for(size_t i=0;i<count;i++) {
        if(i) std::fputc(',',file);
        if(std::isfinite(values[i])) std::fprintf(file,"%.17g",values[i]);else std::fputs("null",file);
    }
    std::fputc(']',file);
}
}
extern "C" int remag_atomic_force_shadow_enabled() {
    const char* v=std::getenv("REMAG_ATOMIC_FORCE_SHADOW");return v&&std::strcmp(v,"1")==0;
}
extern "C" int remag_atomic_grid_force_enabled() {return remag::atomic_grid_force_enabled();}
extern "C" int remag_atomic_force_active() {return remag_atomic_force_shadow_enabled()||remag_atomic_grid_force_enabled();}
extern "C" int remag_atomic_force_begin(int kind,int points,int atoms,int nsd,const int* mesh,int nsm,
        double dvol,double volume,int forces,int stress,const double* force,const double* tensor) {
    try {
        require(remag_atomic_force_active()&&!remag::in_scf()&&!shadow&&
            (kind==1||kind==2)&&points>0&&atoms>0&&mesh&&force&&tensor,"Invalid atomic-force lifetime/configuration");
        require(!(remag_atomic_force_shadow_enabled()&&remag_atomic_grid_force_enabled()),"Choose atomic force production or shadow mode");
        auto s=std::make_unique<Shadow>();s->epoch=remag::geometry_epoch();
        s->final=remag::borrow_final_grid_state(s->epoch,kind);const auto& f=s->final;
        require(points==f.shape.np*f.shape.nsp&&atoms<=f.shape.nuo&&std::equal(mesh,mesh+3,f.mesh)&&
            nsm==f.nsm&&dvol==f.dvol&&volume==f.volume&&
            nsd==(kind==1?std::min(2,f.energy_spins):1),"Atomic-force fields differ in geometry/configuration");
        s->points=points;s->atoms=atoms;s->problem={dvol,volume,kind,nsd,forces,stress};
        s->force.assign(force,force+3*atoms);s->stress.assign(tensor,tensor+9);s->authoritative=remag::atomic_grid_force_enabled();
        if(s->authoritative) remag::final_force_accumulator(s->epoch,atoms,forces,stress,force,tensor);
        s->basis_shadow=remag::atomic_basis_shadow_enabled();s->basis_device=remag::atomic_basis_device_enabled();
        if(s->basis_shadow||s->basis_device)remag::begin_atomic_basis(s->epoch,kind,points,f.shape.nsp,atoms,mesh,nsm,f.partial_core);
        shadow=std::move(s);return 0;
    } catch(const std::exception& e) {return failed(e);}
}
extern "C" int remag_atomic_force_point(int point,int atom,const double* gradient,const double* displacement) {
    try {
        require(shadow&&!remag::in_scf()&&shadow->epoch==remag::geometry_epoch()&&gradient&&displacement,"Invalid atomic-force derivative lifetime");
        auto& s=*shadow;require(!s.basis_device,"CPU derivative callback in device mode");require(point>0&&point<=s.points&&atom>0&&atom<=s.atoms,"Invalid atomic-force point/atom index");
        require(s.point.size()<(size_t(4)<<30)/56,"Atomic-force derivative geometry exceeds 4 GiB limit");
        // Exact inverse of the pinned reord permutation. No electronic data
        // cross the host boundary: only immutable geometry indices change.
        if(s.problem.kind==2) {
            const auto& f=s.final;const int sub=(point-1)%f.shape.nsp,cluster=(point-1)/f.shape.nsp,n=f.nsm;
            const int nx=f.mesh[0]/n,ny=f.mesh[1]/n;
            const int x=(cluster%nx)*n+sub%n,y=((cluster/nx)%ny)*n+(sub/n)%n,z=(cluster/(nx*ny))*n+sub/(n*n);
            point=1+x+f.mesh[0]*(y+f.mesh[1]*z);
            require(point>0&&point<=s.points,"Invalid sequential deformation-density index");
        }
        s.point.push_back(point);s.atom.push_back(atom);
        s.gradient.insert(s.gradient.end(),gradient,gradient+3);s.displacement.insert(s.displacement.end(),displacement,displacement+3);
        return 0;
    } catch(const std::exception& e) {return failed(e);}
}
extern "C" int remag_atomic_force_finish(const double* reference_force,const double* reference_stress) {
    try {
        require(shadow&&!remag::in_scf()&&shadow->epoch==remag::geometry_epoch()&&reference_force&&reference_stress,"Invalid atomic-force completion");
        const auto& s=*shadow;
        require(s.basis_shadow==remag::atomic_basis_shadow_enabled()&&s.basis_device==remag::atomic_basis_device_enabled(),"Atomic basis shadow mode changed during force use");
        const auto accumulator=s.authoritative?remag::final_force_accumulator(s.epoch,s.atoms,s.problem.forces,s.problem.stress,
            reference_force,reference_stress):remag::FinalForceAccumulatorView{};
        Device device(accumulator.stream);int zero=0;
        remag::AtomicGridForceView v{};
        if(s.basis_device){
            require(s.point.empty()&&s.atom.empty()&&s.gradient.empty()&&s.displacement.empty(),"Unexpected CPU atomic derivatives");
            v=remag::prepare_atomic_basis(s.epoch,s.problem.kind,0,device.stream);
        }else v={s.points,int(s.point.size()),s.atoms,device.upload(s.point.size(),s.point.data()),
            device.upload(s.atom.size(),s.atom.data()),device.upload(s.gradient.size(),s.gradient.data()),device.upload(s.displacement.size(),s.displacement.data())};
        auto p=device.upload(1,&s.problem);
        auto f=s.authoritative?accumulator.force:device.upload(s.force.size(),s.force.data());
        auto t=s.authoritative?accumulator.stress:device.upload(9,s.stress.data());
        auto status=s.authoritative?accumulator.status:device.upload(1,&zero);check(cudaStreamSynchronize(device.stream));
        if(s.basis_shadow)remag::prepare_atomic_basis(s.epoch,s.problem.kind,s.point.size(),device.stream);
        check(cudaStreamBeginCapture(device.stream,cudaStreamCaptureModeGlobal));
        if(s.basis_shadow||s.basis_device)remag::enqueue_atomic_basis(s.epoch,s.problem.kind,status,device.stream);
        check(remag::add_atomic_grid_force_device(v,p,s.final.potential,s.final.auxiliary,s.final.deformation,f,t,status,device.stream));
        check(cudaStreamEndCapture(device.stream,&device.graph));size_t nodes=0;
        check(cudaGraphGetNodes(device.graph,nullptr,&nodes));require(nodes==(s.basis_shadow?3u:s.basis_device?2u:1u),"Unexpected atomic-force graph");
        std::vector<cudaGraphNode_t> graph_nodes(nodes);check(cudaGraphGetNodes(device.graph,graph_nodes.data(),&nodes));
        for(auto node:graph_nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));require(type==cudaGraphNodeTypeKernel,"Forbidden atomic-force graph node");}
        check(cudaGraphInstantiate(&device.executable,device.graph,0));check(cudaGraphLaunch(device.executable,device.stream));
        check(cudaStreamSynchronize(device.stream));
        if(s.authoritative) {
            // No force/stress/status readback, no per-term file publication.
            // A sticky device error suppresses subsequent kernels and is
            // checked by dfscf before any final result becomes authoritative.
            remag::record_final_atomic_force(s.epoch,{s.problem.kind,s.points,v.entries,s.problem.diagonal_spins,int(nodes),device.staged,
                56*s.point.size(),56*s.point.size()});
            remag::complete_final_grid_force(s.epoch,s.problem.kind);shadow.reset();return 0;
        }
        int code=-1;device.download(&code,status,1);
        std::vector<double> force(s.force.size()),stress(9);device.download(force.data(),f,force.size());device.download(stress.data(),t,9);
        bool finite=true;double ferr=0.,terr=0.;
        for(size_t i=0;i<force.size();i++) {finite&=std::isfinite(force[i])&&std::isfinite(reference_force[i]);ferr=std::max(ferr,std::abs(force[i]-reference_force[i]));}
        for(int i=0;i<9;i++) {finite&=std::isfinite(stress[i])&&std::isfinite(reference_stress[i]);terr=std::max(terr,std::abs(stress[i]-reference_stress[i]));}
        const bool passed=code==0&&finite&&ferr<1.e-10&&terr<1.e-12;
        static int sequence=0;char filename[96];std::snprintf(filename,sizeof(filename),"remag-atomic-force-shadow-%04d.json",++sequence);
        FILE* file=std::fopen(filename,"w");require(file,"Cannot retain atomic-force evidence");
        std::fprintf(file,"{\"scope\":\"Core/neutral contraction shadow; CPU result preserved; host-staged exact radial derivatives\",\"kind\":%d,\"epoch\":%llu,\"atoms\":%d,\"points\":%d,\"entries\":%zu,\"diagonal_spins\":%d,\"status\":%d,\"passed\":%s,\"graph_nodes\":%zu,\"setup_bytes\":%zu,\"export_bytes\":%zu,\"electronic_inputs_resident\":true,\"retained_electronic_bytes\":%zu,\"mesh\":[%d,%d,%d],\"nsm\":%d,\"partial_core\":%d,\"max_force_error_ry_per_bohr\":%.17g,\"max_stress_error_ry_per_bohr3\":%.17g,",
            s.problem.kind,static_cast<unsigned long long>(s.epoch),s.atoms,s.points,s.point.size(),s.problem.diagonal_spins,code,passed?"true":"false",nodes,device.staged,device.exported,s.final.bytes,s.final.mesh[0],s.final.mesh[1],s.final.mesh[2],s.final.nsm,s.final.partial_core,ferr,terr);
        array(file,"reference_force",reference_force,force.size());std::fputc(',',file);array(file,"device_force",force.data(),force.size());std::fputc(',',file);
        array(file,"reference_stress",reference_stress,9);std::fputc(',',file);array(file,"device_stress",stress.data(),9);std::fputs("}\n",file);std::fclose(file);
        if(passed) remag::complete_final_grid_force(s.epoch,s.problem.kind);
        shadow.reset();require(passed,"Atomic-force shadow differs from upstream");return 0;
    } catch(const std::exception& e) {return failed(e);}
}
