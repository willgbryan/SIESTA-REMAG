// SPDX-License-Identifier: GPL-3.0-or-later
// Publication-owner probe. Integral geometries are empty; grid/atomic kernels
// give nonzero forces. Material/integral arithmetic is qualified separately.
#include "final_grid_state.hpp"
#include "pair_force.hpp"
#include "nonlocal_compact.hpp"
#include "atomic_grid_force.hpp"
#include "grid_force.hpp"
#include "execution.hpp"
#include "population_state.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
#ifdef REMAG_CHECK_POPULATION
constexpr bool with_population=true;
#else
constexpr bool with_population=false;
#endif
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
struct Device {
    cudaStream_t stream=nullptr;std::vector<void*> memory;
    Device(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Device(){if(stream)cudaStreamSynchronize(stream);for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* upload(size_t n,const T* source){T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),std::max(size_t(1),n)*sizeof(T)));memory.push_back(p);
        if(n)check(cudaMemcpy(p,source,n*sizeof(T),cudaMemcpyHostToDevice));
        return p;}
};
template<class F>int captured(cudaStream_t stream,int expected,F fn){
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));fn();check(cudaStreamEndCapture(stream,&graph));
    size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));require(n==size_t(expected),"Unexpected captured kernel count");
    std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(graph,nodes.data(),&n));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));require(type==cudaGraphNodeTypeKernel,"Non-kernel capture node");}
    check(cudaGraphInstantiate(&executable,graph,0));check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));return int(n);
}
struct Result {uint64_t epoch=0;bool published=false,failed=false;int nodes=0;size_t exported=0;std::vector<double> output,population;};
}
int main(int argc,char** argv){
    try{
        require(argc==5,"Expected scenario, spins, partial-core, report");const std::string scenario=argv[1];int nh=std::stoi(argv[2]),core=std::stoi(argv[3]);
        require((nh==1||nh==2||nh==4||nh==8)&&(core==0||core==1),"Invalid fixture spin/core");
        const bool bad_observables=with_population&&scenario=="failed_observables";
        const bool bad_population=with_population&&scenario=="failed_population";
        const bool recovery=scenario=="recovery",bad_status=scenario=="failed_status",bad_finite=scenario=="failed_nonfinite",stale=scenario=="stale";
        require(scenario=="normal"||recovery||bad_status||bad_finite||stale||bad_population||bad_observables,"Invalid fixture scenario");
        for(const char* v:{"REMAG_INTEGRAL_FORCE","REMAG_GRID_FORCE","REMAG_ATOMIC_GRID_FORCE"})setenv(v,"1",1);
        for(const char* v:{"REMAG_PAIR_FORCE_SHADOW","REMAG_NONLOCAL_FORCE_SHADOW","REMAG_TWO_CENTER_SHADOW","REMAG_ATOMIC_FORCE_SHADOW","REMAG_ATOMIC_BASIS_SHADOW","REMAG_ATOMIC_BASIS"})unsetenv(v);
        if(with_population)setenv("REMAG_POPULATION","1",1);else unsetenv("REMAG_POPULATION");
        std::vector<std::string> rejections;std::vector<Result> results;
        auto reject=[&](const char* label,auto fn){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}require(caught,label);rejections.emplace_back(label);};
        for(int geometry=0;geometry<(recovery?2:1);geometry++){
            Device d;require(remag_phase_enter(0,geometry,0)==0,"Initial phase failed");remag_geometry_changed();const auto epoch=remag::geometry_epoch();
            const int ng=std::min(nh,4),nsd=std::min(nh,2),nonlocal=nh==8?3:2;results.emplace_back();auto& result=results.back();result.epoch=epoch;
            double density[]={.4,.3,.2,-.1,.07,.08,.12,.14};
            if(bad_observables)density[0]=-1.e308;
            const double energy[4]={},potential[]={1.2,1.8,.2,.3},auxiliary=.6,deformation=.8;
            double hartree[9],tensor[9],force[]={.1,-.2,.3};for(int i=0;i<9;i++){hartree[i]=.001*(i+1);tensor[i]=.01*(i+1);}
            auto D=d.upload(nh,density),E=d.upload(ng,energy),V=d.upload(ng,potential),A=d.upload(1,&auxiliary),R=d.upload(1,&deformation),HS=d.upload(9,hartree);
            int one=1,zero=0,ends[]={0,1},atom_pair[]={1,1};double phi=.7,occupation=.3;
            double core_grad[]={.2,-.3,.4},neutral_grad[]={-.2,.5,-.1},df_grad[]={.3,.2,-.4},dx[]={.4,.2,-.3};
            auto num=d.upload(1,&one),ptr=d.upload(1,&zero),list=d.upload(1,&one),lookup=d.upload(1,&zero);
            auto xyz=d.upload(3,dx);
            remag::GridContractionView grid{{1,1,1,1,1,1},d.upload(2,ends),list,ptr,lookup,nullptr,1};
            remag::GridForceView gv{grid,1,1,list,d.upload(1,&phi),d.upload(3,df_grad),xyz,d.upload(1,&occupation)};
            remag::AtomicGridForceView av{1,1,1,list,list,d.upload(3,core_grad),xyz};auto neutral=d.upload(3,neutral_grad);
            remag::AtomicGridForceProblem cp{.1,2.,1,nsd,1,1},np{.1,2.,2,1,1,1};remag::GridForceProblem gp{.1,2.,1,1};
            auto pc=d.upload(1,&cp),pn=d.upload(1,&np);auto pg=d.upload(1,&gp);
            remag::PairForceProblem kp{2.,1,0},op{2.,4,0};auto pk=d.upload(1,&kp),po=d.upload(1,&op);
            remag::PairForceView pair{0,1,1,1,1,nsd,list,list,d.upload(2,atom_pair),num,ptr,list,d.upload(1,&phi),xyz};
            remag::NonlocalCenter center{};remag::NonlocalNeighbor neighbor{};remag::NonlocalBlock block{};
            remag::NonlocalCompactView nl{0,0,0,0,1,1,1,1,nh,nsd,nh==8?1:0,1,2.,d.upload(1,&center),d.upload(1,&neighbor),d.upload(1,&block),num,ptr,list,lookup,pair.integrals,xyz};
            auto weights=d.upload(1,&phi);
            if(with_population){
                int species_id=1;double valence=bad_observables?std::numeric_limits<double>::max():2.4;
                if(recovery){int bad_ends[]={0,2};reject("bad_population_partition",[&]{remag::prepare_population_state(epoch,1,1,1,nh,1,bad_ends,&species_id,&valence);});}
                if(recovery){int invalid_id=2;reject("bad_population_species",[&]{remag::prepare_population_state(epoch,1,1,1,nh,1,ends,&invalid_id,&valence);});}
                remag::prepare_population_state(epoch,1,1,1,nh,1,ends,&species_id,&valence);
                species_id=-1;valence=std::numeric_limits<double>::quiet_NaN();
                if(recovery){double out[4]{};reject("population_read_before_binding",[&]{remag::read_final_population(epoch,3,1,1,nh,ng,out);});}
                int count=bad_population?2:1;auto popnum=d.upload(1,&count),popptr=d.upload(1,&zero);auto popoverlap=d.upload(1,&phi);
                remag::bind_population_topology(epoch,popnum,popptr,popoverlap,d.stream);
                // Prove topology/overlap retain their own device storage.
                check(cudaMemset(popnum,0xff,4));check(cudaMemset(popptr,0xff,4));check(cudaMemset(popoverlap,0xff,8));
                if(recovery)reject("population_duplicate_binding",[&]{remag::bind_population_topology(epoch,popnum,popptr,popoverlap,d.stream);});
            }
            require(remag_phase_enter(2,geometry,0)==0,"Final phase failed");
            remag::FinalGridStateView input{grid.shape,nh,ng,epoch,.1,2.,D,E,V,A,R,0,{1,1,1},1,core,HS};
            if(recovery){auto missing=input;missing.hartree_stress=nullptr;reject("missing_hartree",[&]{remag::retain_final_grid_state(missing,d.stream);});}
            remag::retain_final_grid_state(input,d.stream);check(cudaMemset(HS,0,9*sizeof(double)));check(cudaMemset(D,0,nh*sizeof(double)));
            if(recovery){
                reject("early_grid",[&]{remag::borrow_final_grid_state(epoch,core?1:2);});
                reject("early_nonlocal",[&]{remag::borrow_final_pair_state(epoch,nonlocal);});
                reject("early_collect",[&]{remag::collect_final_forces(epoch,1,force,tensor);});
                reject("early_publish",[&]{remag::complete_final_force_publication(epoch);});
            }
            auto fields=remag::borrow_final_pair_state(epoch,1);auto pool=remag::final_force_accumulator(epoch,1,1,1,force,tensor);
            if(with_population&&recovery){
                reject("population_early_evaluate",[&]{remag::evaluate_final_population(fields,pool);});
                reject("population_early_publish",[&]{remag::complete_population_publication(epoch);});
            }
            if(recovery){
                reject("unrecorded_kinetic",[&]{remag::complete_final_pair_shadow(epoch,1);});
                force[0]+=1.;reject("changed_initial_force",[&]{remag::final_force_accumulator(epoch,1,1,1,force,tensor);});force[0]=.1;
                reject("retained_pair",[&]{remag::record_final_integral_force(epoch,{1,0,1,0,0,true});});
            }
            result.nodes+=captured(pool.stream,1,[&]{check(remag::add_pair_force_device(pair,pk,fields.density,pool.energy,pool.force,pool.stress,pool.status,pool.stream));});
            remag::record_final_integral_force(epoch,{1,0,1,0,0});remag::complete_final_pair_shadow(epoch,1);
            fields=remag::borrow_final_pair_state(epoch,nonlocal);
            if(recovery){
                reject("nonlocal_unmarked_reuse",[&]{remag::record_final_integral_force(epoch,{nonlocal,1,2,0,0});});
                reject("nonlocal_retained_generator",[&]{remag::record_final_integral_force(epoch,{nonlocal,1,3,0,0,true});});
            }
            result.nodes+=captured(pool.stream,2,[&]{check(remag::add_nonlocal_compact_device(nl,fields.density,weights,pool.energy+1,pool.force,pool.stress,pool.status,pool.stream));});
            remag::record_final_integral_force(epoch,{nonlocal,0,2,0,0,recovery&&geometry==1});remag::complete_final_pair_shadow(epoch,nonlocal);
            if(recovery)reject("grid_before_hartree",[&]{remag::borrow_final_grid_state(epoch,core?1:2);});
            if(bad_status){int status=7;check(cudaMemcpy(pool.status,&status,sizeof(status),cudaMemcpyHostToDevice));}
            remag::add_final_hartree_stress(epoch);
            if(recovery)reject("duplicate_hartree",[&]{remag::add_final_hartree_stress(epoch);});
            for(int kind=core?1:2;kind<=2;kind++){
                fields=remag::borrow_final_grid_state(epoch,kind);auto same=remag::final_force_accumulator(epoch,1,1,1,force,tensor);
                require(same.energy==pool.energy&&same.force==pool.force&&same.stress==pool.stress&&same.status==pool.status&&same.stream==pool.stream,"Changed shared accumulator");
                auto current=av;if(kind==2)current.gradient=neutral;
                result.nodes+=captured(pool.stream,1,[&]{check(remag::add_atomic_grid_force_device(current,kind==1?pc:pn,fields.potential,fields.auxiliary,fields.deformation,pool.force,pool.stress,pool.status,pool.stream));});
                remag::record_final_atomic_force(epoch,{kind,1,1,kind==1?nsd:1,1,88,56,56});remag::complete_final_grid_force(epoch,kind);
            }
            fields=remag::borrow_final_grid_state(epoch,4);
            result.nodes+=captured(pool.stream,1,[&]{check(remag::add_grid_force_device(gv,nh,pg,fields.density,fields.potential,fields.auxiliary,pool.force,pool.stress,pool.status,pool.stream));});
            remag::complete_final_grid_force(epoch,4);fields=remag::borrow_final_pair_state(epoch,4);
            result.nodes+=captured(pool.stream,1,[&]{check(remag::add_pair_force_device(pair,po,fields.energy_density,pool.energy,pool.force,pool.stress,pool.status,pool.stream));});
            remag::record_final_integral_force(epoch,{4,0,1,0,0});remag::complete_final_pair_shadow(epoch,4);
            require(!remag::final_grid_forces_complete(epoch),"Analysis released before publication");
            if(recovery){reject("reset_before_collection",[&]{remag::reset_final_grid_state();});tensor[0]+=1.;reject("changed_publication_input",[&]{remag::collect_final_forces(epoch,1,force,tensor);});tensor[0]=.01;}
            if(bad_finite){double invalid=std::numeric_limits<double>::quiet_NaN();check(cudaMemcpy(pool.energy,&invalid,8,cudaMemcpyHostToDevice));}
            if(stale)remag_geometry_changed();
            if(bad_observables){int status=-1;check(cudaMemcpy(&status,pool.status,4,cudaMemcpyDeviceToHost));require(status==0,"Observable failure fixture already failed forces");}
            if(bad_status||bad_finite||stale||bad_population||bad_observables){
                reject("failed_collect",[&]{remag::collect_final_forces(epoch,1,force,tensor);});result.failed=true;
                if(bad_observables){int status=0;check(cudaMemcpy(&status,pool.status,4,cudaMemcpyDeviceToHost));require(status==16,"Derived population failure did not join shared status");}
                reject("retry_collect",[&]{remag::collect_final_forces(epoch,1,force,tensor);});
                reject("failed_publish",[&]{remag::complete_final_force_publication(epoch);});
                reject("failed_reset",[&]{remag::reset_final_grid_state();});
                reject("replace_failed_state",[&]{remag::retain_final_grid_state(input,d.stream);});
                require(!remag::final_grid_forces_complete(epoch),"Failed publication released analysis");
                if(with_population){double out[4]{};reject("failed_population_read",[&]{remag::read_final_population(epoch,3,1,1,nh,ng,out);});}

            }else{
                const auto publication=remag::collect_final_forces(epoch,1,force,tensor);result.exported=publication.exported_bytes;
                result.output=publication.energy;result.output.insert(result.output.end(),publication.force.begin(),publication.force.end());result.output.insert(result.output.end(),publication.stress.begin(),publication.stress.end());
                require(!remag::final_grid_forces_complete(epoch),"Collection alone released analysis");
                if(with_population&&recovery){double out[4]{};reject("population_read_before_publication",[&]{remag::read_final_population(epoch,3,1,1,nh,ng,out);});}

                if(recovery){reject("duplicate_collect",[&]{remag::collect_final_forces(epoch,1,force,tensor);});reject("reset_before_publication",[&]{remag::reset_final_grid_state();});}
                remag::complete_final_force_publication(epoch);require(remag::final_grid_forces_complete(epoch),"Successful publication did not release analysis");result.published=true;
                if(recovery)reject("duplicate_publish",[&]{remag::complete_final_force_publication(epoch);});
                if(with_population){
                    for(int kind=1;kind<=3;kind++){
                        std::vector<double> output(ng);remag::read_final_population(epoch,kind,1,1,nh,int(output.size()),output.data());
                        for(int component=0;component<ng;component++){
                            double wanted=density[component];if(nh==8&&component>=2)wanted=.5*(wanted+density[component+4]);
                            require(std::isfinite(output[component])&&std::abs(output[component]-.7*wanted)<1.e-13,"Population does not use retained D/S");
                        }
                        result.population.insert(result.population.end(),output.begin(),output.end());
                    }
                    // Derived outputs are independently reconstructed by the Python audit.
                    const int sizes[]={5,6,5,10};
                    for(int kind=4;kind<=7;kind++){
                        std::vector<double> output(sizes[kind-4]);
                        remag::read_final_population(epoch,kind,1,1,nh,int(output.size()),output.data());
                        result.population.insert(result.population.end(),output.begin(),output.end());
                    }
                    std::vector<double> repeated(ng);remag::read_final_population(epoch,1,1,1,nh,int(repeated.size()),repeated.data());
                    require(std::equal(repeated.begin(),repeated.end(),result.population.begin()),"Repeated population export changed");
                    if(recovery){double out[4]{};reject("population_wrong_shape",[&]{remag::read_final_population(epoch,3,2,1,nh,ng,out);});}
                    if(recovery){double out[4]{};reject("population_wrong_output_size",[&]{remag::read_final_population(epoch,7,1,1,nh,4,out);});}
                }
                remag::reset_final_grid_state();
                if(with_population&&recovery){double out[4]{};reject("population_read_after_reset",[&]{remag::read_final_population(epoch,3,1,1,nh,ng,out);});}

            }
        }
        FILE* f=std::fopen(argv[4],"w");require(f,"Cannot write probe report");
        std::fprintf(f,"{\"scenario\":\"%s\",\"spins\":%d,\"partial_core\":%d,\"rejections\":[",scenario.c_str(),nh,core);
        for(size_t i=0;i<rejections.size();i++)std::fprintf(f,"%s\"%s\"",i?",":"",rejections[i].c_str());
        std::fputs("],\"cycles\":[",f);
        for(size_t i=0;i<results.size();i++){auto& r=results[i];std::fprintf(f,"%s{\"epoch\":%llu,\"published\":%s,\"failed\":%s,\"captured_nodes\":%d,\"successful_export_bytes\":%zu,\"output\":[",i?",":"",static_cast<unsigned long long>(r.epoch),r.published?"true":"false",r.failed?"true":"false",r.nodes,r.exported);
            for(size_t j=0;j<r.output.size();j++)std::fprintf(f,"%s%.17g",j?",":"",r.output[j]);
            if(with_population){
                std::fputs("],\"population\":[",f);
                for(size_t j=0;j<r.population.size();j++)std::fprintf(f,"%s%.17g",j?",":"",r.population[j]);
            }
            std::fputs("]}",f);}
        std::fputs("]}\n",f);std::fclose(f);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Integral publication probe: %s\n",e.what());return 1;}
}
