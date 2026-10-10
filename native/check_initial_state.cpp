// SPDX-License-Identifier: GPL-3.0-or-later
#include "initial_state.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
namespace {
void require(bool b){if(!b)throw std::runtime_error("Initial owner probe assertion failed");}
void check(cudaError_t e){require(e==cudaSuccess);}
template<class F>void rejected(F&& fn,int& count){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}require(failed);count++;}
void bridge(int code){require(code==0);}
}
int main(int argc,char** argv){
    try {
        require(argc==3);int spins=std::atoi(argv[1]),requests=std::atoi(argv[2]),rejections=0;
        setenv("REMAG_GPU_INITIALIZATION","1",1);setenv("REMAG_DEVICE_SCF","1",1);setenv("REMAG_POPULATION","1",1);setenv("REMAG_INTEGRAL_FORCE","1",1);
        const int rows=3,capacity=5,atoms=2,species=2;
        int num[]={1,2,1},ptr[]={0,1,4},lasto[]={0,2,3},ids[]={2,1};
        double overlap[]={.25,-.5,.125,0.,.75},valence[]={8.,10.};
        std::vector<double> density(size_t(capacity)*spins),view(100),actual(density.size()),exports;
        for(size_t i=0;i<density.size();i++)density[i]=.125+double(i)*.03125;
        const auto original=density;
        rejected([&]{remag::read_initial_population(7,atoms,rows,spins,10,view.data());},rejections);
        rejected([&]{bridge(remag_initial_state_end());},rejections);
        rejected([&]{bridge(remag_initial_state_begin(0,rows,capacity,spins,species,requests,num,ptr,overlap,density.data(),lasto,ids,valence));},rejections);
        bridge(remag_initial_state_begin(atoms,rows,capacity,spins,species,requests,num,ptr,overlap,density.data(),lasto,ids,valence));
        rejected([&]{bridge(remag_initial_state_begin(atoms,rows,capacity,spins,species,requests,num,ptr,overlap,density.data(),lasto,ids,valence));},rejections);
        // Host values cease to be authoritative once the initial owner stages D.
        std::fill(density.begin(),density.end(),std::numeric_limits<double>::quiet_NaN());
        remag_geometry_changed();const auto epoch=remag::geometry_epoch();
        rejected([&]{remag::borrow_initial_density(epoch,rows,capacity,spins,num,ptr,overlap,density.data());},rejections);
        rejected([&]{remag::read_initial_population(7,atoms,rows,spins,9,view.data());},rejections);
        if(requests) {
            const int c=std::min(spins,4),sizes[]={c*rows,c*atoms,c,5*rows,6*atoms,5*species,10};
            for(int k=1;k<=7;k++) {
                if(!(requests&2)&&k!=7)continue;
                remag::read_initial_population(k,atoms,rows,spins,sizes[k-1],view.data());
                exports.insert(exports.end(),view.begin(),view.begin()+sizes[k-1]);
                auto saved=view;
                remag::read_initial_population(k,atoms,rows,spins,sizes[k-1],view.data());
                require(saved==view);
            }
            if(requests==1)rejected([&]{remag::read_initial_population(1,atoms,rows,spins,c*rows,view.data());},rejections);
        } else rejected([&]{remag::read_initial_population(7,atoms,rows,spins,10,view.data());},rejections);
        bridge(remag_initial_state_end());
        rejected([&]{bridge(remag_initial_state_end());},rejections);
        rejected([&]{remag::read_initial_population(7,atoms,rows,spins,10,view.data());},rejections);
        rejected([&]{remag::borrow_initial_density(epoch+1,rows,capacity,spins,num,ptr,overlap,density.data());},rejections);
        rejected([&]{remag::borrow_initial_density(epoch,rows,capacity,spins,num,ptr,overlap,original.data());},rejections);
        overlap[0]+=1.;rejected([&]{remag::borrow_initial_density(epoch,rows,capacity,spins,num,ptr,overlap,density.data());},rejections);overlap[0]-=1.;
        auto device=remag::borrow_initial_density(epoch,rows,capacity,spins,num,ptr,overlap,density.data());
        rejected([&]{remag::borrow_initial_density(epoch,rows,capacity,spins,num,ptr,overlap,density.data());},rejections);
        rejected([&]{remag::read_initial_population(7,atoms,rows,spins,10,view.data());},rejections);
        check(cudaMemcpy(actual.data(),device,actual.size()*sizeof(double),cudaMemcpyDeviceToHost));
        require(std::memcmp(actual.data(),original.data(),actual.size()*sizeof(double))==0);
        rejected([&]{remag::release_initial_density(epoch+1);},rejections);
        remag::release_initial_density(epoch);
        rejected([&]{remag::borrow_initial_density(epoch,rows,capacity,spins,num,ptr,overlap,density.data());},rejections);
        rejected([&]{bridge(remag_initial_state_begin(atoms,rows,capacity,spins,species,requests,num,ptr,overlap,density.data(),lasto,ids,valence));},rejections);
        FILE* raw=std::fopen("initial-owner.bin","wb");require(raw);
        const std::vector<double>* arrays[]={&original,&actual,&exports};
        for(const auto* values:arrays)
            require(std::fwrite(values->data(),sizeof(double),values->size(),raw)==values->size());
        std::fclose(raw);
        FILE* f=std::fopen("probe.json","w");require(f);
        std::fprintf(f,"{\"passed\":true,\"spins\":%d,\"requests\":%d,\"rejections\":%d,\"device_density_preserved_after_host_poison\":true}\n",spins,requests,rejections);std::fclose(f);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Initial owner probe: %s\n",e.what());return 1;}
}
