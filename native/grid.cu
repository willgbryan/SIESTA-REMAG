// SPDX-License-Identifier: GPL-3.0-or-later
// Sparse orbital/grid contractions. Basis values and topology persist across SCF calls.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "resident.hpp"
#include "execution.hpp"
#include "grid.hpp"
#include "initial_mesh_state.hpp"

namespace {
void check(cudaError_t c) { if(c!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(c)); }
struct Counts {
    unsigned long long calls=0, uploads=0, launches=0, h2d=0, d2h=0, allocations=0, live=0, peak=0;
    unsigned long long topology_prepares=0,density_cluster8_enqueues=0,potential_spin_enqueues=0,mesh_value_device_bindings=0,avoided_mesh_value_upload_bytes=0;
    unsigned long long resident_hits=0,avoided_h2d=0;
    unsigned long long epoch_hits=0,validation_bytes=0,avoided_validation_bytes=0;
    unsigned long long retained_grid_density=0,retained_grid_potential=0,avoided_grid_download=0,avoided_grid_upload=0;
    double seconds=0;
    ~Counts() {
        if(FILE* f=std::fopen("remag-grid.json","w")) {
            std::fprintf(f,"{\"calls\":%llu,\"basis_uploads\":%llu,\"kernel_launch_count\":%llu,"
                "\"host_to_device_bytes\":%llu,\"device_to_host_bytes\":%llu,"
                "\"device_allocation_count\":%llu,\"peak_grid_device_bytes\":%llu,\"wall_seconds\":%.9f,"
                "\"resident_density_hits\":%llu,\"avoided_density_host_to_device_bytes\":%llu,"
                "\"geometry_epoch_hits\":%llu,\"host_validation_bytes\":%llu,\"avoided_host_validation_bytes\":%llu,"
                "\"resident_grid_density_producers\":%llu,\"resident_grid_potential_consumers\":%llu,"
                "\"avoided_grid_density_download_bytes\":%llu,\"avoided_grid_potential_upload_bytes\":%llu,\"topology_only_prepares\":%llu,\"density_cluster8_host_enqueues\":%llu,\"potential_spin_host_enqueues\":%llu,\"mesh_value_device_bindings\":%llu,\"avoided_mesh_value_upload_bytes\":%llu}\n",
                calls,uploads,launches,h2d,d2h,allocations,peak,seconds,resident_hits,avoided_h2d,
                epoch_hits,validation_bytes,avoided_validation_bytes,retained_grid_density,retained_grid_potential,
                avoided_grid_download,avoided_grid_upload,topology_prepares,density_cluster8_enqueues,potential_spin_enqueues,mesh_value_device_bindings,avoided_mesh_value_upload_bytes);
            std::fclose(f);
        }
    }
} counts;
struct Buffer {
    void* p=nullptr; size_t bytes=0;
    void reserve(size_t n) {
        if(n<=bytes) return;
        if(p) { check(cudaFree(p)); counts.live-=bytes; p=nullptr; bytes=0; }
        check(cudaMalloc(&p,n)); bytes=n; counts.live+=n; counts.allocations++;
        counts.peak=std::max(counts.peak,counts.live);
    }
    void upload(const void* src,size_t n,remag::Transfer boundary=remag::Transfer::grid_topology) {
        reserve(n); check(cudaMemcpy(p,src,n,cudaMemcpyHostToDevice)); counts.h2d+=n;
        remag::transfer(boundary,true,n);
    }
    ~Buffer() { if(p) cudaFree(p); }
};
struct Plan {
    uint64_t fingerprint=0;
    uint64_t sparse_fingerprint=0;
    uint64_t epoch=0;
    std::array<int,9> shape{};
    std::array<const void*,7> sources{};
    Buffer ends,orbs,phi,lookup,translations,input,output;
    const float* device_phi=nullptr;
} plan;
std::mutex grid_mutex;
bool grid_leased=false;
uint64_t hash(uint64_t h,const void* ptr,size_t bytes) {
    // FNV-1a verifies geometry/sparsity changes even if Fortran reuses addresses.
    auto p=static_cast<const unsigned char*>(ptr);
    for(size_t i=0;i<bytes;i++) { h^=p[i]; h*=1099511628211ULL; }
    return h;
}
__device__ int pair_index(int i,int j,int nuo,int no,int cells,const int* trans,const int* lookup) {
    const int relative=trans[(i/nuo)*cells+j/nuo];
    return lookup[(size_t)(i%nuo)*no+relative*nuo+j%nuo];
}
__global__ void density(int np,int nsp,int nuo,int no,int cells,int nnz,int nh,
                       const int* ends,const int* orbs,const int* trans,const int* lookup,
                       const float* phi,const double* dm,double* rho) {
    const int ip=blockIdx.x, spin=blockIdx.y, sub=blockIdx.z;
    const int begin=ends[ip], nc=ends[ip+1]-begin;
    double value=0.;
    for(long long pair=threadIdx.x;pair<(long long)nc*nc;pair+=blockDim.x) {
        int a=begin+pair/nc,b=begin+pair%nc;
        int index=pair_index(orbs[a]-1,orbs[b]-1,nuo,no,cells,trans,lookup);
        if(index<0) continue;
        double d=dm[(size_t)spin*nnz+index];
        if(nh==8 && spin>=2) d=.5*(d+dm[(size_t)(spin+4)*nnz+index]);
        value+=d*(double)phi[(size_t)a*nsp+sub]*(double)phi[(size_t)b*nsp+sub];
    }
    __shared__ double sums[256];
    sums[threadIdx.x]=value;
    __syncthreads();
    for(int stride=blockDim.x/2;stride;stride/=2) {
        if(threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x==0) rho[((size_t)spin*np+ip)*nsp+sub]=sums[0];
}
// A SIESTA nsm=2 cluster contains eight adjacent mesh points. Keep each
// point's original pair/thread summation order, sharing the expensive sparse
// lookup and density coefficient across the eight points in registers.
__global__ void density_cluster8(int np,int nuo,int no,int cells,int nnz,int nh,
                       const int* ends,const int* orbs,const int* trans,const int* lookup,
                       const float* phi,const double* dm,double* rho) {
    const int ip=blockIdx.x,spin=blockIdx.y,begin=ends[ip],nc=ends[ip+1]-begin;
    double value[8]{};
    for(long long pair=threadIdx.x;pair<(long long)nc*nc;pair+=blockDim.x) {
        const int a=begin+pair/nc,b=begin+pair%nc;
        const int index=pair_index(orbs[a]-1,orbs[b]-1,nuo,no,cells,trans,lookup);
        if(index<0) continue;
        double d=dm[(size_t)spin*nnz+index];
        if(nh==8 && spin>=2) d=.5*(d+dm[(size_t)(spin+4)*nnz+index]);
        #pragma unroll
        for(int sub=0;sub<8;sub++)
            value[sub]+=d*(double)phi[(size_t)a*8+sub]*(double)phi[(size_t)b*8+sub];
    }
    __shared__ double sums[8][256];
    #pragma unroll
    for(int sub=0;sub<8;sub++) sums[sub][threadIdx.x]=value[sub];
    __syncthreads();
    for(int stride=blockDim.x/2;stride;stride/=2) {
        if(threadIdx.x<stride) {
            #pragma unroll
            for(int sub=0;sub<8;sub++) sums[sub][threadIdx.x]+=sums[sub][threadIdx.x+stride];
        }
        __syncthreads();
    }
    if(threadIdx.x<8) rho[((size_t)spin*np+ip)*8+threadIdx.x]=sums[threadIdx.x][0];
}
__global__ void potential(int np,int nsp,int nuo,int no,int cells,int nnz,int nh,double host_dvol,const double* device_dvol,
                         const int* ends,const int* orbs,const int* trans,const int* lookup,
                         const float* phi,const double* v,double* vs) {
    int ip=blockIdx.x,spin=blockIdx.y,begin=ends[ip],nc=ends[ip+1]-begin;
    const double dvol=device_dvol?*device_dvol:host_dvol;
    for(long long pair=threadIdx.x;pair<(long long)nc*nc;pair+=blockDim.x) {
        int a=begin+pair/nc,b=begin+pair%nc;
        int index=pair_index(orbs[a]-1,orbs[b]-1,nuo,no,cells,trans,lookup);
        if(index<0) continue;
        double value=0.;
        for(int sub=0;sub<nsp;sub++)
            value+=(double)phi[(size_t)a*nsp+sub]*v[((size_t)spin*np+ip)*nsp+sub]*(double)phi[(size_t)b*nsp+sub];
        atomicAdd(&vs[(size_t)spin*nnz+index],value*dvol);
        if(nh==8 && spin>=2) atomicAdd(&vs[(size_t)(spin+4)*nnz+index],value*dvol);
    }
}

// One block shares orbital-pair lookup and basis loads across the active spin
// components. The potential at this mesh cluster is uniform across each block.
// Keep each spin's sub-point arithmetic and the existing sparse atomic update;
// SOC components 2/3 still add to their corresponding 6/7 blocks.
template<int NG>
__global__ void potential_spins(int np,int nsp,int nuo,int no,int cells,int nnz,int nh,
        double host_dvol,const double* device_dvol,const int* ends,const int* orbs,
        const int* trans,const int* lookup,const float* phi,const double* v,double* vs) {
    const int ip=blockIdx.x,begin=ends[ip],nc=ends[ip+1]-begin;
    const double dvol=device_dvol?*device_dvol:host_dvol;
    __shared__ double cluster_v[NG*64];
    for(int j=threadIdx.x;j<NG*nsp;j+=blockDim.x)
        cluster_v[j]=v[(size_t(j/nsp)*np+ip)*nsp+j%nsp];
    __syncthreads();
    for(long long pair=threadIdx.x;pair<(long long)nc*nc;pair+=blockDim.x) {
        const int a=begin+pair/nc,b=begin+pair%nc;
        const int index=pair_index(orbs[a]-1,orbs[b]-1,nuo,no,cells,trans,lookup);
        if(index<0) continue;
        double value[NG]{};
        for(int sub=0;sub<nsp;sub++) {
            const double pa=phi[size_t(a)*nsp+sub],pb=phi[size_t(b)*nsp+sub];
            #pragma unroll
            for(int spin=0;spin<NG;spin++)
                value[spin]+=pa*cluster_v[spin*nsp+sub]*pb;
        }
        #pragma unroll
        for(int spin=0;spin<NG;spin++) {
            atomicAdd(&vs[size_t(spin)*nnz+index],value[spin]*dvol);
            if(nh==8 && spin>=2) atomicAdd(&vs[size_t(spin+4)*nnz+index],value[spin]*dvol);
        }
    }
}
} // namespace

namespace remag {
namespace {
bool valid(const GridContractionView& v,int ng,int nh) {
    const auto& p=v.shape;
    return p.no>0&&p.np>0&&p.nsp>0&&p.nuo>0&&p.nnz>0&&p.cells>0&&
        p.no%p.nuo==0&&p.no/p.nuo==p.cells&&ng>0&&ng<=4&&ng<=nh&&
        (nh==1||nh==2||nh==4||nh==8)&&v.ends&&v.orbs&&v.translations&&v.lookup&&v.phi;
}
GridContractionView view() {
    const auto& p=plan.shape;
    if(!plan.fingerprint||p[3]<1||plan.epoch!=geometry_epoch())
        throw std::runtime_error("Grid contraction has no matching geometry owner");
    return {{p[0],p[1],p[2],p[3],p[4],p[0]/p[3]},static_cast<const int*>(plan.ends.p),
        static_cast<const int*>(plan.orbs.p),static_cast<const int*>(plan.translations.p),
        static_cast<const int*>(plan.lookup.p),(plan.device_phi?plan.device_phi:static_cast<const float*>(plan.phi.p)),p[5]};
}
}
namespace {
GridContractionView checked_grid_view() {
    if(grid_leased) throw std::runtime_error("Grid geometry is exclusively leased to a cycle");
    auto v=view();
    const char* verify=std::getenv("REMAG_GRID_VERIFY_EPOCH");
    if(verify&&std::strcmp(verify,"1")==0) {
        const auto& p=plan.shape;const auto& s=plan.sources;
        auto h=hash(14695981039346656037ULL,p.data(),sizeof(p));
        h=hash(h,s[0],(size_t(p[1])+1)*sizeof(int));h=hash(h,s[1],size_t(p[5])*sizeof(int));
        h=hash(h,s[2],size_t(p[3])*sizeof(int));h=hash(h,s[3],size_t(p[3])*sizeof(int));
        h=hash(h,s[4],size_t(p[4])*sizeof(int));
        if(plan.device_phi)h=hash(h,&plan.epoch,sizeof(plan.epoch));
        else h=hash(h,s[5],size_t(p[5])*p[2]*sizeof(float));
        if(h!=plan.fingerprint||std::memcmp(s[6],p.data()+6,3*sizeof(int)))
            throw std::runtime_error("Grid view inputs changed inside their lifetime");
    }
    return v;
}
}
GridContractionView grid_contraction_view() {
    std::lock_guard<std::mutex> guard(grid_mutex);return checked_grid_view();
}
void validate_grid_sparse_topology(int rows,int columns,int capacity,const int* num,const int* ptr,const int* list) {
    std::lock_guard<std::mutex> guard(grid_mutex);auto v=checked_grid_view();
    if(in_scf()||rows!=v.shape.nuo||columns!=v.shape.no||capacity!=v.shape.nnz||!num||!ptr||!list)
        throw std::runtime_error("Pair sparse shape differs from retained grid geometry");
    auto h=hash(14695981039346656037ULL,num,size_t(rows)*sizeof(int));
    h=hash(h,ptr,size_t(rows)*sizeof(int));h=hash(h,list,size_t(capacity)*sizeof(int));
    if(h!=plan.sparse_fingerprint)throw std::runtime_error("Pair sparse topology differs from retained grid geometry");
}
GridContractionView lease_grid_contraction() {
    std::lock_guard<std::mutex> guard(grid_mutex);auto v=checked_grid_view();grid_leased=true;return v;
}
void release_grid_contraction() {
    std::lock_guard<std::mutex> guard(grid_mutex);
    if(!grid_leased) throw std::runtime_error("Grid geometry has no cycle lease");grid_leased=false;
}
cudaError_t contract_density_device(const GridContractionView& v,int ng,int nh,
        const double* dm,double* rho,cudaStream_t stream) {
    if(!valid(v,ng,nh)||!dm||!rho||dm==rho) return cudaErrorInvalidValue;
    const auto& p=v.shape;
    if(p.nsp==8) {
        counts.density_cluster8_enqueues++; // Host enqueue/capture count, not executed SCF iterations.
        density_cluster8<<<dim3(p.np,ng),256,0,stream>>>(p.np,p.nuo,p.no,p.cells,p.nnz,nh,
            v.ends,v.orbs,v.translations,v.lookup,v.phi,dm,rho);
    } else {
    density<<<dim3(p.np,ng,p.nsp),256,0,stream>>>(p.np,p.nsp,p.nuo,p.no,p.cells,p.nnz,nh,
        v.ends,v.orbs,v.translations,v.lookup,v.phi,dm,rho);
    }
    return cudaGetLastError();
}
cudaError_t add_grid_potential_device(const GridContractionView& v,int ng,int nh,
        const double* values,double host_dvol,const double* device_dvol,double* h,cudaStream_t stream) {
    if(!valid(v,ng,nh)||!values||!h||values==h) return cudaErrorInvalidValue;
    const auto& p=v.shape;
    const char* option=std::getenv("REMAG_GRID_POTENTIAL_SPINS");
    if(option&&std::strcmp(option,"1")==0&&p.nsp<=64&&(ng==1||ng==2||ng==4)) {
        counts.potential_spin_enqueues++; // Capture/enqueue count, not runtime iteration count.
        if(ng==1) potential_spins<1><<<p.np,256,0,stream>>>(p.np,p.nsp,p.nuo,p.no,p.cells,p.nnz,nh,host_dvol,device_dvol,
            v.ends,v.orbs,v.translations,v.lookup,v.phi,values,h);
        else if(ng==2) potential_spins<2><<<p.np,256,0,stream>>>(p.np,p.nsp,p.nuo,p.no,p.cells,p.nnz,nh,host_dvol,device_dvol,
            v.ends,v.orbs,v.translations,v.lookup,v.phi,values,h);
        else potential_spins<4><<<p.np,256,0,stream>>>(p.np,p.nsp,p.nuo,p.no,p.cells,p.nnz,nh,host_dvol,device_dvol,
            v.ends,v.orbs,v.translations,v.lookup,v.phi,values,h);
    } else {
        potential<<<dim3(p.np,ng),256,0,stream>>>(p.np,p.nsp,p.nuo,p.no,p.cells,p.nnz,nh,host_dvol,device_dvol,
            v.ends,v.orbs,v.translations,v.lookup,v.phi,values,h);
    }
    return cudaGetLastError();
}
}

extern "C" int remag_grid_enabled() {
    const char* v=std::getenv("REMAG_GRID");
    return v && std::strcmp(v,"1")==0; // opt-in until upstream regression passes
}
extern "C" int remag_grid(int mode,int no,int np,int nsp,int nuo,int nnz,int ng,int nh,
                          const int* nsc,const int* ends,const int* orbs,const int* num,
                          const int* ptr,const int* cols,const float* phi,
                          const double* input,double* output,double dvol) {
    std::lock_guard<std::mutex> guard(grid_mutex);
    auto start=std::chrono::steady_clock::now();
    try {
        if(grid_leased) return -117;
        if(no<1||np<1||nsp<1||nuo<1||nnz<1||no%nuo||ng<1||ng>4||(nh!=1&&nh!=2&&nh!=4&&nh!=8)) return -110;
        if(mode!=0 && mode!=1 && mode!=2) return -111;
        if(!nsc||!ends||!orbs||!num||!ptr||!cols||!phi||(mode!=2&&(!input||!output))) return -110;
        int visible=0; check(cudaGetDeviceCount(&visible));
        if(visible!=1) return -112;
        const int cells=no/nuo, nt=ends[np];
        if(nsc[0]*nsc[1]*nsc[2]!=cells || nt<0) return -113;
        if(size_t(nuo)*no*sizeof(int) > 512ULL*1024*1024) return -114;
        std::array<int,9> shape={no,np,nsp,nuo,nnz,nt,nsc[0],nsc[1],nsc[2]};
        std::array<const void*,7> sources={ends,orbs,num,ptr,cols,phi,nsc};
        auto epoch=remag::geometry_epoch();
        const bool device_values=remag::initial_mesh_enabled();
        if(device_values&&mode!=2)throw std::runtime_error("Device mesh values require device SCF topology preparation");
        const float* device_phi=device_values?remag::initial_mesh_values(epoch,no,np,nsp,nt,ends,orbs):nullptr;
        const char* epoch_option=std::getenv("REMAG_GRID_EPOCH");
        const char* verify_option=std::getenv("REMAG_GRID_VERIFY_EPOCH");
        bool enabled=!epoch_option||std::strcmp(epoch_option,"0")!=0;
        bool verify=verify_option&&std::strcmp(verify_option,"1")==0;
        bool reuse=enabled&&epoch&&epoch==plan.epoch&&shape==plan.shape&&sources==plan.sources;
        size_t scanned=sizeof(shape)+(size_t(np)+1+size_t(nt)+2*size_t(nuo)+size_t(nnz))*sizeof(int)
                       +(device_values?0:size_t(nt)*nsp*sizeof(float));
        uint64_t h=plan.fingerprint;
        if(!reuse||verify) {
            h=hash(14695981039346656037ULL,shape.data(),sizeof(shape));
            h=hash(h,ends,(size_t(np)+1)*sizeof(int));h=hash(h,orbs,size_t(nt)*sizeof(int));
            h=hash(h,num,size_t(nuo)*sizeof(int));h=hash(h,ptr,size_t(nuo)*sizeof(int));
            h=hash(h,cols,size_t(nnz)*sizeof(int));
            if(device_values)h=hash(h,&epoch,sizeof(epoch));
            else h=hash(h,phi,size_t(nt)*nsp*sizeof(float));
            counts.validation_bytes+=scanned;
            if(reuse&&h!=plan.fingerprint) throw std::runtime_error("Grid state changed without geometry invalidation");
        } else {
            counts.avoided_validation_bytes+=scanned;
        }
        if(reuse) counts.epoch_hits++;
        if(h!=plan.fingerprint) {
            std::vector<int> lookup(size_t(nuo)*no,-1),translations(size_t(cells)*cells);
            for(int row=0;row<nuo;row++) for(int k=ptr[row];k<ptr[row]+num[row];k++) {
                if(k<0||k>=nnz||cols[k]<1||cols[k]>no) return -115;
                lookup[size_t(row)*no+cols[k]-1]=k;
            }
            for(int i=0;i<cells;i++) for(int j=0;j<cells;j++) {
                int x=(j%nsc[0]-i%nsc[0]+nsc[0])%nsc[0];
                int y=((j/nsc[0])%nsc[1]-(i/nsc[0])%nsc[1]+nsc[1])%nsc[1];
                int z=(j/(nsc[0]*nsc[1])-i/(nsc[0]*nsc[1])+nsc[2])%nsc[2];
                translations[size_t(i)*cells+j]=x+nsc[0]*(y+nsc[1]*z);
            }
            plan.lookup.upload(lookup.data(),lookup.size()*sizeof(int));
            plan.translations.upload(translations.data(),translations.size()*sizeof(int));
            plan.ends.upload(ends,(size_t(np)+1)*sizeof(int));
            plan.orbs.upload(orbs,size_t(nt)*sizeof(int));
            if(device_values){counts.mesh_value_device_bindings++;counts.avoided_mesh_value_upload_bytes+=size_t(nt)*nsp*sizeof(float);}
            else plan.phi.upload(phi,size_t(nt)*nsp*sizeof(float));
            plan.fingerprint=h;counts.uploads++;
            auto sparse=hash(14695981039346656037ULL,num,size_t(nuo)*sizeof(int));
            sparse=hash(sparse,ptr,size_t(nuo)*sizeof(int));plan.sparse_fingerprint=hash(sparse,cols,size_t(nnz)*sizeof(int));
        }
        plan.epoch=epoch;plan.shape=shape;plan.sources=sources;plan.device_phi=device_phi;
        // Stage geometry for a device-owned initial density. In this mode the
        // electronic pointers are opaque and may be null: no contraction,
        // electronic allocation, host scan, upload or download is performed.
        if(mode==2) {
            counts.topology_prepares++;
            counts.seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            return 0;
        }
        size_t grids=size_t(np)*nsp*ng*sizeof(double),sparse=size_t(nnz)*nh*sizeof(double);
        // Geometry was validated above while holding the adapter lock. The
        // public borrower performs its own lock/validation before capture.
        remag::GridContractionView view{{no,np,nsp,nuo,nnz,cells},static_cast<const int*>(plan.ends.p),
            static_cast<const int*>(plan.orbs.p),static_cast<const int*>(plan.translations.p),
            static_cast<const int*>(plan.lookup.p),(plan.device_phi?plan.device_phi:static_cast<const float*>(plan.phi.p)),nt};
        bool retained_output=false;
        bool retained_matrix=false;
        if(mode==0) {
            auto dm=remag_resident_density(input,sparse);
            if(dm) { counts.resident_hits++;counts.avoided_h2d+=sparse; }
            else { plan.input.upload(input,sparse,remag::Transfer::grid_density);dm=static_cast<double*>(plan.input.p); }
            double* target=remag_flow_density_target(output,grids);
            retained_output=target!=nullptr;
            if(!target) {plan.output.reserve(grids);target=static_cast<double*>(plan.output.p);}
            check(remag::contract_density_device(view,ng,nh,dm,target,nullptr));
        } else {
            const double* v=remag_flow_potential(input,grids);
            const double* device_dvol=nullptr;
            if(v) {device_dvol=remag_flow_dvol();counts.retained_grid_potential++;counts.avoided_grid_upload+=grids;}
            else {plan.input.upload(input,grids,remag::Transfer::grid_potential);v=static_cast<const double*>(plan.input.p);}
            double* target=remag_state_grid_target(output,sparse);
            retained_matrix=target!=nullptr;
            if(!target) {plan.output.upload(output,sparse,remag::Transfer::grid_matrix);target=static_cast<double*>(plan.output.p);}
            check(remag::add_grid_potential_device(view,ng,nh,v,device_dvol?0.:dvol,device_dvol,target,nullptr));
        }
        check(cudaGetLastError());
        size_t bytes=mode==0?grids:sparse;
        if(retained_output) {
            remag_flow_density_ready(output,bytes);counts.retained_grid_density++;counts.avoided_grid_download+=bytes;
        } else if(retained_matrix) {
            remag_state_grid_ready(output,bytes);
        } else {
            check(cudaMemcpy(output,plan.output.p,bytes,cudaMemcpyDeviceToHost));counts.d2h+=bytes;
            remag::transfer(remag::Transfer::grid_output,false,bytes);
        }
        counts.calls++;counts.launches++;
        counts.seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"REMAG grid failure: %s\n",e.what());return -116;
    }
}
