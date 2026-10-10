// SPDX-License-Identifier: GPL-3.0-or-later
// Deterministic sparse -> Gamma/k-space assembly for serial SIESTA diagonalization.
#include "resident.hpp"
#include "execution.hpp"
#include "hamiltonian.hpp"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" int remag_gpu_enabled();
namespace {
void check(cudaError_t code) { if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
bool verify() { const char* p=std::getenv("REMAG_GRID_VERIFY_EPOCH");return p&&std::strcmp(p,"1")==0; }
struct Counts {
    unsigned long long prepares=0,ends=0,builds=0,setups=0,h2d=0,avoided=0,checked=0,launches=0;
    double seconds=0.;
    ~Counts() {
        if(FILE* f=std::fopen("remag-hs.json","w")) {
            std::fprintf(f,"{\"prepares\":%llu,\"ends\":%llu,\"matrix_builds\":%llu,\"geometry_uploads\":%llu,"
                "\"host_to_device_bytes\":%llu,\"device_to_host_bytes\":0,\"avoided_dense_upload_bytes\":%llu,"
                "\"debug_host_validation_bytes\":%llu,\"kernel_launch_count\":%llu,\"assembly_dispatch_seconds\":%.9f,"
                "\"scope\":\"sparse H upload each diagonalization; geometry-owned S/topology/k data; no matrix downloads\"}\n",
                prepares,ends,builds,setups,h2d,avoided,checked,launches,seconds);
            std::fclose(f);
        }
    }
} counts;
struct Buffer {
    void* p=nullptr;size_t capacity=0;
    void reserve(size_t bytes) {
        if(bytes<=capacity) return;
        if(p) {check(cudaFree(p));p=nullptr;capacity=0;}
        check(cudaMalloc(&p,bytes));capacity=bytes;
    }
    void upload(const void* source,size_t bytes,remag::Transfer boundary) {
        reserve(bytes);
        if(bytes) {check(cudaMemcpy(p,source,bytes,cudaMemcpyHostToDevice));counts.h2d+=bytes;remag::transfer(boundary,true,bytes);}
    }
    ~Buffer() {if(p) cudaFree(p);}
};
struct Snapshot {
    std::vector<unsigned char> bytes;
    void save(const void* source,size_t size) {
        bytes.resize(size);if(size) std::memcpy(bytes.data(),source,size);
    }
    bool matches(const void* source) const {
        counts.checked+=bytes.size();return bytes.empty()||std::memcmp(bytes.data(),source,bytes.size())==0;
    }
};
using Config=remag::HamiltonianAssemblyConfig;
using Pair=remag::HamiltonianPair;
struct Entry {unsigned long long key;int index;};
struct State {
    bool active=false,configured=false,pending=false,leased=false,geometry_only=false;
    uint64_t epoch=0;
    Config config{};
    // num, ptr, cols, mapping, xij, k, S, H: H can change between calls.
    std::array<const void*,8> source{};
    std::array<Snapshot,8> snapshots;
    const void *host_h=nullptr,*host_s=nullptr;
    const double* borrowed_h=nullptr;
    int spin=0,k=0;
    std::vector<int> completed;
    Buffer device_config,pairs,indices,xij,kpoints,overlap,hamiltonian;
} state;
bool shape_equal(const Config& a,const Config& b) {
    return a.kind==b.kind&&a.n==b.n&&a.nnz==b.nnz&&a.nh==b.nh&&a.no==b.no&&a.nk==b.nk&&a.ldh==b.ldh;
}
int spin_blocks(const Config& p) {return p.kind<2?p.nh:1;}
bool invariant_matches() {
    for(size_t i=0;i<7;i++) if(!state.snapshots[i].matches(state.source[i])) return false;
    return true;
}

// One thread owns a dense orbital pair. Grouped indices retain upstream's
// original within-row order, including duplicate periodic images. No FP atomics.
__global__ void assemble_real(const Config* input,const Pair* pairs,const int* indices,
                             const double* hs,const double* ss,int spin,double* h,double* s) {
    const Config p=*input;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p.pairs) return;
    Pair pair=pairs[i];double hv=0.,sv=0.;
    for(int j=pair.begin;j<pair.end;j++) {
        int index=indices[j];sv+=ss[index];hv+=hs[index+size_t(p.ldh)*spin];
    }
    size_t out=pair.row+size_t(p.n)*pair.col;h[out]=hv;s[out]=sv;
}
__global__ void assemble_complex(const Config* input,const Pair* pairs,const int* indices,
        const double* hs,const double* ss,const double* xij,const double* kpoints,int spin,int k,
        cuDoubleComplex* h,cuDoubleComplex* s) {
    const Config p=*input;int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p.pairs) return;
    Pair pair=pairs[i];double sr=0.,si=0.,hr[4]={0.,0.,0.,0.},hi[4]={0.,0.,0.,0.};
    for(int j=pair.begin;j<pair.end;j++) {
        int index=indices[j];double c=1.,t=0.;
        if(p.kind%2) {
            double angle=kpoints[3*k]*xij[3*size_t(index)]+kpoints[3*k+1]*xij[3*size_t(index)+1]+
                         kpoints[3*k+2]*xij[3*size_t(index)+2];
            c=cos(angle);t=-sin(angle);
        }
        sr+=ss[index]*c;si+=ss[index]*t;
        if(p.kind<2) {
            double value=hs[index+size_t(p.ldh)*spin];hr[0]+=value*c;hi[0]+=value*t;
        } else {
            double a[4]={hs[index],hs[index+size_t(p.ldh)],hs[index+2*size_t(p.ldh)],
                         hs[index+size_t(p.ldh)*(p.nh==8?6:2)]};
            double b[4]={p.nh==8?hs[index+4*size_t(p.ldh)]:0.,p.nh==8?hs[index+5*size_t(p.ldh)]:0.,
                         -hs[index+3*size_t(p.ldh)],hs[index+size_t(p.ldh)*(p.nh==8?7:3)]};
            for(int block=0;block<4;block++) {
                hr[block]+=a[block]*c-b[block]*t;hi[block]+=a[block]*t+b[block]*c;
            }
        }
    }
    if(p.kind<2) {
        size_t out=pair.row+size_t(p.n)*pair.col;
        h[out]=make_cuDoubleComplex(hr[0],hi[0]);s[out]=make_cuDoubleComplex(sr,si);
    } else {
        size_t dim=2*size_t(p.n),base=2*size_t(pair.row)+dim*2*pair.col;
        h[base]=make_cuDoubleComplex(hr[0],hi[0]);h[base+1+dim]=make_cuDoubleComplex(hr[1],hi[1]);
        h[base+dim]=make_cuDoubleComplex(hr[2],hi[2]);h[base+1]=make_cuDoubleComplex(hr[3],hi[3]);
        s[base]=s[base+1+dim]=make_cuDoubleComplex(sr,si);
    }
}
}

namespace remag {
HamiltonianAssemblyView hamiltonian_assembly_view() {
    if(state.leased||!state.active||state.pending||state.epoch!=geometry_epoch())
        throw std::runtime_error("Sparse assembly view requires an active, unarmed geometry owner");
    if(verify()&&(!invariant_matches()||(!state.geometry_only&&!state.snapshots[7].matches(state.source[7]))))
        throw std::runtime_error("Sparse assembly view inputs changed inside their lifetime");
    return {state.config,static_cast<const Config*>(state.device_config.p),static_cast<const Pair*>(state.pairs.p),
        static_cast<const int*>(state.indices.p),static_cast<const double*>(state.overlap.p),
        static_cast<const double*>(state.xij.p),static_cast<const double*>(state.kpoints.p),
        state.geometry_only?nullptr:(state.borrowed_h?state.borrowed_h:static_cast<const double*>(state.hamiltonian.p))};
}
HamiltonianAssemblyView lease_hamiltonian_assembly() {
    if(std::any_of(state.completed.begin(),state.completed.end(),[](int n){return n!=0;}))
        throw std::runtime_error("Cannot lease H/S geometry after a legacy solve");
    auto v=hamiltonian_assembly_view();state.leased=true;return v;
}
void release_hamiltonian_assembly() {
    if(!state.leased) throw std::runtime_error("H/S geometry has no cycle lease");
    state.leased=false;state.active=false;state.pending=false;state.completed.clear();state.borrowed_h=nullptr;
}
cudaError_t assemble_hamiltonian_device(const HamiltonianAssemblyView& v,const double* h,
        int spin,int k,void* dense_h,void* dense_s,cudaStream_t stream) {
    const auto& p=v.shape;
    if(p.kind<0||p.kind>3||p.n<1||p.n>INT_MAX/2||p.pairs<1||p.pairs>INT_MAX-127||p.nnz<1||p.ldh<p.nnz||
       p.nk<1||(!(p.kind%2)&&p.nk!=1)||(p.kind<2?(p.nh!=1&&p.nh!=2):(p.nh!=4&&p.nh!=8))||
       spin<0||spin>=spin_blocks(p)||k<0||k>=p.nk||!v.config||!v.pairs||!v.indices||!v.overlap||
       (p.kind%2&&(!v.xij||!v.kpoints))||!h||!dense_h||!dense_s||dense_h==dense_s||
       dense_h==h||dense_s==h||dense_h==v.overlap||dense_s==v.overlap) return cudaErrorInvalidValue;
    const size_t n=size_t(p.n)*(p.kind>=2?2:1),item=p.kind?sizeof(cuDoubleComplex):sizeof(double);
    if(n>SIZE_MAX/n/item) return cudaErrorInvalidValue;
    const size_t bytes=n*n*item;
    auto status=cudaMemsetAsync(dense_h,0,bytes,stream);if(status!=cudaSuccess) return status;
    status=cudaMemsetAsync(dense_s,0,bytes,stream);if(status!=cudaSuccess) return status;
    const int blocks=(p.pairs+127)/128;
    if(p.kind) assemble_complex<<<blocks,128,0,stream>>>(v.config,v.pairs,v.indices,h,v.overlap,v.xij,v.kpoints,spin,k,
        static_cast<cuDoubleComplex*>(dense_h),static_cast<cuDoubleComplex*>(dense_s));
    else assemble_real<<<blocks,128,0,stream>>>(v.config,v.pairs,v.indices,h,v.overlap,spin,
        static_cast<double*>(dense_h),static_cast<double*>(dense_s));
    return cudaGetLastError();
}
}

extern "C" int remag_hs_enabled() {
    const char* p=std::getenv("REMAG_HS");return p&&std::strcmp(p,"1")==0;
}
extern "C" int remag_hs_armed() {return state.active?1:0;}
static int prepare_hs(bool geometry_only,int kind,int n,int nnz,int nh,int no,int nk,int ldh,
        const int* num,const int* ptr,const int* cols,const int* mapping,const double* xij,
        const double* kpoints,const double* h,const double* s) {
    try {
        if(!remag_hs_enabled()||!remag_gpu_enabled()||state.active||kind<0||kind>3||n<1||n>INT_MAX/2||
           nnz<1||ldh<nnz||no<n||nk<1||(!(kind%2)&&nk!=1)||
           (kind<2?(nh!=1&&nh!=2):(nh!=4&&nh!=8))||!num||!ptr||!cols||(!geometry_only&&!h)||!s||
           (kind%2&&(!mapping||!xij||!kpoints))||remag::geometry_epoch()==0) return -170;
        Config p{kind,n,nnz,nh,no,nk,ldh,0};
        std::array<const void*,8> sources{num,ptr,cols,kind%2?mapping:nullptr,kind%2?xij:nullptr,
                                        kind%2?kpoints:nullptr,s,h};
        bool reuse=state.configured&&shape_equal(state.config,p)&&state.epoch==remag::geometry_epoch()&&
                   std::equal(sources.begin(),sources.begin()+7,state.source.begin());
        if(reuse&&verify()&&!invariant_matches()) return -171;
        if(!reuse) {
            std::vector<Entry> entries;
            size_t used=0;
            for(int col=0;col<n;col++) {
                if(num[col]<0||ptr[col]<0||static_cast<long long>(ptr[col])+num[col]>nnz) return -172;
                used+=num[col];if(used>INT_MAX) return -172;
                for(int j=0;j<num[col];j++) {
                    int index=ptr[col]+j,orb=cols[index];
                    if(orb<1||(kind%2&&orb>no)||!std::isfinite(s[index])) return -172;
                    int row=kind%2?mapping[orb-1]-1:(orb-1)%n;
                    if(row<0||row>=n) return -172;
                    if(kind%2) for(int a=0;a<3;a++) if(!std::isfinite(xij[3*size_t(index)+a])) return -172;
                    entries.push_back({static_cast<unsigned long long>(col)*n+row,index});
                }
            }
            if(entries.empty()) return -172;
            if(kind%2) for(size_t i=0;i<3*size_t(nk);i++) if(!std::isfinite(kpoints[i])) return -172;
            std::stable_sort(entries.begin(),entries.end(),[](const Entry& a,const Entry& b){return a.key<b.key;});
            std::vector<Pair> pairs;std::vector<int> indices;indices.reserve(entries.size());
            for(size_t i=0;i<entries.size();) {
                size_t begin=i;auto key=entries[i].key;
                while(i<entries.size()&&entries[i].key==key) {indices.push_back(entries[i].index);i++;}
                pairs.push_back({int(key%n),int(key/n),int(begin),int(i)});
            }
            p.pairs=static_cast<int>(pairs.size());
            state.device_config.upload(&p,sizeof(p),remag::Transfer::hs_geometry);
            state.pairs.upload(pairs.data(),pairs.size()*sizeof(Pair),remag::Transfer::hs_geometry);
            state.indices.upload(indices.data(),indices.size()*sizeof(int),remag::Transfer::hs_geometry);
            state.overlap.upload(s,size_t(nnz)*sizeof(double),remag::Transfer::hs_geometry);
            if(kind%2) {
                state.xij.upload(xij,3*size_t(nnz)*sizeof(double),remag::Transfer::hs_geometry);
                state.kpoints.upload(kpoints,3*size_t(nk)*sizeof(double),remag::Transfer::hs_geometry);
            }
            std::array<size_t,7> sizes{size_t(n)*sizeof(int),size_t(n)*sizeof(int),size_t(nnz)*sizeof(int),
                kind%2?size_t(no)*sizeof(int):0,kind%2?3*size_t(nnz)*sizeof(double):0,
                kind%2?3*size_t(nk)*sizeof(double):0,size_t(nnz)*sizeof(double)};
            for(size_t i=0;i<7;i++) state.snapshots[i].save(sources[i],sizes[i]);
            state.config=p;state.epoch=remag::geometry_epoch();state.configured=true;counts.setups++;
        }
        state.source=sources;
        state.geometry_only=geometry_only;
        state.borrowed_h=nullptr;
        if(!geometry_only) {
            state.borrowed_h=remag_state_hamiltonian(h,size_t(ldh)*nh*sizeof(double));
            if(!state.borrowed_h) state.hamiltonian.upload(h,size_t(ldh)*nh*sizeof(double),remag::Transfer::hs_values);
            if(verify()) state.snapshots[7].save(h,size_t(ldh)*nh*sizeof(double));
        }
        state.completed.assign(size_t(nk)*spin_blocks(state.config),0);
        state.active=true;state.pending=false;counts.prepares++;return 0;
    } catch(const std::exception& e) {
        state.active=false;std::fprintf(stderr,"REMAG H/S prepare: %s\n",e.what());return -173;
    }
}
extern "C" int remag_hs_prepare(int kind,int n,int nnz,int nh,int no,int nk,int ldh,
        const int* num,const int* ptr,const int* cols,const int* mapping,const double* xij,
        const double* kpoints,const double* h,const double* s) {
    return prepare_hs(false,kind,n,nnz,nh,no,nk,ldh,num,ptr,cols,mapping,xij,kpoints,h,s);
}
extern "C" int remag_hs_prepare_geometry(int kind,int n,int nnz,int nh,int no,int nk,int ldh,
        const int* num,const int* ptr,const int* cols,const int* mapping,const double* xij,
        const double* kpoints,const double* s) {
    return prepare_hs(true,kind,n,nnz,nh,no,nk,ldh,num,ptr,cols,mapping,xij,kpoints,nullptr,s);
}
extern "C" int remag_hs_select(int spin,int k,const void* host_h,const void* host_s) {
    if(state.geometry_only||state.leased||!state.active||state.pending||spin<0||spin>=spin_blocks(state.config)||k<0||k>=state.config.nk||
       !host_h||!host_s||state.epoch!=remag::geometry_epoch()) return -174;
    if(state.completed[size_t(k)*spin_blocks(state.config)+spin]>=(state.config.kind%2?2:1)) return -174;
    state.spin=spin;state.k=k;state.host_h=host_h;state.host_s=host_s;state.pending=true;return 0;
}
bool remag_assemble_hs(int n,bool complex_values,const void* host_h,const void* host_s,void* h,void* s) {
    if(state.leased) throw std::runtime_error("Legacy solver cannot consume cycle-leased H/S geometry");
    if(!state.active) return false;
    auto start=std::chrono::steady_clock::now();
    bool pending=state.pending;state.pending=false;
    const auto& p=state.config;
    if(!pending||n!=p.n*(p.kind>=2?2:1)||complex_values!=(p.kind!=0)||host_h!=state.host_h||host_s!=state.host_s||
       state.epoch!=remag::geometry_epoch()) throw std::runtime_error("Dense solver does not match armed sparse H/S owner");
    auto view=remag::hamiltonian_assembly_view();
    size_t bytes=size_t(n)*n*(complex_values?sizeof(cuDoubleComplex):sizeof(double));
    check(remag::assemble_hamiltonian_device(view,view.hamiltonian,state.spin,state.k,h,s,nullptr));
    state.completed[size_t(state.k)*spin_blocks(p)+state.spin]++;counts.builds++;counts.launches++;counts.avoided+=2*bytes;
    counts.seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return true;
}
extern "C" int remag_hs_end() {
    if(state.leased||!state.active||state.pending||state.epoch!=remag::geometry_epoch()||
       std::any_of(state.completed.begin(),state.completed.end(),[](int n){return n!=(state.config.kind%2?2:1);})) return -175;
    state.active=false;counts.ends++;return 0;
}
