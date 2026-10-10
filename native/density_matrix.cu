// SPDX-License-Identifier: GPL-3.0-or-later
// Occupation-weighted sparse DM/energy-DM from retained real or spinor eigenvectors.
#include "resident.hpp"
#include "execution.hpp"
#include "density_matrix.hpp"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" int remag_state_active();

namespace {
void check(cudaError_t c) { if(c!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(c)); }
struct Counts {
    unsigned long long builds=0,calls=0,captures=0,launches=0,h2d=0,d2h=0,allocations=0,live=0,peak=0,topologies=0;
    unsigned long long normalizations=0,normalization_misses=0;
    unsigned long long device_normalizations=0,overlap_uploads=0;
    unsigned long long grouped_enqueues=0,grouped_pairs=0,unique_pairs=0;
    double seconds=0;
    ~Counts() {
        if(FILE* f=std::fopen("remag-dm.json","w")) {
            std::fprintf(f,"{\"unique_pair_host_enqueues\":%llu,\"grouped_pairs_total\":%llu,\"unique_pairs_total\":%llu,",
                grouped_enqueues,grouped_pairs,unique_pairs);
            std::fprintf(f,"\"builds\":%llu,\"calls\":%llu,\"resident_vector_captures\":%llu,"
                "\"custom_kernel_launch_count\":%llu,\"host_to_device_bytes\":%llu,\"device_to_host_bytes\":%llu,"
                "\"device_allocation_count\":%llu,\"peak_dm_device_bytes\":%llu,\"topology_uploads\":%llu,"
                "\"accumulation_enqueue_seconds\":%.9f,\"normalization_updates\":%llu,"
                "\"normalization_cache_misses\":%llu,\"device_charge_normalizations\":%llu,"
                "\"normalization_overlap_uploads\":%llu,\"library_kernel_launch_count\":null}\n",
                builds,calls,captures,launches,h2d,d2h,allocations,peak,topologies,seconds,normalizations,normalization_misses,
                device_normalizations,overlap_uploads);
            std::fclose(f);
        }
    }
} counts;
struct Buffer {
    void* p=nullptr; size_t bytes=0;
    void reserve(size_t n) {
        if(n<=bytes) return;
        if(p) { check(cudaFree(p));counts.live-=bytes;p=nullptr;bytes=0; }
        check(cudaMalloc(&p,n));bytes=n;counts.live+=n;counts.allocations++;
        counts.peak=std::max(counts.peak,counts.live);
    }
    void upload(const void* src,size_t n,remag::Transfer boundary=remag::Transfer::dm_topology) {
        reserve(n);check(cudaMemcpy(p,src,n,cudaMemcpyHostToDevice));counts.h2d+=n;
        remag::transfer(boundary,true,n);
    }
    ~Buffer() { if(p) cudaFree(p); }
};
struct Vectors { Buffer values; int n=0,nev=0; bool complex_values=false,valid=false; } vectors[3];
int requested_slot=-1;
bool requested_export=true;
struct Plan {
    bool active=false,finished=false,leased=false;
    int accumulated=0;
    int kind=0,n=0,nnz=0,nh=0,eh=0,pairs=0;
    uint64_t fingerprint=0;
    Buffer rows,cols,indices,xyz,occ,energies,d,e;
    int unique_pairs=0;
    Buffer unique_rows,unique_cols,pair_unique;
    std::vector<double> host_density,host_energy;
} plan;
struct Normalization {
    bool requested=false,completed=false;
    int rows=0,nh=0,eh=0,spinor=0;
    double charge=0.,tolerance=0.;
    const double* source=nullptr;
    uint64_t epoch=0;
    int cached_rows=0;
    const double* cached_source=nullptr;
    std::vector<double> host_overlap;
    Buffer overlap,partial,result;
} normalization;
using NormalizationResult=remag::DensityNormalizationResult;
uint64_t hash(uint64_t h,const void* p,size_t n) {
    auto b=static_cast<const unsigned char*>(p);
    for(size_t i=0;i<n;i++) { h^=b[i];h*=1099511628211ULL; }
    return h;
}
using Z=cuDoubleComplex;
// Tile both read and write directions without arithmetic or conjugation.
template<class T> __global__ void transpose_vectors(int n,int nev,const T* source,T* target) {
    __shared__ T tile[32][33];
    int row=blockIdx.x*32+threadIdx.x,state=blockIdx.y*32+threadIdx.y;
    for(int j=0;j<32;j+=8) if(row<n&&state+j<nev) tile[threadIdx.y+j][threadIdx.x]=source[size_t(state+j)*n+row];
    __syncthreads();
    row=blockIdx.x*32+threadIdx.y;state=blockIdx.y*32+threadIdx.x;
    for(int j=0;j<32;j+=8) if(row+j<n&&state<nev) target[size_t(row+j)*nev+state]=tile[threadIdx.x][threadIdx.y+j];
}
__device__ Z product(Z a,Z b) { return make_cuDoubleComplex(a.x*b.x+a.y*b.y,a.y*b.x-a.x*b.y); }
__device__ Z phase(Z a,double co,double si) { return make_cuDoubleComplex(a.x*co+a.y*si,a.y*co-a.x*si); }
__device__ double reduce(double a) {
    for(int s=16;s;s/=2) a+=__shfl_down_sync(0xffffffff,a,s);
    return a;
}
__global__ void normalize(size_t nd,size_t ne,double factor,double* d,double* e) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<nd) d[i]*=factor;
    if(i<ne) e[i]*=factor;
}
__global__ void charge_trace(int rows,int spinor,const double* d,const double* s,double* partial) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ double values[256];
    values[threadIdx.x]=i<size_t(rows)*spinor?d[i]*s[i%rows]:0.;
    __syncthreads();
    for(int stride=128;stride;stride/=2) {
        if(threadIdx.x<stride) values[threadIdx.x]+=values[threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x==0) partial[blockIdx.x]=values[0];
}
__global__ void charge_scale(int blocks,const double* partial,double target,double tolerance,NormalizationResult* result) {
    double q=0.;
    for(int i=0;i<blocks;i++) q+=partial[i];
    result->charge=q;
    result->status=(!isfinite(q)||q==0.||fabs(q/target-1.)>tolerance)?-134:0;
    result->scale=result->status?1.:target/q;
}
__global__ void normalize_owned(size_t nd,size_t ne,const NormalizationResult* result,double* d,double* e) {
    if(result->status) return;
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<nd) d[i]*=result->scale;
    if(i<ne) e[i]*=result->scale;
}

int normalize_before_export() {
    auto& v=normalization;
    if(v.rows!=plan.nnz||v.nh!=plan.nh||v.eh!=plan.eh) return -135;
    auto epoch=remag::geometry_epoch();
    bool reuse=epoch&&v.epoch==epoch&&v.rows==v.cached_rows&&v.source==v.cached_source;
    const char* verify=std::getenv("REMAG_GRID_VERIFY_EPOCH");
    if(reuse&&verify&&std::strcmp(verify,"1")==0&&
       std::memcmp(v.source,v.host_overlap.data(),size_t(v.rows)*sizeof(double))!=0)
        throw std::runtime_error("Overlap changed without geometry invalidation");
    // Unversioned callers always upload; the serial overlap is fixed per geometry.
    if(!reuse) {
        v.overlap.upload(v.source,size_t(v.rows)*sizeof(double),remag::Transfer::normalization_overlap);
        v.host_overlap.assign(v.source,v.source+v.rows);
        v.epoch=epoch;v.cached_rows=v.rows;v.cached_source=v.source;counts.overlap_uploads++;
    }
    int blocks=(size_t(v.rows)*v.spinor+255)/256;
    v.partial.reserve(size_t(blocks)*sizeof(double));v.result.reserve(sizeof(NormalizationResult));
    auto result=static_cast<NormalizationResult*>(v.result.p);
    auto d=static_cast<double*>(plan.d.p),e=static_cast<double*>(plan.e.p);
    check(remag::normalize_density_device({v.rows,v.nh,v.eh,v.spinor,v.charge,v.tolerance},
        static_cast<double*>(v.overlap.p),static_cast<double*>(v.partial.p),result,d,e,nullptr));
    counts.launches+=3;
    // Temporary host error boundary; the charge and scale themselves stay on device.
    int status=0;
    check(cudaMemcpy(&status,static_cast<char*>(v.result.p)+offsetof(NormalizationResult,status),sizeof(int),cudaMemcpyDeviceToHost));
    remag::transfer(remag::Transfer::normalization_status,false,sizeof(int));counts.d2h+=sizeof(int);
    if(status) return status;
    v.requested=false;v.completed=true;counts.device_normalizations++;return 0;
}

// One warp per sparse orbital pair; transposed eigenvectors make state loads contiguous.
template<int Mode> __global__ void build(int pairs,int nnz,int nh,int nev,int outspin,
        const int* rows,const int* cols,const int* indices,const double* xyz,
        const void* values,const double* occ,const double* energies,double cutoff,
        double kx,double ky,double kz,double* d,double* e) {
    int lane=threadIdx.x%32, pair=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(pair>=pairs) return;
    int i=rows[pair],j=cols[pair],ind=indices[pair];
    double angle=kx*xyz[3*pair]+ky*xyz[3*pair+1]+kz*xyz[3*pair+2];
    double co=cos(angle),si=sin(angle),ds[8]={},es[4]={};
    for(int state=lane;state<nev;state+=32) {
        double q=occ[state];
        if(fabs(q)<=cutoff) continue;
        double v[8]={};
        if constexpr(Mode==0) {
            auto z=static_cast<const double*>(values);
            v[0]=z[(size_t)i*nev+state]*z[(size_t)j*nev+state];
        } else if constexpr(Mode==1) {
            auto z=static_cast<const Z*>(values);
            v[0]=phase(product(z[(size_t)i*nev+state],z[(size_t)j*nev+state]),co,si).x;
        } else {
            auto z=static_cast<const Z*>(values);
            Z iu=z[(size_t)(2*i)*nev+state],id=z[(size_t)(2*i+1)*nev+state];
            Z ju=z[(size_t)(2*j)*nev+state],jd=z[(size_t)(2*j+1)*nev+state];
            Z a=phase(product(iu,ju),co,si),b=phase(product(id,jd),co,si);
            Z c=phase(product(iu,jd),co,si),f=phase(product(id,ju),co,si);
            if(nh==4) c=make_cuDoubleComplex(.5*(c.x+f.x),.5*(c.y-f.y));
            v[0]=a.x;v[1]=b.x;v[2]=c.x;v[3]=-c.y;
            v[4]=a.y;v[5]=b.y;v[6]=f.x;v[7]=f.y;
        }
        constexpr int nd=Mode==2?8:1,ne=Mode==2?4:1;
        for(int s=0;s<nd;s++) ds[s]+=q*v[s];
        for(int s=0;s<ne;s++) es[s]+=(q*energies[state])*v[s];
    }
    int nd=Mode==2?nh:1,ne=Mode==2?4:1;
    for(int s=0;s<nd;s++) { double v=reduce(ds[s]);if(lane==0) d[(size_t)(s+outspin)*nnz+ind]+=v; }
    for(int s=0;s<ne;s++) { double v=reduce(es[s]);if(lane==0) e[(size_t)(s+outspin)*nnz+ind]+=v; }
}
#include "density_unique_kernels.inc"
} // namespace

namespace remag {
namespace {
bool valid_shape(const DensityMatrixShape& p) {
    return p.kind>=0&&p.kind<=3&&p.n>0&&p.nnz>0&&p.pairs>0&&p.pairs<=p.nnz&&
        ((p.kind<2&&p.nh>=1&&p.nh<=2&&p.eh==p.nh)||
         (p.kind>=2&&(p.nh==4||p.nh==8)&&p.eh==4));
}
}
DensityMatrixView density_matrix_view() {
    if(plan.leased||!plan.active) throw std::runtime_error("Density geometry has no available legacy owner");
    return {{plan.kind,plan.n,plan.nnz,plan.nh,plan.eh,plan.pairs},
        static_cast<const int*>(plan.rows.p),static_cast<const int*>(plan.cols.p),
        static_cast<const int*>(plan.indices.p),static_cast<const double*>(plan.xyz.p),
        plan.unique_pairs,static_cast<const int*>(plan.unique_rows.p),
        static_cast<const int*>(plan.unique_cols.p),static_cast<const int*>(plan.pair_unique.p)};
}
DensityMatrixView lease_density_matrix() {
    if(plan.accumulated||requested_slot>=0||normalization.requested||normalization.completed||
       std::any_of(std::begin(vectors),std::end(vectors),[](const Vectors& v){return v.valid;}))
        throw std::runtime_error("Cannot lease density geometry with pending legacy work");
    auto v=density_matrix_view();plan.leased=true;return v;
}
void release_density_matrix() {
    if(!plan.leased) throw std::runtime_error("Density geometry has no cycle lease");
    plan.leased=false;plan.active=false;plan.finished=false;plan.host_density.clear();plan.host_energy.clear();
}
cudaError_t clear_density_device(const DensityMatrixShape& p,double* d,double* e,cudaStream_t stream) {
    if(!valid_shape(p)||!d||!e||d==e) return cudaErrorInvalidValue;
    auto status=cudaMemsetAsync(d,0,size_t(p.nnz)*p.nh*sizeof(double),stream);
    if(status!=cudaSuccess) return status;
    return cudaMemsetAsync(e,0,size_t(p.nnz)*p.eh*sizeof(double),stream);
}
cudaError_t transpose_vectors_device(int n,int nev,bool complex_values,const void* source,void* target,cudaStream_t stream) {
    if(n<1||nev<1||nev>n||!source||!target||source==target) return cudaErrorInvalidValue;
    dim3 block(32,8),grid((n-1)/32+1,(nev-1)/32+1);
    if(complex_values) transpose_vectors<<<grid,block,0,stream>>>(n,nev,static_cast<const Z*>(source),static_cast<Z*>(target));
    else transpose_vectors<<<grid,block,0,stream>>>(n,nev,static_cast<const double*>(source),static_cast<double*>(target));
    return cudaGetLastError();
}
cudaError_t accumulate_density_device(const DensityMatrixView& v,int nev,int outspin,
        const void* values,const double* q,const double* energies,double cutoff,
        double kx,double ky,double kz,double* d,double* e,cudaStream_t stream,double* unique_scratch) {
    const auto& p=v.shape;
    if(!valid_shape(p)||nev<1||size_t(nev)>size_t(p.n)*(p.kind>=2?2:1)||
       outspin<0||outspin>=(p.kind<2?p.nh:1)||!v.rows||!v.cols||!v.indices||!v.xyz||
       !values||!q||!energies||!d||!e||d==e||!std::isfinite(cutoff)||cutoff<0.||
       !std::isfinite(kx)||!std::isfinite(ky)||!std::isfinite(kz)) return cudaErrorInvalidValue;
    if(unique_scratch) {
        if(v.unique_pairs<1||v.unique_pairs>p.pairs||!v.unique_rows||!v.unique_cols||!v.pair_unique)
            return cudaErrorInvalidValue;
        const int blocks=(v.unique_pairs+3)/4,scatter=(p.pairs+255)/256;
        if(p.kind==0)contract_unique<0><<<blocks,128,0,stream>>>(v.unique_pairs,nev,v.unique_rows,v.unique_cols,values,q,energies,cutoff,unique_scratch);
        else if(p.kind==1)contract_unique<1><<<blocks,128,0,stream>>>(v.unique_pairs,nev,v.unique_rows,v.unique_cols,values,q,energies,cutoff,unique_scratch);
        else contract_unique<2><<<blocks,128,0,stream>>>(v.unique_pairs,nev,v.unique_rows,v.unique_cols,values,q,energies,cutoff,unique_scratch);
        auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
        if(p.kind==0)scatter_unique<0><<<scatter,256,0,stream>>>(p.pairs,v.unique_pairs,p.nnz,p.nh,outspin,v.indices,v.pair_unique,v.xyz,unique_scratch,0.,0.,0.,d,e);
        else if(p.kind==1)scatter_unique<1><<<scatter,256,0,stream>>>(p.pairs,v.unique_pairs,p.nnz,p.nh,outspin,v.indices,v.pair_unique,v.xyz,unique_scratch,kx,ky,kz,d,e);
        else scatter_unique<2><<<scatter,256,0,stream>>>(p.pairs,v.unique_pairs,p.nnz,p.nh,0,v.indices,v.pair_unique,v.xyz,unique_scratch,kx,ky,kz,d,e);
        counts.grouped_enqueues++; // Enqueue/capture calls, not executed SCF iterations.
        return cudaGetLastError();
    }
    int blocks=(p.pairs-1)/4+1;
    if(p.kind==0) build<0><<<blocks,128,0,stream>>>(p.pairs,p.nnz,p.nh,nev,outspin,v.rows,v.cols,v.indices,v.xyz,values,q,energies,cutoff,0.,0.,0.,d,e);
    else if(p.kind==1) build<1><<<blocks,128,0,stream>>>(p.pairs,p.nnz,p.nh,nev,outspin,v.rows,v.cols,v.indices,v.xyz,values,q,energies,cutoff,kx,ky,kz,d,e);
    else build<2><<<blocks,128,0,stream>>>(p.pairs,p.nnz,p.nh,nev,0,v.rows,v.cols,v.indices,v.xyz,values,q,energies,cutoff,kx,ky,kz,d,e);
    return cudaGetLastError();
}
cudaError_t normalize_density_device(const DensityNormalizationConfig& p,const double* overlap,
        double* partial,DensityNormalizationResult* result,double* d,double* e,cudaStream_t stream) {
    if(p.rows<1||p.nh<1||p.eh<1||p.spinor<1||p.spinor>2||p.spinor>p.nh||
       !std::isfinite(p.charge)||p.charge<=0.||!std::isfinite(p.tolerance)||p.tolerance<0.||
       !overlap||!partial||!result||!d||!e||d==e) return cudaErrorInvalidValue;
    int blocks=(size_t(p.rows)*p.spinor+255)/256;
    charge_trace<<<blocks,256,0,stream>>>(p.rows,p.spinor,d,overlap,partial);
    auto status=cudaGetLastError();if(status!=cudaSuccess) return status;
    charge_scale<<<1,1,0,stream>>>(blocks,partial,p.charge,p.tolerance,result);
    status=cudaGetLastError();if(status!=cudaSuccess) return status;
    size_t nd=size_t(p.rows)*p.nh,ne=size_t(p.rows)*p.eh;
    normalize_owned<<<(std::max(nd,ne)+255)/256,256,0,stream>>>(nd,ne,result,d,e);
    return cudaGetLastError();
}
}

extern "C" int remag_dm_enabled() {
    const char* v=std::getenv("REMAG_DM");return v && std::strcmp(v,"1")==0;
}
extern "C" int remag_dm_normalization_enabled() {
    const char* v=std::getenv("REMAG_NORMALIZE");return v&&std::strcmp(v,"1")==0&&remag_dm_enabled();
}
extern "C" int remag_dm_normalization_arm(int rows,int nh,int eh,int spinor,const double* s,double charge,double tolerance) {
    auto& v=normalization;
    if(plan.leased||v.requested||v.completed||rows<1||spinor<1||spinor>2||spinor>nh||!s||
       !std::isfinite(charge)||charge<=0.||!std::isfinite(tolerance)||tolerance<0.) return -136;
    v.requested=true;v.rows=rows;v.nh=nh;v.eh=eh;v.spinor=spinor;v.source=s;v.charge=charge;v.tolerance=tolerance;
    return 0;
}
extern "C" int remag_dm_normalization_complete() {
    auto& v=normalization;
    if(plan.leased) return -136;
    bool completed=v.completed;
    // Unsupported upstream density builders did not consume the request. Preserve
    // their original CPU normalization; never leave a request for a later call.
    v.completed=false;v.requested=false;return completed?1:0;
}
extern "C" int remag_arm_vectors(int slot,int export_host) {
    if(plan.leased) return -129;
    if(slot<0||slot>=3||requested_slot>=0) return -120;
    requested_slot=slot;requested_export=export_host!=0;vectors[slot].valid=false;return 0;
}
bool remag_vector_request(int& slot,bool& export_host) {
    slot=requested_slot;export_host=requested_export;requested_slot=-1;requested_export=true;return slot>=0;
}
void remag_capture_vectors(int slot,int n,int nev,bool complex_values,const void* source) {
    if(plan.leased) throw std::runtime_error("Legacy vector capture cannot enter a cycle lease");
    if(slot<0||slot>=3||nev<1||nev>n) throw std::runtime_error("Invalid resident eigenvector request");
    auto& v=vectors[slot];v.valid=false;
    v.values.reserve(size_t(n)*nev*(complex_values?sizeof(Z):sizeof(double)));
    check(remag::transpose_vectors_device(n,nev,complex_values,source,v.values.p,nullptr));
    v.n=n;v.nev=nev;v.complex_values=complex_values;v.valid=true;counts.captures++;counts.launches++;
}
const double* remag_resident_density(const double* host,size_t bytes) {
    if(plan.leased) throw std::runtime_error("Legacy density consumer cannot enter a cycle lease");
    if(const double* owned=remag_state_density(host,bytes)) return owned;
    if(!plan.finished||bytes!=plan.host_density.size()*sizeof(double)) return nullptr;
    // The upstream mixer may have changed or transformed the DM: reuse only exact content.
    return std::memcmp(host,plan.host_density.data(),bytes)==0?static_cast<double*>(plan.d.p):nullptr;
}

extern "C" int remag_dm_normalize(int rows,int nh,int eh,double factor,const double* d,const double* e) {
    try {
        if(plan.leased) return -133;
        if(!plan.finished||rows!=plan.nnz||nh!=plan.nh||eh!=plan.eh||!std::isfinite(factor)||
           plan.host_density.size()!=size_t(plan.nnz)*plan.nh||
           plan.host_energy.size()!=size_t(plan.nnz)*plan.eh) {
            plan.finished=false;counts.normalization_misses++;return 0;
        }
        // Verify the only host-side change was the upstream scalar normalization.
        // Mixing, reordered layouts and any other modifications invalidate residency.
        for(size_t i=0;i<plan.host_density.size();i++) {
            double scaled=plan.host_density[i]*factor;
            if(std::memcmp(&scaled,d+i,sizeof(double))!=0) { plan.finished=false;counts.normalization_misses++;return 0; }
        }
        for(size_t i=0;i<plan.host_energy.size();i++) {
            double scaled=plan.host_energy[i]*factor;
            if(std::memcmp(&scaled,e+i,sizeof(double))!=0) { plan.finished=false;counts.normalization_misses++;return 0; }
        }
        size_t nd=plan.host_density.size(),ne=plan.host_energy.size();
        normalize<<<(std::max(nd,ne)+255)/256,256>>>(nd,ne,factor,static_cast<double*>(plan.d.p),static_cast<double*>(plan.e.p));
        check(cudaGetLastError());check(cudaDeviceSynchronize());counts.launches++;counts.normalizations++;
        plan.host_density.assign(d,d+nd);plan.host_energy.assign(e,e+ne);return 0;
    } catch(const std::exception& ex) { std::fprintf(stderr,"REMAG DM normalize: %s\n",ex.what());return -133; }
}

extern "C" int remag_dm_begin(int kind,int n,int nnz,int nh,int eh,int no,
        const int* num,const int* ptr,const int* cols,const int* mapping,const double* xyz) {
    try {
        if(plan.active||kind<0||kind>3||n<1||nnz<1) return -121;
        if((kind<2 && (nh<1||nh>2||eh!=nh)) || (kind>=2 && ((nh!=4&&nh!=8)||eh!=4))) return -122;
        int visible=0;check(cudaGetDeviceCount(&visible));if(visible!=1) return -123;
        plan.finished=false;
        const char* option=std::getenv("REMAG_DENSITY_UNIQUE_PAIRS");
        if(option&&std::strcmp(option,"0")&&std::strcmp(option,"1"))return -124;
        const bool grouping=!option||std::strcmp(option,"1")==0;
        std::vector<int> rows,columns,indices,unique_rows,unique_cols,pair_unique;
        std::vector<int> lookup(grouping?n:0,-1);
        std::vector<double> coords;
        for(int row=0;row<n;row++) {
            if(grouping)std::fill(lookup.begin(),lookup.end(),-1);
            if(num[row]<0||ptr[row]<0||size_t(ptr[row])+num[row]>size_t(nnz)) return -124;
            for(int p=ptr[row];p<ptr[row]+num[row];p++) {
                if(cols[p]<1) return -125;
                int col=(cols[p]-1)%n;
                if(kind%2) {
                    if(cols[p]>no) return -125;
                    col=mapping[cols[p]-1]-1;if(col<0||col>=n) return -125;
                }
                if(grouping) {
                    if(lookup[col]<0) {
                        lookup[col]=int(unique_rows.size());unique_rows.push_back(row);unique_cols.push_back(col);
                    }
                    pair_unique.push_back(lookup[col]);
                }
                rows.push_back(row);columns.push_back(col);indices.push_back(p);
                for(int x=0;x<3;x++) coords.push_back(kind%2?xyz[3*p+x]:0.);
            }
        }
        if(rows.empty()) return -126;
        uint64_t h=14695981039346656037ULL;
        int dims[]={kind,n,nnz,nh,eh,no,int(grouping)};h=hash(h,dims,sizeof(dims));
        h=hash(h,rows.data(),rows.size()*sizeof(int));h=hash(h,columns.data(),columns.size()*sizeof(int));
        h=hash(h,indices.data(),indices.size()*sizeof(int));h=hash(h,coords.data(),coords.size()*sizeof(double));
        if(h!=plan.fingerprint) {
            plan.rows.upload(rows.data(),rows.size()*sizeof(int));plan.cols.upload(columns.data(),columns.size()*sizeof(int));
            plan.indices.upload(indices.data(),indices.size()*sizeof(int));plan.xyz.upload(coords.data(),coords.size()*sizeof(double));
            plan.unique_pairs=0;
            if(grouping&&unique_rows.size()<rows.size()) {
                plan.unique_rows.upload(unique_rows.data(),unique_rows.size()*sizeof(int));
                plan.unique_cols.upload(unique_cols.data(),unique_cols.size()*sizeof(int));
                plan.pair_unique.upload(pair_unique.data(),pair_unique.size()*sizeof(int));
                plan.unique_pairs=int(unique_rows.size());
            }
            plan.fingerprint=h;counts.topologies++;
        }
        if(plan.unique_pairs) {counts.grouped_pairs+=rows.size();counts.unique_pairs+=plan.unique_pairs;}
        plan.kind=kind;plan.n=n;plan.nnz=nnz;plan.nh=nh;plan.eh=eh;plan.pairs=int(rows.size());
        plan.d.reserve(size_t(nnz)*nh*sizeof(double));plan.e.reserve(size_t(nnz)*eh*sizeof(double));
        check(remag::clear_density_device({kind,n,nnz,nh,eh,plan.pairs},static_cast<double*>(plan.d.p),static_cast<double*>(plan.e.p),nullptr));
        plan.active=true;plan.accumulated=0;counts.builds++;return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"REMAG DM begin: %s\n",e.what());return -127; }
}
extern "C" int remag_dm_accumulate(int slot,int nev,int outspin,const double* occ,
        const double* energies,double cutoff,const double* k) {
    auto start=std::chrono::steady_clock::now();
    try {
        if(plan.leased||!plan.active||slot<0||slot>=3||nev<0||outspin<0||outspin>=plan.nh) return -128;
        if(plan.kind>=2 && outspin!=0) return -128;
        if(nev==0) return 0;
        auto& v=vectors[slot];
        if(!v.valid||v.n!=plan.n*(plan.kind>=2?2:1)||v.nev!=nev||v.complex_values!=(plan.kind!=0)) return -129;
        v.valid=false;
        const double *q=nullptr,*ee=nullptr;
        if(!remag_retained_occupations(occ,energies,nev,q,ee)) {
            plan.occ.upload(occ,size_t(nev)*sizeof(double),remag::Transfer::dm_occupations);
            plan.energies.upload(energies,size_t(nev)*sizeof(double),remag::Transfer::dm_energies);
            q=static_cast<const double*>(plan.occ.p);ee=static_cast<const double*>(plan.energies.p);
        }
        auto d=static_cast<double*>(plan.d.p),e=static_cast<double*>(plan.e.p);
        check(remag::accumulate_density_device(remag::density_matrix_view(),nev,outspin,v.values.p,q,ee,cutoff,
            plan.kind==0?0.:k[0],plan.kind==0?0.:k[1],plan.kind==0?0.:k[2],d,e,nullptr));
        // The producer, accumulation, normalization and next consumers share
        // the default stream in this adapter. No global barrier is required.
        plan.accumulated++;counts.calls++;counts.launches++;
        counts.seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"REMAG DM accumulate: %s\n",e.what());return -130; }
}
extern "C" int remag_dm_finish(double* d,double* e) {
    try {
        if(plan.leased||!plan.active) return -131;
        if(normalization.requested) {
            int status=normalize_before_export();
            if(status) return status;
        }
        size_t nd=size_t(plan.nnz)*plan.nh,ne=size_t(plan.nnz)*plan.eh;
        if(remag_state_active()&&!normalization.completed) throw std::runtime_error("Owned density was not normalized");
        if(remag_state_publish_density(d,e,static_cast<const double*>(plan.d.p),static_cast<const double*>(plan.e.p),
                                       nd*sizeof(double),ne*sizeof(double))) {
            // The sparse owner is now the only valid consumer. The legacy
            // host-content cache must not appear reusable during the next
            // geometry's CPU normalization with empty snapshot vectors.
            plan.host_density.clear();plan.host_energy.clear();plan.active=false;plan.finished=false;return 0;
        }
        check(cudaMemcpy(d,plan.d.p,nd*sizeof(double),cudaMemcpyDeviceToHost));
        remag::transfer(remag::Transfer::dm_outputs,false,nd*sizeof(double));
        check(cudaMemcpy(e,plan.e.p,ne*sizeof(double),cudaMemcpyDeviceToHost));counts.d2h+=(nd+ne)*sizeof(double);
        remag::transfer(remag::Transfer::dm_outputs,false,ne*sizeof(double));
        plan.host_density.assign(d,d+nd);plan.host_energy.assign(e,e+ne);plan.active=false;plan.finished=true;return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"REMAG DM finish: %s\n",e.what());return -132; }
}
