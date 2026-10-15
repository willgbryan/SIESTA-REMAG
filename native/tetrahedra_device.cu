// SPDX-License-Identifier: GPL-3.0-or-later
// GPU tetrahedron occupations; see tetrahedra_device.hpp. The arithmetic is
// tetra_occupations in tetrahedra.cpp, step for step.
#include "tetrahedra_device.hpp"
#include "tetrahedron_math.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace remag {
namespace {
constexpr int threads=256,search=1024;
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
// Kawamura et al. (2014) fit, as kFit in tetrahedra.cpp.
__constant__ int fit[4][20]={
    {1440,0,30,0,-38,7,17,-28,-56,9,-46,9,-38,-28,17,7,-18,-18,12,-18},
    {0,1440,0,30,-28,-38,7,17,9,-56,9,-46,7,-38,-28,17,-18,-18,-18,12},
    {30,0,1440,0,17,-28,-38,7,-46,9,-56,9,17,7,-38,-28,12,-18,-18,-18},
    {0,30,0,1440,7,17,-28,-38,9,-46,9,-56,-28,17,7,-38,-18,12,-18,-18}};
struct Shape {int method,maxspn,maxe,ne,corners,tetrahedra;size_t items;};
__device__ double energy(const Shape& s,const double* e,int band,int spin,int k) {
    return e[size_t(s.maxe)*(spin+size_t(s.maxspn)*k)+band];
}
// Sorted corner energies of every (spin, band, tetrahedron) and the
// permutation back to the corners; per-block lowest and highest energies.
__global__ void sort_items(Shape s,const int* stencil,const int* point,const double* energies,double* sorted,
                           unsigned char* order,double* bounds,FermiResult* result) {
    __shared__ double low[threads],high[threads];double lo=HUGE_VAL,hi=-HUGE_VAL;
    for(size_t item=blockIdx.x*size_t(blockDim.x)+threadIdx.x;item<s.items;item+=size_t(gridDim.x)*blockDim.x) {
        const size_t ntet=s.tetrahedra;
        const int t=int(item%ntet),band=int((item/ntet)%s.ne),spin=int(item/(ntet*s.ne));
        const int* st=stencil+size_t(t)*s.corners;double e[4];
        for(int i=0;i<4;i++) {
            if(s.method==3) {
                double sum=0.;
                for(int c=0;c<20;c++) sum+=fit[i][c]*energy(s,energies,band,spin,point[st[c]]);
                e[i]=sum/1260.;
            } else e[i]=energy(s,energies,band,spin,point[st[i]]);
            if(!isfinite(e[i])) atomicCAS(&result->status,0,tetra_bad_input);
        }
        unsigned char o[4]={0,1,2,3};
        for(int i=1;i<4;i++) for(int j=i;j>0;j--) {      // insertion sort, ties by corner index
            const int x=o[j-1],y=o[j];
            if(e[y]<e[x]||(e[y]==e[x]&&y<x)) {o[j-1]=y;o[j]=x;} else break;
        }
        for(int i=0;i<4;i++) {sorted[4*item+i]=e[o[i]];order[4*item+i]=o[i];}
        lo=fmin(lo,sorted[4*item]);hi=fmax(hi,sorted[4*item+3]);
    }
    low[threadIdx.x]=lo;high[threadIdx.x]=hi;__syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) {low[threadIdx.x]=fmin(low[threadIdx.x],low[threadIdx.x+step]);
                              high[threadIdx.x]=fmax(high[threadIdx.x],high[threadIdx.x+step]);}
        __syncthreads();
    }
    if(!threadIdx.x) {bounds[2*blockIdx.x]=low[0];bounds[2*blockIdx.x+1]=high[0];}
}
__device__ double block_sum(double value,double* scratch) {
    scratch[threadIdx.x]=value;__syncthreads();
    for(int step=search/2;step;step/=2) {if(threadIdx.x<step) scratch[threadIdx.x]+=scratch[threadIdx.x+step];__syncthreads();}
    const double total=scratch[0];__syncthreads();return total;
}
// The host's bisection, in one block of `search` threads: states(ef) sums
// tetra_count over the items in a fixed thread order.
__global__ void __launch_bounds__(search) find_fermi(Shape s,int nspinor,int blocks,const double* bounds,const double* sorted,
                           const FermiProblem* problem,double* fermi,FermiResult* result) {
    __shared__ double scratch[search];__shared__ int flag;
    const double full=nspinor==1?2.:1.,share=1./s.tetrahedra,charge=problem->charge;
    double low=HUGE_VAL,high=-HUGE_VAL;
    for(int i=0;i<blocks;i++) {low=fmin(low,bounds[2*i]);high=fmax(high,bounds[2*i+1]);}
    auto states=[&](double ef) {
        double n=0.;
        for(size_t item=threadIdx.x;item<s.items;item+=search) n+=tetra_count(sorted+4*item,ef);
        return block_sum(n,scratch)*full*share;
    };
    int status=result->status,iterations=0;
    const double capacity=full*s.maxspn*s.ne,tolerance=1.e-12*fmax(1.,charge);
    if(!status&&!(charge>0.)) status=tetra_bad_input;
    if(!status&&capacity<charge-tolerance) status=tetra_too_few_states;
    double ef=high;
    if(!status&&capacity>charge+tolerance) {
        double a=low,b=high;bool found=false;
        for(int iteration=0;iteration<200&&!found;iteration++) {
            ef=.5*(a+b);const double n=states(ef);iterations=iteration+1;
            if(fabs(n-charge)<tolerance) found=true;
            else if(n<charge) a=ef; else b=ef;
            if(b-a<=1.e-15*fmax(1.,fabs(ef))) found=fabs(states(.5*(a+b))-charge)<1.e2*tolerance;
        }
        if(!found) status=tetra_no_fermi;
    }
    // The highest band must lie above ef everywhere, or states are missing.
    if(!threadIdx.x) flag=0;__syncthreads();
    if(!status&&capacity>charge+tolerance)
        for(size_t q=threadIdx.x;q<size_t(s.maxspn)*s.tetrahedra;q+=search) {
            const int spin=int(q/s.tetrahedra),t=int(q%s.tetrahedra);
            if(sorted[4*((size_t(spin)*s.ne+s.ne-1)*s.tetrahedra+t)]<ef) flag=1;
        }
    __syncthreads();
    if(!status&&flag) status=tetra_top_band_occupied;
    const double counted=status?0.:states(ef);
    if(!threadIdx.x) {
        *fermi=ef;result->fermi=ef;result->entropy=0.;result->charge=counted;result->iterations=iterations;
        result->status=status;
    }
}
// Corner weights of every item, returned to the original corner order.
__global__ void corner_weights(Shape s,int nspinor,const double* fermi,const double* sorted,const unsigned char* order,
                               double* corner,const FermiResult* result) {
    if(result->status) return;
    const double full=nspinor==1?2.:1.,share=1./s.tetrahedra,ef=*fermi;
    for(size_t item=blockIdx.x*size_t(blockDim.x)+threadIdx.x;item<s.items;item+=size_t(gridDim.x)*blockDim.x) {
        double w[4];tetra_weights(sorted+4*item,ef,s.method==2,w);
        for(int i=0;i<4;i++) corner[4*item+order[4*item+i]]=w[i]*full*share;
    }
}
// Each (k, spin, band) sums its incident (tetrahedron, corner) pairs in
// tetrahedron order, as the host's single-thread accumulation.
__global__ void gather(Shape s,int nk,const int* offsets,const int* incidence,const double* corner,
                       double* occupations,const FermiResult* result) {
    const size_t states=size_t(nk)*s.maxspn*s.maxe;
    for(size_t q=blockIdx.x*size_t(blockDim.x)+threadIdx.x;q<states;q+=size_t(gridDim.x)*blockDim.x) {
        const int band=int(q%s.maxe),spin=int((q/s.maxe)%s.maxspn),k=int(q/(size_t(s.maxe)*s.maxspn));
        double total=0.;
        if(!result->status&&band<s.ne) {
            const size_t base=(size_t(spin)*s.ne+band)*s.tetrahedra;
            for(int j=offsets[k];j<offsets[k+1];j++) {
                const int t=incidence[j]/s.corners,c=incidence[j]%s.corners;
                const double* x=corner+4*(base+t);double sum=0.;
                if(s.method==3) {for(int i=0;i<4;i++) sum+=fit[i][c]*x[i];sum/=1260.;}
                else sum=x[c];
                total+=sum;
            }
        }
        occupations[q]=total;
    }
}
// QE's grouping, as tetra_occupations: bands within 1e-6 Ry of a group's first
// band share its total; a group reaching the top band is left as it is.
__global__ void average(Shape s,int nk,const double* energies,double* occupations,const FermiResult* result) {
    if(result->status) return;
    for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<nk*s.maxspn;q+=gridDim.x*blockDim.x) {
        const int k=q/s.maxspn,spin=q%s.maxspn;
        const double* e=energies+size_t(s.maxe)*(spin+size_t(s.maxspn)*k);
        double* w=occupations+size_t(s.maxe)*(spin+size_t(s.maxspn)*k);
        for(int b=0;b<s.ne;b++) {
            double sum=w[b];
            for(int c=b+1;c<s.ne;c++) {
                if(fabs(e[b]-e[c])<1.e-6) {sum+=w[c];continue;}
                for(int x=b;x<c;x++) w[x]=sum/(c-b);
                break;
            }
        }
    }
}
__global__ void begin(FermiResult* result) {*result=FermiResult{};}
}

TetraDevicePlan::TetraDevicePlan(const TetraGrid& grid,int method,int nspinor,int maxspn,int maxe,int ne,cudaStream_t stream)
    :method_(method),nspinor_(nspinor),maxspn_(maxspn),maxe_(maxe),ne_(ne),stream_(stream) {
    if(method<1||method>3||grid.corners!=(method==3?20:4)||grid.tetrahedra<1||(nspinor!=1&&nspinor!=2)||
       maxspn<1||maxspn>2||(nspinor==1&&maxspn!=1)||ne<1||ne>maxe||grid.point.empty())
        throw std::runtime_error("Invalid tetrahedron plan");
    corners_=grid.corners;tetrahedra_=grid.tetrahedra;
    nk_=1+*std::max_element(grid.point.begin(),grid.point.end());
    items_=size_t(maxspn)*ne*tetrahedra_;
    blocks_=int(std::min<size_t>(1024,(items_+threads-1)/threads));
    // Incidence of each listed k: (tetrahedron, corner) pairs whose grid point
    // maps to it, in increasing tetrahedron then corner order.
    std::vector<int> count(nk_+1,0);
    for(size_t j=0;j<grid.stencil.size();j++) count[grid.point[grid.stencil[j]]+1]++;
    for(int k=0;k<nk_;k++) count[k+1]+=count[k];
    std::vector<int> incidence(grid.stencil.size()),fill(count.begin(),count.end()-1);
    for(size_t j=0;j<grid.stencil.size();j++) incidence[fill[grid.point[grid.stencil[j]]]++]=int(j);
    if(grid.stencil.size()>size_t(INT_MAX)) throw std::runtime_error("Tetrahedron stencil too large");
    try {
        auto alloc=[&](auto** p,size_t bytes) {check(cudaMalloc(reinterpret_cast<void**>(p),bytes));bytes_+=bytes;};
        alloc(&stencil_,grid.stencil.size()*sizeof(int));alloc(&offsets_,count.size()*sizeof(int));
        alloc(&incidence_,incidence.size()*sizeof(int));
        alloc(&point_,grid.point.size()*sizeof(int));
        alloc(&sorted_,4*items_*sizeof(double));alloc(&corner_,4*items_*sizeof(double));
        alloc(&order_,4*items_);alloc(&bounds_,2*size_t(blocks_)*sizeof(double));alloc(&fermi_,sizeof(double));
        check(cudaMemcpy(stencil_,grid.stencil.data(),grid.stencil.size()*sizeof(int),cudaMemcpyHostToDevice));
        check(cudaMemcpy(offsets_,count.data(),count.size()*sizeof(int),cudaMemcpyHostToDevice));
        check(cudaMemcpy(incidence_,incidence.data(),incidence.size()*sizeof(int),cudaMemcpyHostToDevice));
        check(cudaMemcpy(point_,grid.point.data(),grid.point.size()*sizeof(int),cudaMemcpyHostToDevice));
    } catch(...) {release();throw;}
}
void TetraDevicePlan::release() {
    void* buffers[]={stencil_,incidence_,offsets_,point_,sorted_,corner_,bounds_,fermi_,order_};
    for(void* p:buffers) if(p) cudaFree(p);
}
TetraDevicePlan::~TetraDevicePlan() {release();}
size_t TetraDevicePlan::workspace_bytes() const {return bytes_;}
int TetraDevicePlan::apply(const FermiProblem* problem,const double* energies,double* occupations,FermiResult* result) {
    if(!problem||!energies||!occupations||!result) return int(cudaErrorInvalidValue);
    const Shape s{method_,maxspn_,maxe_,ne_,corners_,tetrahedra_,items_};
    begin<<<1,1,0,stream_>>>(result);
    sort_items<<<blocks_,threads,0,stream_>>>(s,stencil_,point_,energies,sorted_,order_,bounds_,result);
    find_fermi<<<1,search,0,stream_>>>(s,nspinor_,blocks_,bounds_,sorted_,problem,fermi_,result);
    corner_weights<<<blocks_,threads,0,stream_>>>(s,nspinor_,fermi_,sorted_,order_,corner_,result);
    const size_t states=size_t(nk_)*maxspn_*maxe_;
    gather<<<int(std::min<size_t>(4096,(states+threads-1)/threads)),threads,0,stream_>>>(s,nk_,offsets_,incidence_,corner_,
                                                                                        occupations,result);
    average<<<std::max(1,(nk_*maxspn_+127)/128),128,0,stream_>>>(s,nk_,energies,occupations,result);
    return int(cudaGetLastError());
}
}
