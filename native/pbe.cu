// SPDX-License-Identifier: GPL-3.0-or-later
// Mesh functional from pinned GridXC cellxc.F90; see GRIDXC_NOTICE.
#include "pbe.hpp"
#include "pbe_math.hpp"
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace remag {
namespace {
constexpr int threads=256,fields=13;
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
__device__ bool shape_ok(const PbeProblem& p,int size,int spins) {
    return p.spins==spins&&p.mesh[0]>0&&p.mesh[1]>0&&p.mesh[2]>0&&
           static_cast<long long>(p.mesh[0])*p.mesh[1]<=INT_MAX/p.mesh[2]&&
           p.mesh[0]*p.mesh[1]*p.mesh[2]==size;
}
__global__ void initialize(const PbeProblem* p,PbeResult* out,int n1,int n2,int n3,int spins) {
    *out=PbeResult{};
    if(p->mesh[0]!=n1||p->mesh[1]!=n2||p->mesh[2]!=n3||p->spins!=spins) out->status=1;
    if(!isfinite(p->volume)||p->volume<=0.) out->status=2;
    for(int i=0;i<9;i++) if(!isfinite(p->gradient[i])) out->status=2;
    const auto b=p->gradient;
    double determinant=b[0]*(b[4]*b[8]-b[5]*b[7])-b[3]*(b[1]*b[8]-b[2]*b[7])+b[6]*(b[1]*b[5]-b[2]*b[4]);
    if(!isfinite(determinant)||determinant==0.) out->status=2;
}
__device__ double weight(int offset) {
    // Original nn=3 Lagrange derivative stencil, in increasing neighbor order.
    const double w[7]={-1./60.,3./20.,-3./4.,0.,3./4.,-3./20.,1./60.};
    return w[offset+3];
}
__device__ int neighbor(int index,const PbeProblem& p,int axis,int delta) {
    const int stride=axis==0?1:axis==1?p.mesh[0]:p.mesh[0]*p.mesh[1];
    const int coordinate=(index/stride)%p.mesh[axis];
    int shifted=(coordinate+delta)%p.mesh[axis];if(shifted<0) shifted+=p.mesh[axis];
    return index+(shifted-coordinate)*stride;
}
__device__ double finite_density(const double* density,int index,PbeResult* out) {
    double value=density[index];
    if(!isfinite(value)) {atomicMax(&out->status,2);return 0.;}
    return value;
}
__device__ void gradient(int index,int size,int spins,const PbeProblem& p,const double* rho,
                         double g[4][3],PbeResult* out) {
    for(int axis=0;axis<3;axis++) for(int offset=-3;offset<=3;offset++) {
        const int j=neighbor(index,p,axis,offset);
        for(int s=0;s<spins;s++) {
            const double value=finite_density(rho,j+size*s,out);
            for(int k=0;k<3;k++) g[s][k]+=p.gradient[k+3*axis]*weight(offset)*value;
        }
    }
}
__global__ void evaluate(int size,int spins,const PbeProblem* problem,const double* density,
                         PbePoint* points,PbeResult* out) {
    const auto p=*problem;const bool valid=shape_ok(p,size,spins);
    for(int index=blockIdx.x*blockDim.x+threadIdx.x;index<size;index+=gridDim.x*blockDim.x) {
        PbePoint point{};
        if(valid) {
            double d[4]{},g[4][3]{};
            for(int s=0;s<spins;s++) d[s]=finite_density(density,index+size*s,out);
            double total=d[0]+(spins==1?0.:d[1]);
            if(!isfinite(total)) atomicMax(&out->status,2);
            if(total>=1.e-15) {
                gradient(index,size,spins,p,density,g,out);
                point=pbe_point(spins,d,g);
                bool finite=isfinite(point.x)&&isfinite(point.c);
                for(int s=0;s<spins;s++) {
                    finite=finite&&isfinite(point.dx[s])&&isfinite(point.dc[s]);
                    for(int k=0;k<3;k++) finite=finite&&isfinite(point.gx[s][k])&&isfinite(point.gc[s][k]);
                }
                if(!finite) {atomicMax(&out->status,2);point=PbePoint{};}
            }
        }
        points[index]=point;
    }
}
__global__ void assemble(int size,int spins,const PbeProblem* problem,const double* density,
                         const PbePoint* points,double* potential,double* partial,PbeResult* out) {
    __shared__ double sums[fields][threads];
    const auto p=*problem;const bool valid=shape_ok(p,size,spins);
    double values[fields]{};
    for(int index=blockIdx.x*blockDim.x+threadIdx.x;index<size;index+=gridDim.x*blockDim.x) {
        if(!valid) {for(int s=0;s<spins;s++) potential[index+size*s]=0.;continue;}
        const auto point=points[index];double d[4]{},g[4][3]{};
        for(int s=0;s<spins;s++) d[s]=finite_density(density,index+size*s,out);
        const double total=d[0]+(spins==1?0.:d[1]);
        const double ex=total*point.x,ec=total*point.c;
        values[0]+=ex;values[1]+=ec;values[2]+=ex;values[3]+=ec;
        for(int s=0;s<spins;s++) {
            double vx=point.dx[s],vc=point.dc[s];
            // Gather the adjoint contributions. Duplicate periodic neighbors
            // in small meshes retain their separate stencil coefficients.
            for(int axis=0;axis<3;axis++) for(int offset=-3;offset<=3;offset++) {
                const auto& source=points[neighbor(index,p,axis,-offset)];
                double x=0.,c=0.;
                for(int k=0;k<3;k++) {
                    x+=source.gx[s][k]*(p.gradient[k+3*axis]*weight(offset));
                    c+=source.gc[s][k]*(p.gradient[k+3*axis]*weight(offset));
                }
                vx+=x;vc+=c;
            }
            potential[index+size*s]=2*(vx+vc);
            if(!isfinite(potential[index+size*s])) atomicMax(&out->status,2);
            values[2]-=d[s]*vx;values[3]-=d[s]*vc;
        }
        // Density at a skipped point can still receive neighboring adjoint
        // potential, but its own pointwise gradient-stress contribution is zero.
        if(total>=1.e-15) {
            gradient(index,size,spins,p,density,g,out);
            for(int col=0;col<3;col++) for(int row=0;row<3;row++)
                for(int s=0;s<spins;s++) values[4+row+3*col]-=g[s][row]*(point.gx[s][col]+point.gc[s][col]);
        }
    }
    for(int field=0;field<fields;field++) sums[field][threadIdx.x]=values[field];
    __syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int field=0;field<fields;field++) sums[field][threadIdx.x]+=sums[field][threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) for(int field=0;field<fields;field++) partial[fields*blockIdx.x+field]=sums[field][0];
}
__global__ void reduce(int size,int blocks,const PbeProblem* p,const double* partial,PbeResult* out) {
    __shared__ double sums[fields][threads];
    for(int field=0;field<fields;field++) {
        double value=0.;for(int i=threadIdx.x;i<blocks;i+=threads) value+=partial[fields*i+field];
        sums[field][threadIdx.x]=value;
    }
    __syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int field=0;field<fields;field++) sums[field][threadIdx.x]+=sums[field][threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) {
        for(int field=0;field<fields;field++) if(!isfinite(sums[field][0])) out->status=2;
        if(out->status) return;
        for(int i=0;i<4;i++) out->energies[i]=2*p->volume/size*sums[i][0];
        for(int i=0;i<9;i++) out->stress[i]=2./size*sums[4+i][0];
        for(int i=0;i<3;i++) out->stress[4*i]+=(out->energies[0]+out->energies[1])/p->volume;
        for(int i=0;i<4;i++) if(!isfinite(out->energies[i])) out->status=2;
        for(int i=0;i<9;i++) if(!isfinite(out->stress[i])) out->status=2;
    }
}
}

PbePlan::PbePlan(int n1,int n2,int n3,int spins,cudaStream_t stream):
    mesh_{n1,n2,n3},spins_(spins),stream_(stream) {
    if((spins!=1&&spins!=2&&spins!=4)||n1<1||n2<1||n3<1||
       static_cast<long long>(n1)*n2>INT_MAX/n3||n1*n2*n3>INT_MAX/spins)
        throw std::runtime_error("Invalid PBE grid shape or spin count");
    size_=n1*n2*n3;blocks_=std::min(1024,(size_-1)/threads+1);
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&points_),size_t(size_)*sizeof(PbePoint)));
        check(cudaMalloc(reinterpret_cast<void**>(&partial_),size_t(blocks_)*fields*sizeof(double)));
    } catch(...) {release();throw;}
}
void PbePlan::release() {if(points_) cudaFree(points_);if(partial_) cudaFree(partial_);}
PbePlan::~PbePlan() {release();}
size_t PbePlan::workspace_bytes() const {return size_t(size_)*sizeof(PbePoint)+size_t(blocks_)*fields*sizeof(double);}
int PbePlan::apply(const PbeProblem* problem,const double* density,double* potential,PbeResult* result) {
    if(!problem||!density||!potential||!result||density==potential) return int(cudaErrorInvalidValue);
    initialize<<<1,1,0,stream_>>>(problem,result,mesh_[0],mesh_[1],mesh_[2],spins_);
    evaluate<<<blocks_,threads,0,stream_>>>(size_,spins_,problem,density,points_,result);
    assemble<<<blocks_,threads,0,stream_>>>(size_,spins_,problem,density,points_,potential,partial_,result);
    reduce<<<1,threads,0,stream_>>>(size_,blocks_,problem,partial_,result);
    return int(cudaGetLastError());
}
}
