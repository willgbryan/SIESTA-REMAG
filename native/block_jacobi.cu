// SPDX-License-Identifier: GPL-3.0-or-later
#include "block_jacobi.hpp"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace remag {
namespace {
std::string location(const char* expression,int line) {
    return std::string(expression)+" at block_jacobi.cu:"+std::to_string(line)+": ";
}
void checked(cudaError_t c,const char* expression,int line) {
    if(c!=cudaSuccess) throw std::runtime_error(location(expression,line)+cudaGetErrorString(c));
}
void checked(cublasStatus_t c,const char* expression,int line) {
    if(c!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error(location(expression,line)+"cuBLAS "+std::to_string(int(c)));
}
void checked(cusolverStatus_t c,const char* expression,int line) {
    if(c!=CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(location(expression,line)+"cuSOLVER "+std::to_string(int(c)));
}
#define check(expression) checked((expression),#expression,__LINE__)
template<class T> __host__ __device__ T value(double x);
template<> __host__ __device__ double value<double>(double x) {return x;}
template<> __host__ __device__ cuDoubleComplex value<cuDoubleComplex>(double x) {return make_cuDoubleComplex(x,0.);}
__device__ double adjoint(double x) {return x;}
__device__ cuDoubleComplex adjoint(cuDoubleComplex x) {return cuConj(x);}
__device__ double average(double x,double y) {return .5*(x+y);}
__device__ cuDoubleComplex average(cuDoubleComplex x,cuDoubleComplex y) {return make_cuDoubleComplex(.5*(x.x+y.x),.5*(x.y+y.y));}
__device__ double real(double x) {return x;}
__device__ double real(cuDoubleComplex x) {return x.x;}
__device__ double magnitude(double x) {return fabs(x);}
__device__ double magnitude(cuDoubleComplex x) {return hypot(x.x,x.y);}
__device__ double squared(double x) {return x*x;}
__device__ double squared(cuDoubleComplex x) {return x.x*x.x+x.y*x.y;}
__device__ bool finite(double x) {return isfinite(x);}
__device__ bool finite(cuDoubleComplex x) {return isfinite(x.x)&&isfinite(x.y);}
struct Reduction {double norm,bound;};
template<class T> __device__ T lower_element(const T* a,int row,int col,int n) {
    if(row==col) return value<T>(real(a[row+col*n]));
    return row>col?a[row+col*n]:adjoint(a[col+row*n]);
}
__global__ void reset(BlockJacobiResult* r,BlockJacobiLocalFailure* failure) {
    *r={};r->relative_offdiagonal=INFINITY;
    *failure={0,-1,-1,-1};
    for(double& x:r->history) x=NAN;
}
template<class T> __global__ void initialize(int n,int padded,const T* h,const T* s,T* a,T* l,BlockJacobiResult* result) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<padded*padded) {int row=i%padded,col=i/padded;a[i]=(row<n&&col<n)?lower_element(h,row,col,n):value<T>(0.);}
    if(i<n*n) {
        T hv=lower_element(h,i%n,i/n,n),sv=lower_element(s,i%n,i/n,n);
        bool valid=finite(hv)&&finite(sv);
        if(!valid) atomicCAS(&result->status,0,2);
        l[i]=finite(sv)?sv:value<T>(i%n==i/n?1.:0.);
    }
}
template<class T> __global__ void guard_cholesky(int n,int padded,const int* info,T* l,T* a,BlockJacobiResult* result) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(*info) atomicCAS(&result->status,0,1);
    if((*info||result->status)&&i<n*n) {l[i]=value<T>(i%n==i/n?1.:0.);a[i%n+(i/n)*padded]=value<T>(0.);}
}
template<class T> __global__ void hermitize(int n,int padded,const T* a,T* b,BlockJacobiResult* result) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=padded*padded) return;
    int row=i%padded,col=i/padded;
    b[i]=(row<n&&col<n)?average(a[i],adjoint(a[col+row*padded])):value<T>(0.);
    if(!finite(b[i])) atomicCAS(&result->status,0,2);
}
template<class T> __global__ void initial_norm(int n,int padded,const T* a,double2* rows) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n) return;
    double norm=0.,bound=0.;for(int col=0;col<n;col++) {T x=a[row+col*padded];norm+=squared(x);bound+=magnitude(x);}
    rows[row]=make_double2(norm,bound);
}
__global__ void reduce_initial(int n,const double2* rows,Reduction* state,BlockJacobiResult* result) {
    double norm=0.,bound=0.;for(int i=0;i<n;i++) {norm+=rows[i].x;bound=fmax(bound,rows[i].y);}
    state->norm=norm>0.?sqrt(norm):1.;state->bound=bound+1.;
    if(!isfinite(norm)||!isfinite(bound)) atomicCAS(&result->status,0,2);
}
template<class T> __global__ void pad_and_identity(int n,int padded,const Reduction* state,T* a,T* v) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=padded*padded) return;
    int row=i%padded,col=i/padded;
    if(row>=n||col>=n) a[i]=value<T>(row==col?state->bound:0.);
    v[i]=value<T>(row==col?1.:0.);
}
template<class T,int Width> __global__ void retain_failure(int round,const T* original,const BlockJacobiResult* result,
        const BlockJacobiLocalFailure* failure,T* failed) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(Width*Width)&&failure->info&&failure->round==round&&failure->sweep==result->sweeps)
        failed[i]=original[failure->pair*(Width*Width)+i];
}
__global__ void local_status(int batch,int round,const int* info,BlockJacobiResult* result,BlockJacobiLocalFailure* failure) {
    for(int i=0;i<batch;i++) if(info[i]) {
        if(!failure->info) *failure={info[i],i,round,result->sweeps};
        atomicCAS(&result->status,0,3);
    }
}
// Each round keeps both matrix axes in the current pair order. Contiguous
// Width-column panels feed right GEMMs; strided Width-row panels feed left
// GEMMs directly. Only the transition between pair orders moves whole arrays.
template<class T,bool Square> __global__ void permute(int padded,const int* indices,const T* source,T* target) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=padded*padded) return;
    int row=i%padded,col=i/padded;
    target[i]=source[(Square?indices[row]:row)+indices[col]*padded];
}
template<class T,int Width> __global__ void gather_paired(int padded,const T* a,T* local,T* original) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=(padded/Width)*(Width*Width)) return;
    int pair=i/(Width*Width),row=i%Width,col=(i/Width)%Width;
    local[i]=original[i]=a[pair*Width+row+(pair*Width+col)*padded];
}
// Gather only the small diagonal pair blocks in the next ordering. The
// full matrix is permuted as it enters the fused two-sided rotation below.
template<class T,int Width> __global__ void gather_paired_mapped(int padded,const int* indices,
        const T* a,T* local,T* original) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(padded/Width)*(Width*Width))return;
    const int pair=i/(Width*Width),row=i%Width,col=(i/Width)%Width;
    local[i]=original[i]=a[indices[pair*Width+row]+indices[pair*Width+col]*padded];
}
__device__ double accumulate_product(double sum,double x,double y) {return fma(x,y,sum);}
__device__ cuDoubleComplex accumulate_product(cuDoubleComplex sum,cuDoubleComplex x,cuDoubleComplex y) {
    sum.x=fma(x.x,y.x,sum.x);sum.x=fma(-x.y,y.y,sum.x);
    sum.y=fma(x.x,y.y,sum.y);sum.y=fma(x.y,y.x,sum.y);return sum;
}
#include "shared_local_jacobi.inc"
// Gather the two paired column panels directly into shared memory and apply
// their local rotation. A block owns 32 rows by 32 output columns; each lane
// computes four columns. No globally reordered copy of V is materialized.
template<class T> __global__ void rotate_paired_vectors(int padded,const int* indices,
        const T* input,const T* local,T* output) {
    __shared__ T panel[32][32],rotation[32][32];
    const int x=threadIdx.x,y=threadIdx.y,row=blockIdx.x*32+x,pair=blockIdx.y;
    for(int k=y;k<32;k+=8) {
        panel[k][x]=input[row+indices[pair*32+k]*padded];
        rotation[k][x]=local[pair*1024+x+k*32];
    }
    __syncthreads();
    T sums[4]={value<T>(0.),value<T>(0.),value<T>(0.),value<T>(0.)};
    for(int k=0;k<32;k++) {
        const T a=panel[k][x];
        #pragma unroll
        for(int c=0;c<4;c++)sums[c]=accumulate_product(sums[c],a,rotation[y+8*c][k]);
    }
    #pragma unroll
    for(int c=0;c<4;c++)output[row+(pair*32+y+8*c)*padded]=sums[c];
}
// Pack the complex inputs into four FP64 planes for SM80 double tensor cores.
// Sixteen warps each own one 8x8 result tile. A four-double shared skew keeps
// every fragment origin 32-byte aligned while separating adjacent columns.
// This changes accumulation grouping, not precision, and is opt-in pending
// direct upstream scientific checks and matched end-to-end measurements.
__global__ void rotate_paired_vectors_fp64(int padded,const int* indices,
        const cuDoubleComplex* input,const cuDoubleComplex* local,cuDoubleComplex* output) {
    namespace wmma=nvcuda::wmma;
    constexpr int stride=36;
    __shared__ __align__(32) double panel_real[32][stride],panel_imag[32][stride];
    __shared__ __align__(32) double rotation_real[32][stride],rotation_imag[32][stride];
    const int tid=threadIdx.x,warp=tid/32,pair=blockIdx.y;
    for(int i=tid;i<1024;i+=512) {
        const int row=i%32,col=i/32;
        const cuDoubleComplex a=input[blockIdx.x*32+row+indices[pair*32+col]*padded];
        const cuDoubleComplex b=local[pair*1024+row+col*32];
        panel_real[col][row]=a.x;panel_imag[col][row]=a.y;
        rotation_real[col][row]=b.x;rotation_imag[col][row]=b.y;
    }
    __syncthreads();
    const int row_tile=(warp%4)*8,col_tile=(warp/4)*8;
    wmma::fragment<wmma::matrix_a,8,8,4,double,wmma::col_major> ar,ai;
    wmma::fragment<wmma::matrix_b,8,8,4,double,wmma::col_major> br,bi;
    wmma::fragment<wmma::accumulator,8,8,4,double> cr,ci;
    wmma::fill_fragment(cr,0.);wmma::fill_fragment(ci,0.);
    #pragma unroll
    for(int k=0;k<32;k+=4) {
        wmma::load_matrix_sync(ar,&panel_real[k][row_tile],stride);
        wmma::load_matrix_sync(ai,&panel_imag[k][row_tile],stride);
        wmma::load_matrix_sync(br,&rotation_real[col_tile][k],stride);
        wmma::load_matrix_sync(bi,&rotation_imag[col_tile][k],stride);
        wmma::mma_sync(cr,ar,br,cr);
        wmma::mma_sync(ci,ar,bi,ci);
        wmma::mma_sync(ci,ai,br,ci);
        #pragma unroll
        for(int i=0;i<ai.num_elements;i++)ai.x[i]=-ai.x[i];
        wmma::mma_sync(cr,ai,bi,cr);
    }
    // Other warps may still be reading the input planes; reuse only after all
    // warp products finish. Stores tile the whole panel without overlap.
    __syncthreads();
    wmma::store_matrix_sync(&panel_real[col_tile][row_tile],cr,stride,wmma::mem_col_major);
    wmma::store_matrix_sync(&panel_imag[col_tile][row_tile],ci,stride,wmma::mem_col_major);
    __syncthreads();
    for(int i=tid;i<1024;i+=512) {
        const int row=i%32,col=i/32;
        output[blockIdx.x*32+row+(pair*32+col)*padded]=
            make_cuDoubleComplex(panel_real[col][row],panel_imag[col][row]);
    }
}
// Each CTA owns one 32x32 tile of U_row^H P(A) U_col. Reuse the shared
// tile for the intermediate right product and the rotation buffer for U_row.
// The full permutation and intermediate right product never reach global memory.
// Off-diagonal tiles are computed once; Hermitian partners share that result.
template<class T> __global__ void rotate_paired_matrix(int padded,const int* indices,
        const T* input,const T* local,T* output) {
    __shared__ T panel[32][33],rotation[32][33];
    const int x=threadIdx.x,y=threadIdx.y,tile=blockIdx.x;
    // Enumerate exactly the lower-triangular tiles, without empty upper CTAs.
    const int row_pair=int((sqrt(8.*tile+1.)-1.)*.5);
    const int col_pair=tile-row_pair*(row_pair+1)/2;
    const int source_row=indices[row_pair*32+x];
    for(int k=y;k<32;k+=8) {
        panel[k][x]=input[source_row+indices[col_pair*32+k]*padded];
        rotation[k][x]=local[col_pair*1024+x+k*32];
    }
    __syncthreads();
    T sums[4]={value<T>(0.),value<T>(0.),value<T>(0.),value<T>(0.)};
    for(int k=0;k<32;k++) {
        const T a=panel[k][x];
        #pragma unroll
        for(int c=0;c<4;c++)sums[c]=accumulate_product(sums[c],a,rotation[y+8*c][k]);
    }
    __syncthreads(); // All readers finish before either shared buffer is reused.
    #pragma unroll
    for(int c=0;c<4;c++)panel[y+8*c][x]=sums[c];
    for(int k=y;k<32;k+=8)rotation[k][x]=local[row_pair*1024+x+k*32];
    __syncthreads();
    #pragma unroll
    for(int c=0;c<4;c++)sums[c]=value<T>(0.);
    for(int k=0;k<32;k++) {
        const T a=adjoint(rotation[x][k]);
        #pragma unroll
        for(int c=0;c<4;c++)sums[c]=accumulate_product(sums[c],a,panel[y+8*c][k]);
    }
    #pragma unroll
    for(int c=0;c<4;c++)output[row_pair*32+x+(col_pair*32+y+8*c)*padded]=sums[c];
    if(row_pair!=col_pair) {
        // Wait for all readers of the intermediate product, then transpose
        // through shared storage so the partner's global writes stay coalesced.
        __syncthreads();
        #pragma unroll
        for(int c=0;c<4;c++)panel[y+8*c][x]=sums[c];
        __syncthreads();
        #pragma unroll
        for(int c=0;c<4;c++)output[col_pair*32+x+(row_pair*32+y+8*c)*padded]=adjoint(panel[x][y+8*c]);
    }
}
// Two FP64 complex products per lower-triangular tile, with no global
// intermediate. The second pass consumes the first pass's shared result and
// the conjugate-transposed row rotation. Hermitian partners share one result.
__global__ void rotate_paired_matrix_fp64(int padded,const int* indices,
        const cuDoubleComplex* input,const cuDoubleComplex* local,cuDoubleComplex* output) {
    namespace wmma=nvcuda::wmma;
    constexpr int stride=36;
    __shared__ __align__(32) double panel_real[32][stride],panel_imag[32][stride];
    __shared__ __align__(32) double rotation_real[32][stride],rotation_imag[32][stride];
    const int tid=threadIdx.x,warp=tid/32,tile=blockIdx.x;
    const int row_pair=int((sqrt(8.*tile+1.)-1.)*.5);
    const int col_pair=tile-row_pair*(row_pair+1)/2;
    for(int i=tid;i<1024;i+=512) {
        const int row=i%32,col=i/32;
        const cuDoubleComplex a=input[indices[row_pair*32+row]+indices[col_pair*32+col]*padded];
        const cuDoubleComplex b=local[col_pair*1024+row+col*32];
        panel_real[col][row]=a.x;panel_imag[col][row]=a.y;
        rotation_real[col][row]=b.x;rotation_imag[col][row]=b.y;
    }
    __syncthreads();
    const int row_tile=(warp%4)*8,col_tile=(warp/4)*8;
    #pragma unroll
    for(int pass=0;pass<2;pass++) {
        const double* left_real=pass==0?&panel_real[0][0]:&rotation_real[0][0];
        const double* left_imag=pass==0?&panel_imag[0][0]:&rotation_imag[0][0];
        const double* right_real=pass==0?&rotation_real[0][0]:&panel_real[0][0];
        const double* right_imag=pass==0?&rotation_imag[0][0]:&panel_imag[0][0];
        wmma::fragment<wmma::matrix_a,8,8,4,double,wmma::col_major> ar,ai;
        wmma::fragment<wmma::matrix_b,8,8,4,double,wmma::col_major> br,bi;
        wmma::fragment<wmma::accumulator,8,8,4,double> cr,ci;
        wmma::fill_fragment(cr,0.);wmma::fill_fragment(ci,0.);
        #pragma unroll
        for(int k=0;k<32;k+=4) {
            wmma::load_matrix_sync(ar,left_real+k*stride+row_tile,stride);
            wmma::load_matrix_sync(ai,left_imag+k*stride+row_tile,stride);
            wmma::load_matrix_sync(br,right_real+col_tile*stride+k,stride);
            wmma::load_matrix_sync(bi,right_imag+col_tile*stride+k,stride);
            wmma::mma_sync(cr,ar,br,cr);
            wmma::mma_sync(ci,ar,bi,ci);
            wmma::mma_sync(ci,ai,br,ci);
            #pragma unroll
            for(int j=0;j<ai.num_elements;j++)ai.x[j]=-ai.x[j];
            wmma::mma_sync(cr,ai,bi,cr);
        }
        __syncthreads(); // Every warp finishes before shared inputs are reused.
        wmma::store_matrix_sync(&panel_real[col_tile][row_tile],cr,stride,wmma::mem_col_major);
        wmma::store_matrix_sync(&panel_imag[col_tile][row_tile],ci,stride,wmma::mem_col_major);
        if(pass==0) {
            for(int i=tid;i<1024;i+=512) {
                const int row=i%32,col=i/32;
                const cuDoubleComplex u=local[row_pair*1024+row+col*32];
                rotation_real[row][col]=u.x;rotation_imag[row][col]=-u.y;
            }
        }
        __syncthreads();
    }
    for(int i=tid;i<1024;i+=512) {
        const int row=i%32,col=i/32;
        output[row_pair*32+row+(col_pair*32+col)*padded]=
            make_cuDoubleComplex(panel_real[col][row],panel_imag[col][row]);
        if(row_pair!=col_pair)
            output[col_pair*32+row+(row_pair*32+col)*padded]=
                make_cuDoubleComplex(panel_real[row][col],-panel_imag[row][col]);
    }
}
template<class T> __global__ void offdiagonal_norm(int padded,const T* a,double* partial) {
    __shared__ double scratch[256];
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    scratch[threadIdx.x]=i>=padded*padded?0.:(!finite(a[i])?NAN:(i%padded!=i/padded?squared(a[i]):0.));__syncthreads();
    for(int stride=128;stride;stride/=2) {if(threadIdx.x<stride) scratch[threadIdx.x]+=scratch[threadIdx.x+stride];__syncthreads();}
    if(!threadIdx.x) partial[blockIdx.x]=scratch[0];
}
__global__ void convergence(int blocks,int maximum,double tolerance,const double* partial,const Reduction* state,BlockJacobiResult* result) {
    double norm=0.;for(int i=0;i<blocks;i++) norm+=partial[i];
    result->relative_offdiagonal=sqrt(norm)/state->norm;
    if(result->sweeps<32) result->history[result->sweeps]=result->relative_offdiagonal;
    result->sweeps++;
    if(!isfinite(result->relative_offdiagonal)) atomicCAS(&result->status,0,2);
    else if(result->relative_offdiagonal>=tolerance&&result->sweeps>=maximum) atomicCAS(&result->status,0,4);
}
__global__ void seed_condition(const int* valid,const BlockJacobiResult* result,cudaGraphConditionalHandle handle) {
    cudaGraphSetConditional(handle,*valid>0&&result->status==0);
}
template<class T> __global__ void publish_seed(int count,const T* v,T* seed,int* valid,const BlockJacobiResult* result) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count&&!result->status)seed[i]=v[i];
    if(i==0)*valid=result->status?0:*valid+1;
}
__global__ void set_condition(const BlockJacobiResult* result,double tolerance,cudaGraphConditionalHandle condition,int initial) {
    cudaGraphSetConditional(condition,result->status==0&&(initial||result->relative_offdiagonal>=tolerance));
}
template<class T> __global__ void sort_diagonal(int n,int padded,const T* a,int* order,double* w,const BlockJacobiResult* result) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=padded||result->status) return;
    double x=real(a[col+col*padded]);int rank=0;
    for(int j=0;j<padded;j++) {double y=real(a[j+j*padded]);rank+=(y<x||(y==x&&j<col));}
    if(rank<n) {order[rank]=col;w[rank]=x;}
}
template<class T> __global__ void gather_vectors(int n,int padded,const int* order,const T* v,T* z,const BlockJacobiResult* result) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*n) z[i]=result->status?value<T>(0.):v[i%n+order[i/n]*padded];
}
}

struct BlockJacobiPlan::Impl {
    virtual ~Impl()=default;
    virtual void warmup(const void*,const void*,BlockJacobiResult*)=0;
    virtual cudaGraphNode_t append(cudaGraph_t,cudaGraph_t,cudaGraphNode_t,const void*,const void*,void*,double*,BlockJacobiResult*,void*,int*)=0;
    size_t bytes=0,host_bytes=0,upload_bytes=0,upload_calls=0,device_allocations=0;
    BlockJacobiLocalFailure* local_failure=nullptr;
    void* failed_local_input=nullptr;
    int nodes=0,forbidden=0,width=0;bool seeded=false;size_t seed_storage_bytes=0;
};
namespace {
template<class T,int Width> struct Plan final:BlockJacobiPlan::Impl {
    int n,padded,batch,rounds,maximum,blocks,chol_workspace=0;double tolerance;
    cudaStream_t stream;cusolverDnHandle_t solver=nullptr;cusolverDnParams_t params=nullptr;cublasHandle_t blas=nullptr;
    syevjInfo_t jacobi=nullptr;int jacobi_workspace=0,seed_chol_workspace=0;bool use_jacobi=false;
    T* seed_chol_work=nullptr;
    std::vector<void*> allocations;
    T *a=nullptr,*l=nullptr,*v=nullptr,*first=nullptr,*second=nullptr,*local=nullptr,*original_local=nullptr,*chol_work=nullptr,*scalars=nullptr;
    double *local_values=nullptr,*partial=nullptr;double2* row_sums=nullptr;
    int *pairs=nullptr,*restore=nullptr,*order=nullptr,*chol_info=nullptr,*eigen_info=nullptr;
    Reduction* reduction=nullptr;unsigned char* eigen_work=nullptr;size_t eigen_bytes=0;
    unsigned char* blas_work=nullptr;
    constexpr static size_t blas_bytes=4*1024*1024;
    constexpr static auto dtype=(std::is_same_v<T,double>)?CUDA_R_64F:CUDA_C_64F;
    constexpr static auto adj=(std::is_same_v<T,double>)?CUBLAS_OP_T:CUBLAS_OP_C;
    template<class U> U* allocate(size_t count,const U* source=nullptr) {
        if(!count) return nullptr;U* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(U)));
        allocations.push_back(p);bytes+=count*sizeof(U);device_allocations++;
        if(source) {
            check(cudaMemcpy(p,source,count*sizeof(U),cudaMemcpyHostToDevice));
            upload_bytes+=count*sizeof(U);upload_calls++;
        }
        return p;
    }
    bool use_shared=false,use_fused_matrix=false,use_fp64_vectors=false,use_fp64_matrix=false;
    Plan(int dimension,int limit,double tol,cudaStream_t execution):n(dimension),maximum(limit),tolerance(tol),stream(execution) {
        try {
            if(n<1||n>4096||limit<1||limit>32||!std::isfinite(tol)||tol<=0.||!stream) throw std::runtime_error("Invalid block Jacobi configuration");
            const char* warm=std::getenv("REMAG_SCF_WARM_START");
            if(warm&&std::string(warm)!="0"&&std::string(warm)!="1")throw std::runtime_error("Invalid eigensolver warm-start mode");
            seeded=warm&&std::string(warm)=="1";
            const char* selected=std::getenv("REMAG_BLOCK_LOCAL_SOLVER");const std::string backend=selected?selected:"syev";
            if(backend!="syev"&&backend!="jacobi") throw std::runtime_error("Invalid local eigensolver backend");
            use_jacobi=backend=="jacobi";
            const char* shared=std::getenv("REMAG_BLOCK_SHARED_JACOBI");
            if(shared&&std::string(shared)!="0"&&std::string(shared)!="1")throw std::runtime_error("Invalid shared local eigensolver mode");
            use_shared=shared&&std::string(shared)=="1";
            if(use_shared&&(!use_jacobi||Width!=32))throw std::runtime_error("Shared local eigensolver requires 32-state Jacobi blocks");
            if(use_jacobi&&Width!=32) throw std::runtime_error("Batched Jacobi requires 32-state local problems");
            const char* fused_matrix=std::getenv("REMAG_BLOCK_FUSED_MATRIX");
            if(fused_matrix&&std::string(fused_matrix)!="0"&&std::string(fused_matrix)!="1")
                throw std::runtime_error("Invalid fused matrix rotation mode");
            use_fused_matrix=fused_matrix&&std::string(fused_matrix)=="1";
            if(use_fused_matrix&&Width!=32)throw std::runtime_error("Fused matrix rotation requires 32-state blocks");
            const char* fp64_vectors=std::getenv("REMAG_BLOCK_FP64_VECTORS");
            if(fp64_vectors&&std::string(fp64_vectors)!="0"&&std::string(fp64_vectors)!="1")
                throw std::runtime_error("Invalid FP64 vector rotation mode");
            use_fp64_vectors=fp64_vectors&&std::string(fp64_vectors)=="1";
            if(use_fp64_vectors&&Width!=32)throw std::runtime_error("FP64 vector rotation requires 32-state blocks");
            const char* fp64_matrix=std::getenv("REMAG_BLOCK_FP64_MATRIX");
            if(fp64_matrix&&std::string(fp64_matrix)!="0"&&std::string(fp64_matrix)!="1")
                throw std::runtime_error("Invalid FP64 matrix rotation mode");
            use_fp64_matrix=fp64_matrix&&std::string(fp64_matrix)=="1";
            if(use_fp64_matrix&&(Width!=32||!use_fused_matrix))
                throw std::runtime_error("FP64 matrix rotation requires fused 32-state blocks");
            width=Width;padded=Width*((n+Width-1)/Width);batch=padded/Width;rounds=padded/(Width/2)-1;blocks=(padded*padded+255)/256;
            check(cusolverDnCreate(&solver));check(cusolverDnSetStream(solver,stream));check(cusolverDnCreateParams(&params));
            check(cublasCreate(&blas));check(cublasSetStream(blas,stream));check(cublasSetPointerMode(blas,CUBLAS_POINTER_MODE_DEVICE));
            // Without user-owned workspace cuBLAS may capture allocation/free
            // nodes, which CUDA 12.6 does not permit inside child graphs.
            blas_work=allocate<unsigned char>(blas_bytes);check(cublasSetWorkspace(blas,blas_work,blas_bytes));
            size_t size=size_t(padded)*padded;seed_storage_bytes=size*sizeof(T);
            a=allocate<T>(size);l=allocate<T>(size_t(n)*n);v=allocate<T>(size);first=allocate<T>(size);second=allocate<T>(size);
            local=allocate<T>(size_t(batch)*(Width*Width));local_values=allocate<double>(padded);partial=allocate<double>(blocks);
            original_local=allocate<T>(size_t(batch)*(Width*Width));failed_local_input=allocate<T>((Width*Width));
            row_sums=allocate<double2>(n);order=allocate<int>(n);chol_info=allocate<int>(1);eigen_info=allocate<int>(batch);reduction=allocate<Reduction>(1);
            local_failure=allocate<BlockJacobiLocalFailure>(1);
            const T constants[]={value<T>(1.),value<T>(0.)};scalars=allocate<T>(2,constants);
            std::vector<int> ring(padded/(Width/2)),mapping(size_t(rounds)*padded);std::iota(ring.begin(),ring.end(),0);
            for(int round=0;round<rounds;round++) {
                for(int pair=0;pair<batch;pair++) {
                    int p=std::min(ring[pair],ring[ring.size()-1-pair]),q=std::max(ring[pair],ring[ring.size()-1-pair]);
                    for(int i=0;i<Width;i++) mapping[round*padded+pair*Width+i]=(i<Width/2?p*(Width/2)+i:q*(Width/2)+i-Width/2);
                }
                std::rotate(ring.begin()+1,ring.end()-1,ring.end());
            }
            // Translate each absolute schedule into the preceding round's
            // coordinates. Restore the original order at each sweep boundary,
            // so repeated graph bodies, convergence and warm bases keep their ABI.
            std::vector<int> inverse(padded),relative(mapping.size()),next(padded);
            std::iota(inverse.begin(),inverse.end(),0);
            for(int round=0;round<rounds;round++) {
                for(int i=0;i<padded;i++) {
                    const int original=mapping[round*padded+i];
                    relative[round*padded+i]=inverse[original];next[original]=i;
                }
                inverse.swap(next);
            }
            pairs=allocate<int>(relative.size(),relative.data());
            restore=allocate<int>(inverse.size(),inverse.data());
            if constexpr(std::is_same_v<T,double>) check(cusolverDnDpotrf_bufferSize(solver,CUBLAS_FILL_MODE_LOWER,n,l,n,&chol_workspace));
            else check(cusolverDnZpotrf_bufferSize(solver,CUBLAS_FILL_MODE_LOWER,n,l,n,&chol_workspace));
            chol_work=allocate<T>(chol_workspace);
            if(seeded) {
                if constexpr(std::is_same_v<T,double>)check(cusolverDnDpotrf_bufferSize(solver,CUBLAS_FILL_MODE_LOWER,padded,second,padded,&seed_chol_workspace));
                else check(cusolverDnZpotrf_bufferSize(solver,CUBLAS_FILL_MODE_LOWER,padded,second,padded,&seed_chol_workspace));
                seed_chol_work=allocate<T>(seed_chol_workspace);
            }
            if(use_shared) {eigen_bytes=host_bytes=0;}
            else if(use_jacobi) {
                check(cusolverDnCreateSyevjInfo(&jacobi));
                check(cusolverDnXsyevjSetTolerance(jacobi,std::numeric_limits<double>::epsilon()));
                check(cusolverDnXsyevjSetMaxSweeps(jacobi,100));check(cusolverDnXsyevjSetSortEig(jacobi,1));
                if constexpr(std::is_same_v<T,double>) check(cusolverDnDsyevjBatched_bufferSize(solver,CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER,Width,local,Width,local_values,&jacobi_workspace,jacobi,batch));
                else check(cusolverDnZheevjBatched_bufferSize(solver,CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER,Width,local,Width,local_values,&jacobi_workspace,jacobi,batch));
                eigen_bytes=size_t(jacobi_workspace)*sizeof(T);
            } else check(cusolverDnXsyevBatched_bufferSize(solver,params,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,Width,dtype,local,Width,
                CUDA_R_64F,local_values,dtype,&eigen_bytes,&host_bytes,batch));
            if(host_bytes) throw std::runtime_error("Small eigensolver requested host workspace");
            eigen_work=allocate<unsigned char>(eigen_bytes);
        } catch(...) {release();throw;}
    }
    ~Plan() override {release();}
    void release() {
        if(jacobi) {cusolverDnDestroySyevjInfo(jacobi);jacobi=nullptr;}
        if(blas) {cublasDestroy(blas);blas=nullptr;}if(params) {cusolverDnDestroyParams(params);params=nullptr;}
        if(solver) {cusolverDnDestroy(solver);solver=nullptr;}for(void* p:allocations) cudaFree(p);allocations.clear();
    }
    void triangular(cublasSideMode_t side,cublasOperation_t op,T* target,int leading) {
        if constexpr(std::is_same_v<T,double>) check(cublasDtrsm(blas,side,CUBLAS_FILL_MODE_LOWER,op,CUBLAS_DIAG_NON_UNIT,n,n,scalars,l,n,target,leading));
        else check(cublasZtrsm(blas,side,CUBLAS_FILL_MODE_LOWER,op,CUBLAS_DIAG_NON_UNIT,n,n,scalars,l,n,target,leading));
    }
    void prepare(const void* h,const void* s,BlockJacobiResult* result) {
        reset<<<1,1,0,stream>>>(result,local_failure);check(cudaGetLastError());
        initialize<<<blocks,256,0,stream>>>(n,padded,static_cast<const T*>(h),static_cast<const T*>(s),a,l,result);check(cudaGetLastError());
        if constexpr(std::is_same_v<T,double>) check(cusolverDnDpotrf(solver,CUBLAS_FILL_MODE_LOWER,n,l,n,chol_work,chol_workspace,chol_info));
        else check(cusolverDnZpotrf(solver,CUBLAS_FILL_MODE_LOWER,n,l,n,chol_work,chol_workspace,chol_info));
        guard_cholesky<<<(n*n+255)/256,256,0,stream>>>(n,padded,chol_info,l,a,result);check(cudaGetLastError());
        triangular(CUBLAS_SIDE_LEFT,CUBLAS_OP_N,a,padded);triangular(CUBLAS_SIDE_RIGHT,adj,a,padded);
        hermitize<<<blocks,256,0,stream>>>(n,padded,a,first,result);check(cudaGetLastError());
        check(cudaMemcpyAsync(a,first,size_t(padded)*padded*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        initial_norm<<<(n+255)/256,256,0,stream>>>(n,padded,a,row_sums);check(cudaGetLastError());
        reduce_initial<<<1,1,0,stream>>>(n,row_sums,reduction,result);check(cudaGetLastError());
        pad_and_identity<<<blocks,256,0,stream>>>(n,padded,reduction,a,v);check(cudaGetLastError());
    }
    void full_multiply(cublasOperation_t op,const T* x,const T* y,T* z) {
        if constexpr(std::is_same_v<T,double>)check(cublasDgemm(blas,op,CUBLAS_OP_N,padded,padded,padded,
            scalars,x,padded,y,padded,scalars+1,z,padded));
        else check(cublasZgemm(blas,op,CUBLAS_OP_N,padded,padded,padded,
            scalars,x,padded,y,padded,scalars+1,z,padded));
    }
    void seed_transform(const void* seed,BlockJacobiResult* result) {
        // Previous V is close to unitary. Cholesky QR controls accumulated
        // roundoff: G=V^H V=L_g L_g^H; Q=V L_g^{-H}; A'=Q^H A Q.
        // L_g uses second, leaving the physical overlap factor L untouched.
        if(seed)check(cudaMemcpyAsync(v,seed,seed_storage_bytes,cudaMemcpyDeviceToDevice,stream));
        full_multiply(adj,v,v,first);
        hermitize<<<blocks,256,0,stream>>>(padded,padded,first,second,result);check(cudaGetLastError());
        if constexpr(std::is_same_v<T,double>)check(cusolverDnDpotrf(solver,CUBLAS_FILL_MODE_LOWER,padded,second,padded,seed_chol_work,seed_chol_workspace,chol_info));
        else check(cusolverDnZpotrf(solver,CUBLAS_FILL_MODE_LOWER,padded,second,padded,seed_chol_work,seed_chol_workspace,chol_info));
        guard_cholesky<<<blocks,256,0,stream>>>(padded,padded,chol_info,second,v,result);check(cudaGetLastError());
        if constexpr(std::is_same_v<T,double>)check(cublasDtrsm(blas,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,adj,
            CUBLAS_DIAG_NON_UNIT,padded,padded,scalars,second,padded,v,padded));
        else check(cublasZtrsm(blas,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,adj,
            CUBLAS_DIAG_NON_UNIT,padded,padded,scalars,second,padded,v,padded));
        full_multiply(CUBLAS_OP_N,a,v,first);full_multiply(adj,v,first,second);
        hermitize<<<blocks,256,0,stream>>>(padded,padded,second,a,result);check(cudaGetLastError());
    }
    void rotate(bool rows,const T* input,T* output) {
        const int m=rows?Width:padded,cols=rows?padded:Width;
        const T* left=rows?local:input;const T* right=rows?input:local;
        const int lda=rows?Width:padded,ldb=rows?padded:Width,ldc=padded;
        const long long panel=static_cast<long long>(padded)*Width;
        const long long stride_left=rows?Width*Width:panel;
        const long long stride_right=rows?Width:Width*Width;
        const long long stride_output=rows?Width:panel;
        // Left products address disjoint row panels of the same column-major
        // output. Their actual elements do not overlap, despite interleaving.
        const auto op=rows?adj:CUBLAS_OP_N;
        if constexpr(std::is_same_v<T,double>) check(cublasDgemmStridedBatched(blas,op,CUBLAS_OP_N,m,cols,Width,scalars,left,lda,stride_left,
            right,ldb,stride_right,scalars+1,output,ldc,stride_output,batch));
        else check(cublasZgemmStridedBatched(blas,op,CUBLAS_OP_N,m,cols,Width,scalars,left,lda,stride_left,
            right,ldb,stride_right,scalars+1,output,ldc,stride_output,batch));
    }
    void sweep(BlockJacobiResult* result) {
        T* current=a;T* reordered=first;
        T* vector_input=v;T* vector_output=second;
        for(int round=0;round<rounds;round++) {
            const int* indices=pairs+round*padded;
            if(!use_fused_matrix) {
                permute<T,true><<<blocks,256,0,stream>>>(padded,indices,current,reordered);check(cudaGetLastError());
            }
            if constexpr(Width!=32) {
                permute<T,false><<<blocks,256,0,stream>>>(padded,indices,v,second);check(cudaGetLastError());
            }
            if(use_fused_matrix)gather_paired_mapped<T,Width><<<(batch*(Width*Width)+255)/256,256,0,stream>>>(padded,indices,current,local,original_local);
            else gather_paired<T,Width><<<(batch*(Width*Width)+255)/256,256,0,stream>>>(padded,reordered,local,original_local);
            check(cudaGetLastError());
            if(use_shared) {
                shared_local_jacobi<<<batch,512,0,stream>>>(local,local_values,eigen_info);check(cudaGetLastError());
            } else {
            // CUDA 12.6 has a documented complex workspace-initialization issue.
            // Keep the device memset explicit and visible in the graph.
            check(cudaMemsetAsync(eigen_work,0,eigen_bytes,stream));
            if(use_jacobi) {
                if constexpr(std::is_same_v<T,double>) check(cusolverDnDsyevjBatched(solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,
                    Width,local,Width,local_values,reinterpret_cast<T*>(eigen_work),jacobi_workspace,eigen_info,jacobi,batch));
                else check(cusolverDnZheevjBatched(solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,
                    Width,local,Width,local_values,reinterpret_cast<T*>(eigen_work),jacobi_workspace,eigen_info,jacobi,batch));
            } else check(cusolverDnXsyevBatched(solver,params,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,Width,dtype,local,Width,
                CUDA_R_64F,local_values,dtype,eigen_work,eigen_bytes,nullptr,0,eigen_info,batch));
            }
            local_status<<<1,1,0,stream>>>(batch,round,eigen_info,result,local_failure);check(cudaGetLastError());
            retain_failure<T,Width><<<(Width*Width+255)/256,256,0,stream>>>(round,original_local,result,local_failure,static_cast<T*>(failed_local_input));check(cudaGetLastError());
            if constexpr(Width==32) {
                if constexpr(std::is_same_v<T,cuDoubleComplex>) {
                    if(use_fp64_vectors)
                        rotate_paired_vectors_fp64<<<dim3(padded/32,batch),512,0,stream>>>(padded,indices,vector_input,local,vector_output);
                    else rotate_paired_vectors<<<dim3(padded/32,batch),dim3(32,8),0,stream>>>(padded,indices,vector_input,local,vector_output);
                } else rotate_paired_vectors<<<dim3(padded/32,batch),dim3(32,8),0,stream>>>(padded,indices,vector_input,local,vector_output);
                check(cudaGetLastError());std::swap(vector_input,vector_output);
            } else rotate(false,second,v);
            if(use_fused_matrix) {
                if constexpr(std::is_same_v<T,cuDoubleComplex>) {
                    if(use_fp64_matrix)
                        rotate_paired_matrix_fp64<<<batch*(batch+1)/2,512,0,stream>>>(padded,indices,current,local,reordered);
                    else rotate_paired_matrix<<<batch*(batch+1)/2,dim3(32,8),0,stream>>>(padded,indices,current,local,reordered);
                } else rotate_paired_matrix<<<batch*(batch+1)/2,dim3(32,8),0,stream>>>(padded,indices,current,local,reordered);
                check(cudaGetLastError());
            } else {
                rotate(false,reordered,current);
                rotate(true,current,reordered);
            }
            std::swap(current,reordered);
        }
        // rounds=2*batch-1 is odd, so current==first and a is free.
        permute<T,true><<<blocks,256,0,stream>>>(padded,restore,current,a);check(cudaGetLastError());
        if constexpr(Width==32) {
            // The odd round count leaves the rotated vectors in second.
            // Restore directly into v, which is the next sweep's input.
            permute<T,false><<<blocks,256,0,stream>>>(padded,restore,vector_input,v);check(cudaGetLastError());
        } else {
            permute<T,false><<<blocks,256,0,stream>>>(padded,restore,v,second);check(cudaGetLastError());
            check(cudaMemcpyAsync(v,second,size_t(padded)*padded*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        }
        offdiagonal_norm<<<blocks,256,0,stream>>>(padded,a,partial);check(cudaGetLastError());
        convergence<<<1,1,0,stream>>>(blocks,maximum,tolerance,partial,reduction,result);check(cudaGetLastError());
    }
    void finish(void* output,double* w,BlockJacobiResult* result) {
        sort_diagonal<<<(padded+255)/256,256,0,stream>>>(n,padded,a,order,w,result);check(cudaGetLastError());
        auto z=static_cast<T*>(output);
        gather_vectors<<<(n*n+255)/256,256,0,stream>>>(n,padded,order,v,z,result);check(cudaGetLastError());
        triangular(CUBLAS_SIDE_LEFT,adj,z,n);
    }
    void warmup(const void* h,const void* s,BlockJacobiResult* result) override {
        prepare(h,s,result);sweep(result);
        // Prime the final triangular solve too: its dimensions/operation can
        // select a different library path from the generalized reduction.
        finish(first,local_values,result);
        if(seeded)seed_transform(nullptr,result); // Prime GEMM/QR library paths before capture.
        check(cudaStreamSynchronize(stream));
    }
    void audit(cudaGraph_t graph) {
        size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> records(count);
        check(cudaGraphGetNodes(graph,records.data(),&count));
        for(auto node:records) {
            cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));nodes++;
            if(type==cudaGraphNodeTypeGraph) {cudaGraph_t child;check(cudaGraphChildGraphNodeGetGraph(node,&child));audit(child);}
            else if(type==cudaGraphNodeTypeMemcpy) {
                cudaMemcpy3DParms p{};check(cudaGraphMemcpyNodeGetParams(node,&p));
                if(p.kind!=cudaMemcpyDeviceToDevice) forbidden++;
            } else if(type!=cudaGraphNodeTypeKernel&&type!=cudaGraphNodeTypeMemset&&type!=cudaGraphNodeTypeEmpty) {
                forbidden++;
                throw std::runtime_error("Unsupported captured node type "+std::to_string(int(type)));
            }
        }
    }
    cudaGraphNode_t capture_child(cudaGraph_t destination,cudaGraphNode_t dependency,const std::function<void()>& work) {
        cudaGraph_t captured=nullptr;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        try {work();} catch(...) {cudaStreamEndCapture(stream,&captured);if(captured) cudaGraphDestroy(captured);throw;}
        check(cudaStreamEndCapture(stream,&captured));cudaGraphNode_t node;
        try {audit(captured);} catch(...) {cudaGraphDestroy(captured);throw;}
        auto status=cudaGraphAddChildGraphNode(&node,destination,dependency?&dependency:nullptr,dependency?1:0,captured);
        cudaGraphDestroy(captured);check(status);nodes++;return node;
    }
    cudaGraphNode_t condition_kernel(cudaGraph_t graph,cudaGraphNode_t dependency,BlockJacobiResult* result,
                                     cudaGraphConditionalHandle handle,int initial) {
        void* args[]={&result,&tolerance,&handle,&initial};cudaKernelNodeParams p{};
        p.func=reinterpret_cast<void*>(set_condition);p.gridDim=dim3(1);p.blockDim=dim3(1);p.kernelParams=args;
        cudaGraphNode_t node;check(cudaGraphAddKernelNode(&node,graph,dependency?&dependency:nullptr,dependency?1:0,&p));nodes++;return node;
    }
    cudaGraphNode_t append(cudaGraph_t root,cudaGraph_t destination,cudaGraphNode_t dependency,const void* h,const void* s,
                          void* z,double* w,BlockJacobiResult* result,void* seed,int* valid) override {
        cudaGraphConditionalHandle handle;check(cudaGraphConditionalHandleCreate(&handle,root,0,0));
        auto init=capture_child(destination,dependency,[&]{prepare(h,s,result);});
        if(seed) {
            cudaGraphConditionalHandle reuse;check(cudaGraphConditionalHandleCreate(&reuse,root,0,0));
            auto selected=capture_child(destination,init,[&]{seed_condition<<<1,1,0,stream>>>(valid,result,reuse);check(cudaGetLastError());});
            cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=reuse;
            p.conditional.type=cudaGraphCondTypeIf;p.conditional.size=1;
            check(cudaGraphAddNode(&init,destination,&selected,1,&p));nodes++;
            capture_child(p.conditional.phGraph_out[0],nullptr,[&]{seed_transform(seed,result);});
        }
        auto ready=condition_kernel(destination,init,result,handle,1);
        cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=handle;
        p.conditional.type=cudaGraphCondTypeWhile;p.conditional.size=1;cudaGraphNode_t loop;
        check(cudaGraphAddNode(&loop,destination,&ready,1,&p));nodes++;auto body=p.conditional.phGraph_out[0];
        auto child=capture_child(body,nullptr,[&]{sweep(result);});condition_kernel(body,child,result,handle,0);
        return capture_child(destination,loop,[&]{
            finish(z,w,result);
            if(seed) {publish_seed<<<blocks,256,0,stream>>>(padded*padded,v,static_cast<T*>(seed),valid,result);check(cudaGetLastError());}
        });
    }
};
}
BlockJacobiPlan::BlockJacobiPlan(int n,bool complex_values,int maximum,double tolerance,cudaStream_t stream) {
    const char* option=std::getenv("REMAG_BLOCK_WIDTH");
    const std::string width=option?option:"64";
    if(width!="32"&&width!="64") throw std::runtime_error("REMAG_BLOCK_WIDTH must be 32 or 64");
    if(width=="32") {
        if(complex_values) impl_=std::make_unique<Plan<cuDoubleComplex,32>>(n,maximum,tolerance,stream);
        else impl_=std::make_unique<Plan<double,32>>(n,maximum,tolerance,stream);
    } else {
        if(complex_values) impl_=std::make_unique<Plan<cuDoubleComplex,64>>(n,maximum,tolerance,stream);
        else impl_=std::make_unique<Plan<double,64>>(n,maximum,tolerance,stream);
    }
}
BlockJacobiPlan::~BlockJacobiPlan()=default;
void BlockJacobiPlan::warmup(const void* h,const void* s,BlockJacobiResult* r) {
    if(!h||!s||!r) throw std::runtime_error("Missing block Jacobi warmup buffer");impl_->warmup(h,s,r);
}
cudaGraphNode_t BlockJacobiPlan::append(cudaGraph_t root,cudaGraph_t destination,cudaGraphNode_t dependency,const void* h,const void* s,
                                      void* z,double* w,BlockJacobiResult* r) {
    if(!root||!destination||!h||!s||!z||!w||!r) throw std::runtime_error("Missing block Jacobi graph/buffer");
    return impl_->append(root,destination,dependency,h,s,z,w,r,nullptr,nullptr);
}
cudaGraphNode_t BlockJacobiPlan::append_seeded(cudaGraph_t root,cudaGraph_t destination,cudaGraphNode_t dependency,
        const void* h,const void* s,void* z,double* w,BlockJacobiResult* r,void* seed,int* valid) {
    if(!root||!destination||!h||!s||!z||!w||!r||!seed||!valid||!impl_->seeded)
        throw std::runtime_error("Missing seeded solver configuration/graph/buffer");
    return impl_->append(root,destination,dependency,h,s,z,w,r,seed,valid);
}
size_t BlockJacobiPlan::seed_bytes() const {return impl_->seed_storage_bytes;}
size_t BlockJacobiPlan::workspace_bytes() const {return impl_->bytes;}
size_t BlockJacobiPlan::host_workspace_bytes() const {return impl_->host_bytes;}
size_t BlockJacobiPlan::setup_upload_bytes() const {return impl_->upload_bytes;}
size_t BlockJacobiPlan::setup_upload_calls() const {return impl_->upload_calls;}
size_t BlockJacobiPlan::device_allocation_count() const {return impl_->device_allocations;}
const BlockJacobiLocalFailure* BlockJacobiPlan::local_failure() const {return impl_->local_failure;}
const void* BlockJacobiPlan::failed_local_input() const {return impl_->failed_local_input;}
int BlockJacobiPlan::local_dimension() const {return impl_->width;}
int BlockJacobiPlan::graph_nodes() const {return impl_->nodes;}
int BlockJacobiPlan::forbidden_nodes() const {return impl_->forbidden;}
}
