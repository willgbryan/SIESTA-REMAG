// SPDX-License-Identifier: GPL-3.0-or-later
// Batched form of pinned SIESTA m_radfft, including its direct small-q correction.
#include "execution.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <memory>
namespace {
unsigned long long calls=0,h2d=0,d2h=0,compact_calls=0;
double elapsed=0.;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void fftcheck(cufftResult e){if(e!=CUFFT_SUCCESS)throw std::runtime_error("radial cuFFT failed: "+std::to_string(int(e)));}
struct Work {
    std::vector<void*> buffers; cufftHandle plan=0;
    ~Work(){if(plan)cufftDestroy(plan);for(auto p:buffers)cudaFree(p);}
    template<class T>T* alloc(size_t n){T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));buffers.push_back(p);return p;}
    template<class T>T* upload(const T* p,size_t n){T* q=alloc<T>(n);check(cudaMemcpy(q,p,n*sizeof(T),cudaMemcpyHostToDevice));h2d+=n*sizeof(T);remag::transfer(remag::Transfer::radial_setup,true,n*sizeof(T));return q;}
};
__device__ double ipow(double x,int n){double y=1.;int m=n<0?-n:n;for(int i=0;i<m;i++)y*=x;return n<0?1./y:y;}
__device__ bool valid_l(int l,int a,int b){return l>=abs(a-b)&&l<=a+b&&((l-abs(a-b))%2==0);}
__global__ void convolution(int nr,int dim1,int pairs,int mode,double dq,double c,const double* a,const double* b,double* f){
    size_t total=size_t(nr+1)*pairs;
    for(size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x;idx<total;idx+=size_t(gridDim.x)*blockDim.x){
        int ir=idx%(nr+1),pair=idx/(nr+1),ia=pair%dim1,ib=pair/dim1;
        double v=c*a[size_t(ia)*(nr+1)+ir]*b[size_t(ib)*(nr+1)+ir];
        if(mode==2){double q=ir*dq;v=v*q*q;} f[idx]=v;
    }
}
__global__ void prepare_fft(int nr,int dim1,int pairs,int l,int n,double dr,double c,double px,double py,
                             const int* la,const int* lb,const double* f,cufftDoubleComplex* z){
    size_t total=size_t(2*nr)*pairs;
    for(size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x;idx<total;idx+=size_t(gridDim.x)*blockDim.x){
        int jr=idx%(2*nr),pair=idx/(2*nr);
        double v=0.;
        if(jr!=0&&jr!=nr&&valid_l(l,la[pair%dim1],lb[pair/dim1])){
            int ir=jr<nr?jr:2*nr-jr;double r=ir*dr;
            double fr=f[size_t(pair)*(nr+1)+ir];
            if(jr>nr){r=-r;if(l%2)fr=-fr;}
            v=c*fr*ipow(r,n-l+1);
        }
        z[idx]=make_cuDoubleComplex(v*px,v*py);
    }
}
__global__ void accumulate(int nr,int pairs,int l,int maxl,double dq,const cufftDoubleComplex* z,double* out){
    size_t total=size_t(nr+1)*pairs;
    for(size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x;idx<total;idx+=size_t(gridDim.x)*blockDim.x){
        int iq=idx%(nr+1),pair=idx/(nr+1);
        size_t target=(size_t(pair)*(maxl+1)+l)*(nr+1)+iq;
        if(iq)out[target]=(out[target]+z[size_t(pair)*2*nr+iq].x)/(iq*dq);
    }
}
__device__ double bessph(int l,double x){
    if(fabs(x)<max(1,2*l-1)){
        double term=1.,sum=0.,x2=x*x;
        for(int i=1;i<=l;i++)term=term*x/(2*i+1);
        for(int i=1;i<=100;i++){sum+=term;term=(-term)*x2/(2*i*(2*i+2*l+1));if(fabs(term)<1.e-15)return sum;}
        return nan("");
    }
    if(l==0)return sin(x)/x;
    if(l==1)return (sin(x)/x-cos(x))/x;
    double y=1./x,prev=sin(x)*y,cur=(prev-cos(x))*y;
    for(int n=1;n<l;n++){double next=(2*n+1)*y*cur-prev;prev=cur;cur=next;}
    return cur;
}
__global__ void small_q(int nr,int dim1,int pairs,int l,int maxl,int mq,double dr,double dq,double c,
                       const int* la,const int* lb,const double* f,double* out){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=pairs*(mq+1))return;
    int iq=idx%(mq+1),pair=idx/(mq+1);
    if(!valid_l(l,la[pair%dim1],lb[pair/dim1]))return;
    double sum=0.;
    if(iq||l==0){
        double q=iq*dq;
        // Preserve upstream sequential accumulation and endpoint inclusion.
        for(int ir=1;ir<=nr;ir++){
            double r=ir*dr,v=r*r*f[size_t(pair)*(nr+1)+ir];
            if(iq)v*=bessph(l,q*r);sum+=v;
        }
        sum=sum*2.*c;
    }
    out[(size_t(pair)*(maxl+1)+l)*(nr+1)+iq]=sum;
}
// Phase, radial division, and the exact first-maximum rule used by PROPOR.
__global__ void radial_prepare(int nr,int dim1,int maxl,double dr,const int* la,const int* lb,
                               double* bank,int* peaks,int* status){
    int entry=blockIdx.x,l=entry%(maxl+1),pair=entry/(maxl+1),tid=threadIdx.x;
    int a=la[pair%dim1],b=lb[pair/dim1];
    if(!valid_l(l,a,b))return;
    double* f=bank+size_t(entry)*(nr+1);
    bool negate=(abs(a-b-l)/2)%2!=0;
    double best=0.;int peak=0;
    for(int ir=1+tid;ir<=nr;ir+=blockDim.x){
        double v=negate?-f[ir]:f[ir];if(l)v/=ipow(ir*dr,l);f[ir]=v;
        if(!isfinite(v))atomicExch(status,1);
        double m=fabs(v);if(m>best||(m==best&&m>0.&&ir<peak)){best=m;peak=ir;}
    }
    __shared__ double magnitude[256];__shared__ int index[256];
    magnitude[tid]=best;index[tid]=peak;__syncthreads();
    for(int stride=128;stride;stride/=2){
        if(tid<stride){double v=magnitude[tid+stride];int ix=index[tid+stride];
            if(v>magnitude[tid]||(v==magnitude[tid]&&v>0.&&ix<index[tid])){magnitude[tid]=v;index[tid]=ix;}}
        __syncthreads();
    }
    if(tid==0){f[0]=l?(4.*f[1]-f[2])/3.:(negate?-f[0]:f[0]);peaks[entry]=index[0];if(!isfinite(f[0]))atomicExch(status,1);}
}
// All lanes advance through candidates in upstream order. Eight warps search
// canonical functions independently; taking the minimum match preserves the
// original first-match semantics, including its absolute 1e-8 tolerance.
__global__ void radial_compress(int nr,int dim1,int pairs,int maxl,double tol,const int* la,const int* lb,
    const double* bank,const int* peaks,int* canonical,int* mapping,double* coefficients,int* result){
    __shared__ int count,match[8];__shared__ double scale[8];
    int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    if(tid==0)count=0;__syncthreads();
    for(int entry=0;entry<pairs*(maxl+1);entry++){
        int l=entry%(maxl+1),pair=entry/(maxl+1);
        if(!valid_l(l,la[pair%dim1],lb[pair/dim1]))continue;
        const double* a=bank+size_t(entry)*(nr+1);
        int found=count;double factor=1.;
        for(int j=warp;j<count;j+=8){
            int prior=canonical[j],peak=peaks[prior];
            if(peak==0){if(lane==0)atomicExch(result+1,2);continue;}
            const double* b=bank+size_t(prior)*(nr+1);
            double ratio=a[peak]/b[peak];bool bad=false;
            for(int base=1;base<=nr;base+=32){
                int ir=base+lane;bool mismatch=ir<=nr&&fabs(a[ir]-b[ir]*ratio)>tol;
                if(__any_sync(0xffffffff,mismatch)){bad=true;break;}
            }
            if(!bad){found=j;factor=ratio;break;}
        }
        if(lane==0){match[warp]=found;scale[warp]=factor;}__syncthreads();
        if(tid==0){
            int chosen=count,winner=-1;
            for(int w=0;w<8;w++)if(match[w]<chosen){chosen=match[w];winner=w;}
            if(winner<0){canonical[count]=entry;count++;coefficients[entry]=1.;}
            else coefficients[entry]=scale[winner];
            mapping[entry]=chosen+1;
        }
        __syncthreads();
    }
    if(tid==0)result[0]=count;
}
__global__ void radial_gather(int nr,int count,const double* bank,const int* canonical,double* compact){
    size_t total=size_t(nr+1)*count;
    for(size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x;idx<total;idx+=size_t(gridDim.x)*blockDim.x){
        int ir=idx%(nr+1),row=idx/(nr+1);compact[idx]=bank[size_t(canonical[row])*(nr+1)+ir];
    }
}
struct Compact {
    Work work;int nr=0,entries=0,count=0;double* bank=nullptr;double* coefficients=nullptr;
    int* mapping=nullptr;int* canonical=nullptr;
};
std::unique_ptr<Compact> pending;
void download(void* host,const void* device,size_t bytes){check(cudaMemcpy(host,device,bytes,cudaMemcpyDeviceToHost));d2h+=bytes;remag::transfer(remag::Transfer::radial_setup,false,bytes);}
}
extern "C" void remag_radial_batch_stats(unsigned long long* counts,double* seconds){counts[0]=calls;counts[1]=h2d;counts[2]=d2h;counts[3]=compact_calls;*seconds=elapsed;}
int radial_batch_impl(int nr,int dim1,int dim2,int maxl,int mode,double rmax,double input_dq,
                                  const int* la,const int* lb,const double* a,const double* b,double* output, double radial_dr=0.,double tol=0.,int* count=nullptr,void** owner=nullptr){
    auto start=std::chrono::steady_clock::now();
    try{
        size_t pairs=size_t(dim1)*dim2,total=size_t(nr+1)*(maxl+1)*pairs;
        if(remag::in_scf()||nr<2||nr>8192||dim1<=0||dim2<=0||pairs>100000||maxl<0||maxl>12||
           total>size_t(256)*1024*1024||!std::isfinite(rmax)||rmax<=0||!std::isfinite(input_dq)||input_dq<=0)
            throw std::runtime_error("Unsupported radial batch dimensions or phase");
        for(int i=0;i<dim1;i++)if(la[i]<0||la[i]>maxl)throw std::runtime_error("Invalid radial angular momentum");
        for(int i=0;i<dim2;i++)if(lb[i]<0||lb[i]>maxl)throw std::runtime_error("Invalid radial angular momentum");
        double pi=4.*atan(1.),dr=rmax/nr,dq=pi/rmax,c=dr/sqrt(2.*pi);
        std::unique_ptr<Compact> compact;
        if(owner){
            if(pending||!count||radial_dr<=0.||tol<=0.||!std::isfinite(radial_dr)||!std::isfinite(tol))
                throw std::runtime_error("Invalid compact radial setup lifetime/constants");
            *owner=nullptr;compact=std::make_unique<Compact>();
        }
        Work plain;Work& w=compact?compact->work:plain;
        auto da=w.upload(a,size_t(nr+1)*dim1);auto db=w.upload(b,size_t(nr+1)*dim2);
        auto dla=w.upload(la,dim1);auto dlb=w.upload(lb,dim2);
        auto f=w.alloc<double>(size_t(nr+1)*pairs);auto out=w.alloc<double>(total);
        auto z=w.alloc<cufftDoubleComplex>(size_t(2*nr)*pairs);
        check(cudaMemset(out,0,total*sizeof(double)));
        convolution<<<std::min(size_t(4096),(size_t(nr+1)*pairs+255)/256),256>>>(nr,dim1,int(pairs),mode,input_dq,pow(2.*pi,1.5),da,db,f);
        fftcheck(cufftPlan1d(&w.plan,2*nr,CUFFT_Z2Z,int(pairs)));
        double p[13][13][2]={};p[0][0][1]=-1.;
        if(maxl>0){p[1][0][1]=-1.;p[1][1][0]=-1.;}
        for(int l=2;l<=maxl;l++)for(int n=0;n<=l;n++)for(int j=0;j<2;j++){
            if(n<l)p[l][n][j]+=(2*l-1)*p[l-1][n][j];
            if(n>=2)p[l][n][j]-=p[l-2][n-2][j];
        }
        for(int l=0;l<=maxl;l++){
            for(int n=0;n<=l;n++){
                prepare_fft<<<std::min(size_t(4096),(size_t(2*nr)*pairs+255)/256),256>>>(nr,dim1,int(pairs),l,n,dr,c,p[l][n][0],p[l][n][1],dla,dlb,f,z);
                fftcheck(cufftExecZ2Z(w.plan,z,z,CUFFT_INVERSE));
                accumulate<<<std::min(size_t(4096),(size_t(nr+1)*pairs+255)/256),256>>>(nr,int(pairs),l,maxl,dq,z,out);
            }
            int mq=l?int(nr*pow(1.e-8,1./l)):0;
            small_q<<<(pairs*(mq+1)+127)/128,128>>>(nr,dim1,int(pairs),l,maxl,mq,dr,dq,c,dla,dlb,f,out);
        }
        if(compact){
            int entries=int(pairs)*(maxl+1);
            auto peaks=w.alloc<int>(entries);auto canonical=w.alloc<int>(entries);auto map=w.alloc<int>(entries);
            auto coeff=w.alloc<double>(entries);auto result=w.alloc<int>(2);
            check(cudaMemset(map,0,entries*sizeof(int)));check(cudaMemset(coeff,0,entries*sizeof(double)));check(cudaMemset(result,0,2*sizeof(int)));
            radial_prepare<<<entries,256>>>(nr,dim1,maxl,radial_dr,dla,dlb,out,peaks,result+1);
            radial_compress<<<1,256>>>(nr,dim1,int(pairs),maxl,tol,dla,dlb,out,peaks,canonical,map,coeff,result);
            check(cudaGetLastError());int state[2]={};download(state,result,sizeof(state));
            if(state[1]||state[0]<=0||state[0]>entries)throw std::runtime_error("Invalid compact radial result");
            compact->nr=nr;compact->entries=entries;compact->count=state[0];compact->bank=out;
            compact->mapping=map;compact->coefficients=coeff;compact->canonical=canonical;
            *count=state[0];*owner=compact.get();pending=std::move(compact);compact_calls++;
        }else {check(cudaGetLastError());download(output,out,total*sizeof(double));}
        calls++;
        elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"REMAG radial batch: %s\n",e.what());return -290;}
}

extern "C" int remag_radial_batch(int nr,int n1,int n2,int maxl,int mode,double rmax,double dq,
    const int* l1,const int* l2,const double* a,const double* b,double* output){
    return radial_batch_impl(nr,n1,n2,maxl,mode,rmax,dq,l1,l2,a,b,output);
}
extern "C" int remag_radial_compact_begin(int nr,int n1,int n2,int maxl,int mode,double rmax,double dq,
    double dr,double tol,const int* l1,const int* l2,const double* a,const double* b,int* count,void** owner){
    return radial_batch_impl(nr,n1,n2,maxl,mode,rmax,dq,l1,l2,a,b,nullptr,dr,tol,count,owner);
}
extern "C" int remag_radial_compact_export(void* owner,double* values,int* mapping,double* coefficients){
    auto start=std::chrono::steady_clock::now();
    try{
        if(!pending||owner!=pending.get()||remag::in_scf()||!values||!mapping||!coefficients)
            throw std::runtime_error("Invalid compact radial export lifetime");
        auto& c=*pending;size_t total=size_t(c.nr+1)*c.count;
        auto packed=c.work.alloc<double>(total);
        radial_gather<<<std::min(size_t(4096),(total+255)/256),256>>>(c.nr,c.count,c.bank,c.canonical,packed);
        check(cudaGetLastError());download(values,packed,total*sizeof(double));
        download(mapping,c.mapping,c.entries*sizeof(int));download(coefficients,c.coefficients,c.entries*sizeof(double));
        pending.reset();elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return 0;
    }catch(const std::exception& e){pending.reset();std::fprintf(stderr,"REMAG compact radial export: %s\n",e.what());return -291;}
}
