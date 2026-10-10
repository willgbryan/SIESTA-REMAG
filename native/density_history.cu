// SPDX-License-Identifier: GPL-3.0-or-later
#include "density_history.hpp"
#include "density_history_math.hpp"
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>
namespace remag {
namespace {
constexpr int threads=256;
struct State {
    uint64_t generation,distribution,epoch,last_push;
    int capacity,nsc[3],count,density_bank,xyz_bank,topology_bank;
};
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ void describe(const State* s,DensityHistoryResult* r) {
    r->count=s->count;r->generation=s->generation;r->epoch=s->epoch;r->last_push=s->last_push;
}
__device__ bool shape(DensityHistoryConfig c,const DensityHistoryCommand* p) {
    if(p->capacity<1||p->capacity>c.capacity)return false;
    long long images=1;
    for(int k=0;k<3;k++) {
        // SIESTA's auxiliary supercell contains the central image and paired
        // positive/negative images. Even axis lengths are not supported.
        if(p->nsc[k]<1||p->nsc[k]%2==0||images>INT_MAX/p->nsc[k])return false;
        images*=p->nsc[k];
    }
    return images<=INT_MAX/c.orbitals;
}
__global__ void begin(DensityHistoryConfig c,int operation,const DensityHistoryCommand* p,
        const int* convergence,State* s,DensityHistoryResult* r) {
    *r=DensityHistoryResult{};describe(s,r);
    if(!p->generation||!p->distribution){r->status=1;return;}
    if(operation==0) {
        if(p->generation<=s->generation){r->status=4;return;}
        if(p->epoch!=0){r->status=1;return;}
        *s=State{};s->generation=p->generation;s->distribution=p->distribution;describe(s,r);return;
    }
    if(p->generation!=s->generation||p->distribution!=s->distribution||!p->epoch){r->status=4;return;}
    if(!shape(c,p)){r->status=1;return;}
    if(operation==1) {
        if(p->epoch<=s->epoch)r->status=4;
        return;
    }
    if(p->epoch!=s->epoch){r->status=4;return;}
    if(p->capacity!=s->capacity){r->status=1;return;}
    for(int k=0;k<3;k++)if(p->nsc[k]!=s->nsc[k]){r->status=1;return;}
    if(operation==2) {
        if(*convergence!=1||s->last_push==s->epoch){r->status=5;return;}
    }else {
        if(p->mode<0||p->mode>3){r->status=1;return;}
        if(!s->count){r->status=6;return;}
        if(s->epoch<=s->last_push){r->status=4;return;}
    }
}
__global__ void validate_topology(DensityHistoryConfig c,const DensityHistoryCommand* p,
        const int* num,const int* ptr,const int* col,DensityHistoryResult* r) {
    if(r->status)return;
    const int columns=c.orbitals*p->nsc[0]*p->nsc[1]*p->nsc[2];
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<c.rows;row+=gridDim.x*blockDim.x) {
        int start=ptr[row],count=num[row];
        if(start<0||start>p->capacity||count<0||count>p->capacity-start){atomicMax(&r->status,1);continue;}
        if(row) {
            int prev=ptr[row-1],length=num[row-1];
            if(prev<0||prev>p->capacity||length<0||length>p->capacity-prev||start<prev+length){atomicMax(&r->status,1);continue;}
        }
        for(int i=start;i<start+count;i++)if(col[i]<1||col[i]>columns)atomicMax(&r->status,1);
    }
}
__global__ void translate_columns(DensityHistoryConfig c,const DensityHistoryCommand* p,
        const State* s,const int* num,const int* ptr,const int* col,int* translated,const DensityHistoryResult* r) {
    if(r->status||!s->epoch)return;
    const int offset=s->topology_bank*c.rows,base=s->topology_bank*c.capacity;
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<c.rows;row+=gridDim.x*blockDim.x)
        for(int j=ptr[offset+row];j<ptr[offset+row]+num[offset+row];j++) {
            int column=col[base+j]-1,image=column/c.orbitals,unit=column%c.orbitals;
            int translated_image=0,stride=1;bool keep=true;
            for(int k=0;k<3;k++) {
                int coordinate=image%s->nsc[k];image/=s->nsc[k];
                if(coordinate>s->nsc[k]/2)coordinate-=s->nsc[k];
                if(abs(coordinate)>p->nsc[k]/2)keep=false;
                int wrapped=(coordinate%p->nsc[k]+p->nsc[k])%p->nsc[k];
                translated_image+=wrapped*stride;stride*=p->nsc[k];
            }
            translated[j]=keep?unit+1+translated_image*c.orbitals:-1;
        }
}
__global__ void make_map(DensityHistoryConfig c,const State* s,const int* newnum,const int* newptr,
        const int* newcol,const int* num,const int* ptr,const int* translated,int* map,const DensityHistoryResult* r) {
    if(r->status)return;
    const int offset=s->topology_bank*c.rows;
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<c.rows;row+=gridDim.x*blockDim.x)
        for(int j=newptr[row];j<newptr[row]+newnum[row];j++) {
            int found=-1;
            if(s->epoch)for(int old=ptr[offset+row];old<ptr[offset+row]+num[offset+row];old++)
                if(translated[old]==newcol[j])found=old; // last duplicate wins
            map[j]=found;
        }
}
__global__ void clear_remap(DensityHistoryConfig c,const State* s,double* density,int* map,const DensityHistoryResult* r) {
    if(r->status)return;
    const size_t bank=size_t(c.capacity)*c.spins*c.depth;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<bank;i+=size_t(gridDim.x)*blockDim.x)
        density[size_t(1-s->density_bank)*bank+i]=0.;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<c.capacity;i+=gridDim.x*blockDim.x)map[i]=-1;
}
__global__ void remap(DensityHistoryConfig c,const State* s,const int* map,double* density,const DensityHistoryResult* r) {
    if(r->status)return;
    const size_t slot=size_t(c.capacity)*c.spins,bank=slot*c.depth;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<slot*s->count;i+=size_t(gridDim.x)*blockDim.x) {
        int old=map[i%c.capacity];
        if(old>=0)density[size_t(1-s->density_bank)*bank+i]=density[size_t(s->density_bank)*bank+(i/c.capacity)*c.capacity+old];
    }
}
__global__ void save_topology(DensityHistoryConfig c,const DensityHistoryCommand* p,const State* s,
        const int* newnum,const int* newptr,const int* newcol,int* num,int* ptr,int* col,const DensityHistoryResult* r) {
    if(r->status)return;
    const int target=1-s->topology_bank;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<c.rows;i+=gridDim.x*blockDim.x) {
        num[target*c.rows+i]=newnum[i];ptr[target*c.rows+i]=newptr[i];
        for(int j=newptr[i];j<newptr[i]+newnum[i];j++)col[target*c.capacity+j]=newcol[j];
    }
    // Padding is never traversed; no read of caller's undefined column padding.
    (void)p;
}
__global__ void finish_advance(const DensityHistoryCommand* p,State* s,DensityHistoryResult* r) {
    if(r->status)return;
    s->epoch=p->epoch;s->capacity=p->capacity;for(int k=0;k<3;k++)s->nsc[k]=p->nsc[k];
    s->density_bank=1-s->density_bank;s->topology_bank=1-s->topology_bank;describe(s,r);
}
__global__ void validate_push(DensityHistoryConfig c,const State* s,const int* num,const int* ptr,
        const double* density,const double* xyz,DensityHistoryResult* r) {
    if(r->status)return;
    int offset=s->topology_bank*c.rows;
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<c.rows;row+=gridDim.x*blockDim.x)
        for(int j=ptr[offset+row];j<ptr[offset+row]+num[offset+row];j++)for(int spin=0;spin<c.spins;spin++)
            if(!isfinite(density[size_t(spin)*s->capacity+j]))atomicMax(&r->status,2);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<3*c.atoms;i+=gridDim.x*blockDim.x)
        if(!isfinite(xyz[i]))atomicMax(&r->status,2);
}
__global__ void push_history(DensityHistoryConfig c,const State* s,const int* num,const int* ptr,
        const double* input,const double* positions,double* density,double* xyz,const DensityHistoryResult* r) {
    if(r->status||!c.depth)return;
    const int count=min(s->count+1,c.depth),drop=s->count==c.depth?1:0,dims=3*c.atoms;
    const size_t slot=size_t(c.capacity)*c.spins,bank=slot*c.depth;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<bank;i+=size_t(gridDim.x)*blockDim.x) {
        int entry=i/slot;double value=0.;
        if(entry<count-1)value=density[size_t(s->density_bank)*bank+i+drop*slot];
        density[size_t(1-s->density_bank)*bank+i]=value;
    }
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<dims*c.depth;i+=gridDim.x*blockDim.x) {
        int entry=i/dims;double value=0.;
        if(entry<count-1)value=xyz[s->xyz_bank*dims*c.depth+i+drop*dims];
        else if(entry==count-1)value=positions[i%dims];
        xyz[(1-s->xyz_bank)*dims*c.depth+i]=value;
    }
    // New active values are installed in a separate kernel, after zeroing.
    (void)num;(void)ptr;(void)input;
}
__global__ void push_values(DensityHistoryConfig c,const State* s,const int* num,const int* ptr,
        const double* input,double* density,const DensityHistoryResult* r) {
    if(r->status||!c.depth)return;
    const size_t slot=size_t(c.capacity)*c.spins,bank=slot*c.depth;
    const size_t dest=size_t(1-s->density_bank)*bank+size_t(min(s->count,c.depth-1))*slot;
    const int offset=s->topology_bank*c.rows;
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<c.rows;row+=gridDim.x*blockDim.x)
        for(int j=ptr[offset+row];j<ptr[offset+row]+num[offset+row];j++)for(int spin=0;spin<c.spins;spin++)
            density[dest+size_t(spin)*c.capacity+j]=input[size_t(spin)*s->capacity+j];
}
__global__ void finish_push(DensityHistoryConfig c,State* s,DensityHistoryResult* r) {
    if(r->status)return;
    s->last_push=s->epoch;
    if(c.depth){s->count=min(c.depth,s->count+1);s->density_bank=1-s->density_bank;s->xyz_bank=1-s->xyz_bank;}
    describe(s,r);
}
__global__ void coefficients(DensityHistoryConfig c,const DensityHistoryCommand* p,const State* s,
        const double* xyz,const double* current,double* coeff,HistoryMathWorkspace* math,DensityHistoryResult* r) {
    if(r->status)return;
    auto info=density_history_coefficients(3*c.atoms,s->count,p->mode,
        xyz+size_t(s->xyz_bank)*3*c.atoms*c.depth,current,coeff,*math);
    r->status=info.status;r->rank=info.rank;r->fallback=info.fallback;
}
__global__ void combine(DensityHistoryConfig c,const State* s,const double* density,const double* coeff,
        double* trial,DensityHistoryResult* r) {
    if(r->status)return;
    const size_t slot=size_t(c.capacity)*c.spins,base=size_t(s->density_bank)*slot*c.depth;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(s->capacity)*c.spins;i+=size_t(gridDim.x)*blockDim.x) {
        const size_t index=(i/s->capacity)*c.capacity+i%s->capacity;
        double value=0.;for(int j=0;j<s->count;j++)value+=coeff[j]*density[base+j*slot+index];
        trial[i]=value;if(!isfinite(value))atomicMax(&r->status,2);
    }
}
__global__ void publish(DensityHistoryConfig c,const State* s,const double* trial,const double* coeff,
        double* output,double* output_coeff,const DensityHistoryResult* r) {
    if(r->status)return;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(s->capacity)*c.spins;i+=size_t(gridDim.x)*blockDim.x)output[i]=trial[i];
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<c.depth;i+=gridDim.x*blockDim.x)output_coeff[i]=i<s->count?coeff[i]:0.;
}
__global__ void inspect_state(DensityHistoryConfig c,const State* s,const double* density,const double* xyz,
        const int* num,const int* ptr,const int* col,double* out,double* positions,int* onum,int* optr,int* ocol,DensityHistoryResult* r,int out_capacity) {
    if(!blockIdx.x&&!threadIdx.x){*r=DensityHistoryResult{};describe(s,r);}
    const size_t slot=size_t(c.capacity)*c.spins,bank=slot*c.depth;
    const size_t out_slot=size_t(out_capacity)*c.spins;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<out_slot*c.depth;i+=size_t(gridDim.x)*blockDim.x)
        out[i]=i<out_slot*s->count&&i%out_capacity<size_t(c.capacity)?
            density[s->density_bank*bank+(i/out_capacity)*c.capacity+i%out_capacity]:0.;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<3*c.atoms*c.depth;i+=gridDim.x*blockDim.x)
        positions[i]=i<3*c.atoms*s->count?xyz[s->xyz_bank*3*c.atoms*c.depth+i]:0.;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<out_capacity;i+=gridDim.x*blockDim.x)ocol[i]=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<c.rows;i+=gridDim.x*blockDim.x) {
        onum[i]=s->epoch?num[s->topology_bank*c.rows+i]:0;optr[i]=s->epoch?ptr[s->topology_bank*c.rows+i]:0;
    }
    // Active columns copied by a separate launch to avoid padding races.
    (void)col;
}
__global__ void inspect_columns(DensityHistoryConfig c,const State* s,const int* num,const int* ptr,const int* col,int* output) {
    if(!s->epoch)return;
    const int offset=s->topology_bank*c.rows;
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<c.rows;row+=gridDim.x*blockDim.x)
        for(int j=ptr[offset+row];j<ptr[offset+row]+num[offset+row];j++)output[j]=col[s->topology_bank*c.capacity+j];
}
// Normalize live bank indices to zero while copying to larger strides. Only
// retained entries and active topology are read; constructor/reset padding may
// be undefined. The source remains untouched until the new owner is complete.
__global__ void grow_storage(DensityHistoryConfig old,DensityHistoryConfig next,const State* s,
        const double* density,const double* xyz,const int* num,const int* ptr,
        double* out,double* positions,int* onum,int* optr,int* ocol) {
    const size_t slot=size_t(next.capacity)*next.spins,bank=slot*next.depth;
    const size_t old_bank=size_t(old.capacity)*old.spins*old.depth;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<2*bank;i+=size_t(gridDim.x)*blockDim.x)
        out[i]=i<slot*s->count&&i%next.capacity<size_t(old.capacity)?
            density[size_t(s->density_bank)*old_bank+(i/next.capacity)*old.capacity+i%next.capacity]:0.;
    const size_t dims=size_t(3)*next.atoms,geom=dims*next.depth;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<2*geom;i+=size_t(gridDim.x)*blockDim.x)
        positions[i]=i<dims*s->count?xyz[size_t(s->xyz_bank)*geom+i]:0.;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<2*next.rows;i+=gridDim.x*blockDim.x) {
        onum[i]=i<next.rows&&s->epoch?num[s->topology_bank*old.rows+i]:0;
        optr[i]=i<next.rows&&s->epoch?ptr[s->topology_bank*old.rows+i]:0;
    }
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<2*next.capacity;i+=gridDim.x*blockDim.x)ocol[i]=0;
}
__global__ void grow_topology(DensityHistoryConfig old,const State* s,const int* num,const int* ptr,
        const int* col,int* out,State* target) {
    if(!blockIdx.x&&!threadIdx.x) {
        *target=*s;target->density_bank=target->xyz_bank=target->topology_bank=0;
    }
    if(!s->epoch)return;
    const int base=s->topology_bank*old.rows;
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<old.rows;row+=gridDim.x*blockDim.x)
        for(int j=ptr[base+row];j<ptr[base+row]+num[base+row];j++)out[j]=col[s->topology_bank*old.capacity+j];
}
}
DensityHistoryPlan::DensityHistoryPlan(DensityHistoryConfig c,cudaStream_t stream):c_(c),stream_(stream) {
    if(c.rows<1||c.orbitals<c.rows||c.atoms<1||(c.spins!=1&&c.spins!=2&&c.spins!=4&&c.spins!=8)||
       c.capacity<1||c.capacity>INT_MAX/8||c.rows>INT_MAX/2||c.depth<0||c.depth>density_history_limit||
       c.atoms>INT_MAX/(3*std::max(1,2*c.depth)))throw std::runtime_error("Invalid density history allocation");
    auto allocate=[](auto** p,size_t bytes){check(cudaMalloc(reinterpret_cast<void**>(p),std::max(size_t(1),bytes)));};
    try {
        allocate(&state_,sizeof(State));check(cudaMemsetAsync(state_,0,sizeof(State),stream_));
        allocate(&math_,sizeof(HistoryMathWorkspace));
        allocate(&density_,size_t(2)*c.capacity*c.spins*c.depth*sizeof(double));
        allocate(&xyz_,size_t(2)*3*c.atoms*c.depth*sizeof(double));
        allocate(&trial_,size_t(c.capacity)*c.spins*sizeof(double));allocate(&coefficients_,size_t(c.depth)*sizeof(double));
        allocate(&num_,size_t(2)*c.rows*sizeof(int));allocate(&ptr_,size_t(2)*c.rows*sizeof(int));
        allocate(&col_,size_t(2)*c.capacity*sizeof(int));allocate(&map_,size_t(c.capacity)*sizeof(int));
        allocate(&translated_,size_t(c.capacity)*sizeof(int));
    }catch(...){release();throw;}
}
void DensityHistoryPlan::release(){void* all[]={state_,math_,density_,xyz_,trial_,coefficients_,num_,ptr_,col_,map_,translated_};for(auto p:all)if(p)cudaFree(p);}
DensityHistoryPlan::~DensityHistoryPlan(){release();}
size_t DensityHistoryPlan::workspace_bytes()const {
    const auto c=c_;
    return sizeof(State)+sizeof(HistoryMathWorkspace)+size_t(4)*(c.rows+c.capacity)*sizeof(int)+
        (size_t(2)*c.depth*(size_t(c.capacity)*c.spins+size_t(3)*c.atoms)+size_t(c.capacity)*c.spins+c.depth)*sizeof(double)+
        (c.depth?0:3); // three minimum one-byte allocations when depth is zero
}
int DensityHistoryPlan::reset(const DensityHistoryCommand* p,DensityHistoryResult* r) {
    if(!p||!r)return int(cudaErrorInvalidValue);
    begin<<<1,1,0,stream_>>>(c_,0,p,nullptr,static_cast<State*>(state_),r);return int(cudaGetLastError());
}
int DensityHistoryPlan::advance(const DensityHistoryCommand* p,const int* num,const int* ptr,const int* col,DensityHistoryResult* r) {
    if(!p||!num||!ptr||!col||!r)return int(cudaErrorInvalidValue);
    auto s=static_cast<State*>(state_);const int blocks=std::min(256,(c_.rows-1)/threads+1);
    begin<<<1,1,0,stream_>>>(c_,1,p,nullptr,s,r);
    validate_topology<<<blocks,threads,0,stream_>>>(c_,p,num,ptr,col,r);
    clear_remap<<<256,threads,0,stream_>>>(c_,s,density_,map_,r);
    translate_columns<<<blocks,threads,0,stream_>>>(c_,p,s,num_,ptr_,col_,translated_,r);
    make_map<<<blocks,threads,0,stream_>>>(c_,s,num,ptr,col,num_,ptr_,translated_,map_,r);
    remap<<<256,threads,0,stream_>>>(c_,s,map_,density_,r);
    save_topology<<<blocks,threads,0,stream_>>>(c_,p,s,num,ptr,col,num_,ptr_,col_,r);
    finish_advance<<<1,1,0,stream_>>>(p,s,r);return int(cudaGetLastError());
}
int DensityHistoryPlan::push(const DensityHistoryCommand* p,const double* density,const double* xyz,const int* convergence,DensityHistoryResult* r) {
    if(!p||!density||!xyz||!convergence||!r)return int(cudaErrorInvalidValue);
    auto s=static_cast<State*>(state_);
    begin<<<1,1,0,stream_>>>(c_,2,p,convergence,s,r);
    validate_push<<<256,threads,0,stream_>>>(c_,s,num_,ptr_,density,xyz,r);
    push_history<<<256,threads,0,stream_>>>(c_,s,num_,ptr_,density,xyz,density_,xyz_,r);
    push_values<<<256,threads,0,stream_>>>(c_,s,num_,ptr_,density,density_,r);
    finish_push<<<1,1,0,stream_>>>(c_,s,r);return int(cudaGetLastError());
}
int DensityHistoryPlan::extrapolate(const DensityHistoryCommand* p,const double* xyz,double* density,double* coeff,DensityHistoryResult* r) {
    if(!p||!xyz||!density||!coeff||!r||density==xyz||coeff==xyz||coeff==density)return int(cudaErrorInvalidValue);
    auto s=static_cast<State*>(state_);
    begin<<<1,1,0,stream_>>>(c_,3,p,nullptr,s,r);
    coefficients<<<1,1,0,stream_>>>(c_,p,s,xyz_,xyz,coefficients_,static_cast<HistoryMathWorkspace*>(math_),r);
    combine<<<256,threads,0,stream_>>>(c_,s,density_,coefficients_,trial_,r);
    publish<<<256,threads,0,stream_>>>(c_,s,trial_,coefficients_,density,coeff,r);return int(cudaGetLastError());
}
int DensityHistoryPlan::reserve(int capacity) {
    if(capacity<1||capacity>INT_MAX/8)return int(cudaErrorInvalidValue);
    if(capacity<=c_.capacity)return int(cudaSuccess);
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(stream_,&capture));
    if(capture!=cudaStreamCaptureStatusNone)return int(cudaErrorStreamCaptureUnsupported);
    auto next=c_;next.capacity=capacity;
    DensityHistoryPlan larger(next,stream_);
    const auto s=static_cast<const State*>(state_);
    grow_storage<<<256,threads,0,stream_>>>(c_,next,s,density_,xyz_,num_,ptr_,larger.density_,larger.xyz_,larger.num_,larger.ptr_,larger.col_);
    grow_topology<<<256,threads,0,stream_>>>(c_,s,num_,ptr_,col_,larger.col_,static_cast<State*>(larger.state_));
    check(cudaGetLastError());check(cudaStreamSynchronize(stream_));
    std::swap(c_,larger.c_);
    std::swap(state_,larger.state_);std::swap(math_,larger.math_);
    std::swap(density_,larger.density_);std::swap(xyz_,larger.xyz_);
    std::swap(trial_,larger.trial_);std::swap(coefficients_,larger.coefficients_);
    std::swap(num_,larger.num_);std::swap(ptr_,larger.ptr_);std::swap(col_,larger.col_);
    std::swap(map_,larger.map_);std::swap(translated_,larger.translated_);
    return int(cudaSuccess);
}
int DensityHistoryPlan::inspect(double* density,double* xyz,int* num,int* ptr,int* col,DensityHistoryResult* r,int output_capacity) {
    if(output_capacity==0)output_capacity=c_.capacity;
    if(!density||!xyz||!num||!ptr||!col||!r||output_capacity<c_.capacity||output_capacity>INT_MAX/8)return int(cudaErrorInvalidValue);
    auto s=static_cast<State*>(state_);
    inspect_state<<<256,threads,0,stream_>>>(c_,s,density_,xyz_,num_,ptr_,col_,density,xyz,num,ptr,col,r,output_capacity);
    inspect_columns<<<256,threads,0,stream_>>>(c_,s,num_,ptr_,col_,col);return int(cudaGetLastError());
}
}
