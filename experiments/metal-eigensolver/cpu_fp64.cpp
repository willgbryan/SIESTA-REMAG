// SPDX-License-Identifier: GPL-3.0-or-later
// Direct Accelerate baseline, with independent per-worker LAPACK workspaces.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <complex>
#include <cstring>
#include <memory>
#include <vector>
using Z=__LAPACK_double_complex;
using I=__LAPACK_int;
static_assert(sizeof(Z)==16 && sizeof(I)==4,"Expected LP64 Accelerate complex ABI");
struct Work {std::vector<Z> b,work; std::vector<double> real;std::vector<I> integer;};
struct CpuPlan {I n,lwork,lrwork,liwork;unsigned workers;std::vector<Work> lane;};
extern "C" void* cpu_fp64_create(unsigned n,unsigned workers) {
    if(n<2 || n>128 || (workers!=1 && workers!=4))return nullptr;
    try {
        auto p=std::make_unique<CpuPlan>();p->n=n;p->workers=workers;
        I type=1,query=-1,info=0,integer;char vectors='V',lower='L';Z work;double real;
        std::vector<Z> a(n*n),b(n*n);std::vector<double> w(n);
        zhegvd_(&type,&vectors,&lower,&p->n,a.data(),&p->n,b.data(),&p->n,w.data(),
            &work,&query,&real,&query,&integer,&query,&info);
        if(info)return nullptr;
        p->lwork=I(work.real());p->lrwork=I(real);p->liwork=integer;
        zheevd_(&vectors,&lower,&p->n,a.data(),&p->n,w.data(),&work,&query,&real,&query,&integer,&query,&info);
        if(info)return nullptr;
        p->lwork=std::max(p->lwork,I(work.real()));p->lrwork=std::max(p->lrwork,I(real));p->liwork=std::max(p->liwork,integer);
        p->lane.resize(workers);
        for(auto& lane:p->lane) {
            lane.b.resize(n*n);lane.work.resize(p->lwork);lane.real.resize(p->lrwork);lane.integer.resize(p->liwork);
        }
        return p.release();
    } catch(...) {return nullptr;}
}
extern "C" void cpu_fp64_destroy(void* p) {delete static_cast<CpuPlan*>(p);}
struct Batch {CpuPlan* plan;const Z *h,*s;Z* v;double* w;I* status;unsigned count;bool cached;};
static void solve_lane(void* pointer,size_t id) {
    auto& batch=*static_cast<Batch*>(pointer);auto& p=*batch.plan;auto& lane=p.lane[id];
    I type=1;char vectors='V',lower='L';Z one(1,0);size_t nn=size_t(p.n)*p.n;
    for(unsigned job=unsigned(id);job<batch.count;job+=p.workers) {
        Z* a=batch.v+job*nn;double* w=batch.w+job*p.n;I* info=batch.status+job;
        std::memcpy(a,batch.h+job*nn,nn*sizeof(Z));std::memcpy(lane.b.data(),batch.s+job*nn,nn*sizeof(Z));
        if(batch.cached) {
            zhegst_(&type,&lower,&p.n,a,&p.n,lane.b.data(),&p.n,info);
            if(*info)continue;
            zheevd_(&vectors,&lower,&p.n,a,&p.n,w,lane.work.data(),&p.lwork,lane.real.data(),&p.lrwork,lane.integer.data(),&p.liwork,info);
            if(!*info)cblas_ztrsm(CblasColMajor,CblasLeft,CblasLower,CblasConjTrans,CblasNonUnit,
                p.n,p.n,&one,lane.b.data(),p.n,a,p.n);
        } else {
            zhegvd_(&type,&vectors,&lower,&p.n,a,&p.n,lane.b.data(),&p.n,w,lane.work.data(),&p.lwork,
                lane.real.data(),&p.lrwork,lane.integer.data(),&p.liwork,info);
        }
    }
}
extern "C" int cpu_fp64_solve(void* pointer,const void* h,const void* s,unsigned count,int cached,void* v,double* w,int* status) {
    if(!pointer || !h || !s || !v || !w || !status || !count || count>512 || (cached!=0 && cached!=1))return 1;
    auto p=static_cast<CpuPlan*>(pointer);
    Batch batch{p,static_cast<const Z*>(h),static_cast<const Z*>(s),static_cast<Z*>(v),w,status,count,cached!=0};
    if(p->workers==1)solve_lane(&batch,0);
    else dispatch_apply_f(p->workers,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),&batch,solve_lane);
    return 0;
}
