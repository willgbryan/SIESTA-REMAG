// SPDX-License-Identifier: GPL-3.0-or-later
// Explicit, per-diagonalization ownership of spectra and occupation weights.
#include "occupations.hpp"
#include "resident.hpp"
#include "execution.hpp"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" int remag_dm_enabled();
namespace {
void check(cudaError_t code) { if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
bool verify() {
    const char* flag=std::getenv("REMAG_GRID_VERIFY_EPOCH");
    return flag&&std::strcmp(flag,"1")==0;
}
struct Counts {
    unsigned long long calls=0,prepares=0,ends=0,captures=0,borrows=0,staging=0,h2d=0,d2h=0,d2d=0,avoided=0,checked=0;
    unsigned long long deferred_calls=0,exports=0,avoided_q_d2h=0;
    double seconds=0.;
    ~Counts() {
        if(FILE* f=std::fopen("remag-occupations.json","w")) {
            std::fprintf(f,"{\"device_occupation_calls\":%llu,\"prepares\":%llu,\"ends\":%llu,"
                "\"spectrum_captures\":%llu,\"resident_input_borrows\":%llu,\"setup_uploads\":%llu,"
                "\"host_to_device_bytes\":%llu,\"device_to_host_bytes\":%llu,\"device_to_device_bytes\":%llu,"
                "\"avoided_occupation_energy_upload_bytes\":%llu,\"debug_host_validation_bytes\":%llu,"
                "\"deferred_occupation_calls\":%llu,\"deferred_spectral_exports\":%llu,"
                "\"avoided_occupation_device_to_host_bytes\":%llu,"
                "\"wall_seconds\":%.9f,\"scope\":\"explicit occupation bridge copies; host exports remain\"}\n",
                calls,prepares,ends,captures,borrows,staging,h2d,d2h,d2d,avoided,checked,
                deferred_calls,exports,avoided_q_d2h,seconds);
            std::fclose(f);
        }
    }
} counts;
struct Buffer {
    void* p=nullptr;
    size_t bytes=0;
    void reserve(size_t n) {
        if(n<=bytes) return;
        if(p) { check(cudaFree(p));p=nullptr;bytes=0; }
        check(cudaMalloc(&p,n));bytes=n;
    }
    void upload(const void* source,size_t n) {
        reserve(n);check(cudaMemcpy(p,source,n,cudaMemcpyHostToDevice));
        counts.h2d+=n;remag::transfer(remag::Transfer::occupation_setup,true,n);
    }
    ~Buffer() { if(p) cudaFree(p); }
};
struct State {
    bool active=false,computed=false,configured=false,resident=false,pending_export=false;
    remag::FermiProblem problem{};
    uint64_t epoch=0;
    const double *host_e=nullptr,*host_q=nullptr,*weight_source=nullptr;
    Buffer config,weights,energies,occupations,result;
    std::vector<int> available;
    std::vector<bool> consumed;
    std::vector<double> weight_snapshot,e_snapshot,q_snapshot;
} state;
bool same(const remag::FermiProblem& a,const remag::FermiProblem& b) {
    return a.nspinor==b.nspinor&&a.maxspn==b.maxspn&&a.nk==b.nk&&a.maxe==b.maxe&&a.ne==b.ne&&
           a.temperature==b.temperature&&a.charge==b.charge;
}
int block(const double* pointer,const double* base,int n) {
    uintptr_t a=reinterpret_cast<uintptr_t>(pointer),b=reinterpret_cast<uintptr_t>(base);
    size_t stride=size_t(state.problem.maxe)*sizeof(double),blocks=state.available.size();
    if(a<b||(a-b)%stride||((a-b)/stride)>=blocks||n<0||n>state.problem.maxe) return -1;
    return static_cast<int>((a-b)/stride);
}
}

extern "C" int remag_fermi_enabled() {
    const char* flag=std::getenv("REMAG_OCCUPATIONS");
    return flag&&std::strcmp(flag,"1")==0&&remag_dm_enabled();
}
extern "C" int remag_spectra_enabled() {
    const char* flag=std::getenv("REMAG_RESIDENT_SPECTRA");
    return flag&&std::strcmp(flag,"1")==0&&remag_fermi_enabled();
}
extern "C" int remag_spectra_armed() { return state.active&&state.resident?1:0; }
bool remag_spectrum_export_required() { return !remag_spectra_armed(); }
extern "C" int remag_fermi_prepare(int nspinor,int maxspn,int nk,int maxe,int ne,
        const double* host_e,const double* host_q,const double* weights,double temperature,double charge) {
    try {
        if(!remag_fermi_enabled()||state.active||!host_e||!host_q||!weights||nk<1||maxe<1||ne<1||ne>maxe||
           maxspn<1||maxspn>2||(nspinor!=1&&nspinor!=2)||(nspinor==1&&maxspn!=1)||
           static_cast<long long>(maxe)*maxspn*nk>INT_MAX) return -150;
        remag::FermiProblem p{nspinor,maxspn,nk,maxe,ne,temperature,charge};
        auto epoch=remag::geometry_epoch();
        bool resident=remag_spectra_enabled();
        // Iterations can replace the same geometry's unpublished spectrum.
        // A new owner/layout/geometry must explicitly export the old one first.
        if(state.pending_export&&(!resident||state.epoch!=epoch||state.host_e!=host_e||state.host_q!=host_q||
           state.problem.maxe!=maxe||state.problem.maxspn!=maxspn||state.problem.nk!=nk)) return -157;
        if(state.pending_export&&verify()) {
            size_t bytes=size_t(maxe)*maxspn*nk*sizeof(double);
            counts.checked+=2*bytes;
            if(std::memcmp(host_e,state.e_snapshot.data(),bytes)||std::memcmp(host_q,state.q_snapshot.data(),bytes)) return -159;
        }
        bool reuse=state.configured&&same(state.problem,p)&&state.epoch==epoch&&state.weight_source==weights;
        if(reuse&&verify()) {
            counts.checked+=size_t(nk)*sizeof(double);
            if(std::memcmp(weights,state.weight_snapshot.data(),size_t(nk)*sizeof(double))) return -151;
        }
        if(!reuse) {
            state.config.upload(&p,sizeof(p));state.weights.upload(weights,size_t(nk)*sizeof(double));
            state.weight_snapshot.assign(weights,weights+nk);
            state.problem=p;state.epoch=epoch;state.weight_source=weights;state.configured=true;counts.staging++;
        }
        size_t size=size_t(maxe)*maxspn*nk;
        state.energies.reserve(size*sizeof(double));state.occupations.reserve(size*sizeof(double));
        state.result.reserve(sizeof(remag::FermiResult));
        state.available.assign(size_t(maxspn)*nk,0);state.consumed.assign(size_t(maxspn)*nk,false);
        if(verify()) {
            state.e_snapshot.resize(size);state.q_snapshot.resize(size);
            if(resident) {
                // Host arrays are opaque placeholders until explicit export;
                // copy bytes, never evaluate possibly uninitialized doubles.
                std::memcpy(state.e_snapshot.data(),host_e,size*sizeof(double));
                std::memcpy(state.q_snapshot.data(),host_q,size*sizeof(double));
            }
        }
        state.resident=resident;
        state.host_e=host_e;state.host_q=host_q;state.computed=false;state.active=true;counts.prepares++;
        return 0;
    } catch(const std::exception& e) {
        state.active=false;std::fprintf(stderr,"REMAG occupation prepare: %s\n",e.what());return -152;
    }
}
void remag_capture_spectrum(const double* host,const double* device,int n) {
    // Sampled-k paths solve twice: E-only into eo, then vectors into a scratch
    // eigenvalue array. Upstream's DM weights use the FIRST spectrum in eo.
    // Freeze the bank once occupations are computed, including during pass 2.
    if(!state.active||state.computed) return;
    int index=block(host,state.host_e,n);
    if(index<0||n<state.problem.ne||state.epoch!=remag::geometry_epoch()||state.available[index])
        throw std::runtime_error("Spectrum producer does not match declared occupation bank");
    auto output=static_cast<double*>(state.energies.p)+size_t(index)*state.problem.maxe;
    check(cudaMemcpy(output,device,size_t(n)*sizeof(double),cudaMemcpyDeviceToDevice));
    counts.d2d+=size_t(n)*sizeof(double);counts.captures++;state.available[index]=n;
    if(verify()&&!state.resident) std::copy_n(host,n,state.e_snapshot.data()+size_t(index)*state.problem.maxe);
}
extern "C" int remag_fermi_armed() { return state.active?1:0; }
extern "C" int remag_fermi_compute(int nspinor,int maxspn,int nk,int maxe,int ne,const double* host_e,
        const double* weights,double temperature,double charge,double* host_q,double* ef,double* entropy) {
    auto start=std::chrono::steady_clock::now();
    try {
        remag::FermiProblem p{nspinor,maxspn,nk,maxe,ne,temperature,charge};
        if(!state.active||state.computed||!same(state.problem,p)||state.host_e!=host_e||state.host_q!=host_q||
           state.epoch!=remag::geometry_epoch()||!weights||!ef||!entropy) return -153;
        if(std::any_of(state.available.begin(),state.available.end(),[ne](int n) { return n<ne; })) return -154;
        if(verify()) {
            counts.checked+=size_t(nk)*sizeof(double);
            if(std::memcmp(weights,state.weight_snapshot.data(),size_t(nk)*sizeof(double))) return -151;
            if(state.resident) {
                size_t bytes=size_t(maxe)*maxspn*nk*sizeof(double);
                counts.checked+=2*bytes;
                if(std::memcmp(host_e,state.e_snapshot.data(),bytes)||std::memcmp(host_q,state.q_snapshot.data(),bytes))
                    return -159;
            }
        }
        check(remag::fermi_device(static_cast<const remag::FermiProblem*>(state.config.p),
            static_cast<const double*>(state.energies.p),static_cast<const double*>(state.weights.p),
            static_cast<double*>(state.occupations.p),static_cast<remag::FermiResult*>(state.result.p),nullptr));
        remag::FermiResult result{};
        check(cudaMemcpy(&result,state.result.p,sizeof(result),cudaMemcpyDeviceToHost));
        counts.d2h+=sizeof(result);remag::transfer(remag::Transfer::occupation_result,false,sizeof(result));
        if(result.status) return -160-result.status;
        size_t n=size_t(maxe)*maxspn*nk;
        if(state.resident) {
            state.pending_export=true;counts.deferred_calls++;counts.avoided_q_d2h+=n*sizeof(double);
        } else {
            check(cudaMemcpy(host_q,state.occupations.p,n*sizeof(double),cudaMemcpyDeviceToHost));
            counts.d2h+=n*sizeof(double);remag::transfer(remag::Transfer::occupation_outputs,false,n*sizeof(double));
        }
        *ef=result.fermi;*entropy=result.entropy;
        if(verify()&&!state.resident) std::copy_n(host_q,n,state.q_snapshot.data());
        state.computed=true;counts.calls++;
        counts.seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"REMAG occupation solve: %s\n",e.what());return -155; }
}
bool remag_retained_occupations(const double* host_q,const double* host_e,int n,const double*& q,const double*& e) {
    if(!state.active) return false;
    int qi=block(host_q,state.host_q,n),ei=block(host_e,state.host_e,n);
    if(!state.computed||qi<0||ei!=qi||state.available[qi]<n||state.consumed[qi]||state.epoch!=remag::geometry_epoch())
        throw std::runtime_error("Occupation consumer does not match current spectrum bank");
    size_t offset=size_t(qi)*state.problem.maxe;
    if(verify()) {
        counts.checked+=2*size_t(n)*sizeof(double);
        if(std::memcmp(host_q,state.q_snapshot.data()+offset,size_t(n)*sizeof(double))||
           std::memcmp(host_e,state.e_snapshot.data()+offset,size_t(n)*sizeof(double)))
            throw std::runtime_error("Host modified an owned occupation or spectrum block");
    }
    q=static_cast<const double*>(state.occupations.p)+offset;e=static_cast<const double*>(state.energies.p)+offset;
    state.consumed[qi]=true;counts.borrows++;counts.avoided+=2*size_t(n)*sizeof(double);
    return true;
}
extern "C" int remag_fermi_end() {
    if(!state.active||!state.computed) return -156;
    if(state.resident&&std::any_of(state.consumed.begin(),state.consumed.end(),[](bool used){return !used;})) return -156;
    state.active=false;counts.ends++;return 0;
}
const remag::FermiResult* remag_fermi_result() {
    if(state.active||!state.pending_export||!state.computed||!state.resident||state.epoch!=remag::geometry_epoch())
        throw std::runtime_error("FD energy result has no completed owner");
    return static_cast<const remag::FermiResult*>(state.result.p);
}
extern "C" int remag_fermi_export(int size,double* host_e,double* host_q) {
    try {
        if(state.active||!state.pending_export||!state.computed||!state.resident||state.epoch!=remag::geometry_epoch()||
           size!=state.problem.maxe*state.problem.maxspn*state.problem.nk||state.host_e!=host_e||state.host_q!=host_q)
            return -158;
        size_t bytes=size_t(size)*sizeof(double);
        if(verify()) {
            counts.checked+=2*bytes;
            if(std::memcmp(host_e,state.e_snapshot.data(),bytes)||std::memcmp(host_q,state.q_snapshot.data(),bytes)) return -159;
        }
        auto download=[](double* host,const double* device,size_t n) {
            check(cudaMemcpy(host,device,n*sizeof(double),cudaMemcpyDeviceToHost));
            counts.d2h+=n*sizeof(double);remag::transfer(remag::Transfer::spectral_export,false,n*sizeof(double));
        };
        auto device_e=static_cast<const double*>(state.energies.p);
        if(std::all_of(state.available.begin(),state.available.end(),[](int n){return n==state.problem.maxe;})) {
            download(host_e,device_e,size);
        } else {
            // Preserve untouched host padding outside each producer's spectrum.
            for(size_t i=0;i<state.available.size();i++) {
                size_t offset=i*state.problem.maxe;
                download(host_e+offset,device_e+offset,state.available[i]);
            }
        }
        download(host_q,static_cast<const double*>(state.occupations.p),size);
        state.pending_export=false;counts.exports++;return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"REMAG spectral export: %s\n",e.what());return -158; }
}
