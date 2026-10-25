// SPDX-License-Identifier: GPL-3.0-or-later
#include "frozen_potential_probe.hpp"
#include "frozen_potential_math.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace remag {
namespace {
void check(cudaError_t code) {
    if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void require(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}
void number(FILE* f,double x) {
    if(std::isfinite(x)) std::fprintf(f,"%.17g",x);
    else std::fputs("null",f);
}
struct File {
    FILE* value;
    explicit File(const char* path):value(std::fopen(path,"w")) {
        require(value,"Cannot create frozen-potential evidence");
    }
    ~File() {if(value) std::fclose(value);}
    void close() {
        FILE* f=value;value=nullptr;
        const bool failed=std::ferror(f)!=0;
        const int code=std::fclose(f);
        require(!failed&&code==0,"Failed writing frozen-potential evidence");
    }
};
template<class T> struct Buffer {
    T* p=nullptr;
    explicit Buffer(size_t n) {check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    ~Buffer() {if(p) cudaFree(p);}
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
};
struct Probe {
    std::vector<std::array<double,4>> rotations{{1.,0.,0.,0.}};
    struct Point {
        int status=0,k=-1,solver_info=0,jobs=0;
        FermiResult fermi{};
        mae::BandDifference difference{};
        double spin[3]{},seconds=0.;
        bool completed=false;
    };
    std::vector<Point> points;
    FermiProblem fermi{};
    size_t workspace=0,downloaded=0;
    bool moments=true;
    double identity_error=0.;

    Probe(const char* path) {
        std::ifstream input(path);require(bool(input),"Cannot read REMAG_FROZEN_ROTATIONS");
        std::string line;
        while(std::getline(input,line)) {
            line=line.substr(0,line.find('#'));
            std::istringstream row(line);row>>std::ws;if(row.eof()) continue;
            std::array<double,4> q{};mae::SpinRotation rotation;
            require(bool(row>>q[0]>>q[1]>>q[2]>>q[3]),"Rotation rows require four quaternion components");
            row>>std::ws;
            require(row.eof()&&mae::spin_rotation(q.data(),rotation),"Invalid frozen-potential quaternion");
            require(rotations.size()<17,"Qualification probe permits at most 16 requested rotations");
            rotations.push_back(q);
        }
        require(input.eof()&&rotations.size()>1,"No rotations requested or rotation file read failed");
        if(const char* option=std::getenv("REMAG_FROZEN_SPIN_EXPECTATIONS")) {
            require(!std::strcmp(option,"0")||!std::strcmp(option,"1"),"Invalid frozen spin-expectation option");
            moments=!std::strcmp(option,"1");
        }
    }
    template<class T> void download(T* out,const T* input,size_t n,cudaStream_t stream) {
        check(cudaMemcpyAsync(out,input,n*sizeof(T),cudaMemcpyDeviceToHost,stream));
        downloaded+=n*sizeof(T);transfer(Transfer::spectral_export,false,n*sizeof(T));
    }
    void report(const char* state) const {
        File output("remag-frozen-potential.json");auto* f=output.value;
        std::fprintf(f,"{\"schema\":1,\"status\":\"%s\",\"scope\":\"Experimental common-generator frozen exchange with fixed full offsite SOC; band differences, not full-SCF total energies\","
            "\"reference\":\"Final SCF generator Hamiltonian\",\"energy_unit\":\"Ry\",\"temperature_ry\":",
            state);
        number(f,fermi.temperature);std::fputs(",\"electrons\":",f);number(f,fermi.charge);
        std::fprintf(f,","
            "\"bands\":%d,\"kpoints\":%d,\"solver_lanes\":1,\"spin_expectations\":%s,\"explicit_plan_workspace_bytes\":%zu,"
            "\"qualification_download_bytes\":%zu,\"generator_spectrum_max_abs_error_ry\":",
            fermi.maxe,fermi.nk,moments?"true":"false",workspace,downloaded);
        if(points.empty()||!points[0].completed) std::fputs("null",f);
        else std::fprintf(f,"%.17g",identity_error);
        std::fputs(",\"physical_accuracy_qualified\":false,\"points\":[",f);
        for(size_t i=0;i<points.size();i++) {
            const auto& p=points[i];const auto& q=rotations[i];const auto& d=p.difference;
            std::fprintf(f,"%s{\"index\":%zu,\"quaternion\":[%.17g,%.17g,%.17g,%.17g],\"status\":%d,\"failed_k\":%d,\"solver_info\":%d,\"eigensystem_jobs\":%d,\"completed\":%s,\"seconds\":",
                i?",":"",i,q[0],q[1],q[2],q[3],p.status,p.k,p.solver_info,p.jobs,p.completed?"true":"false");
            if(p.completed) number(f,p.seconds);else std::fputs("null",f);
            if(p.completed) {
                std::fprintf(f,",\"fermi_ry\":%.17g,\"charge\":%.17g,\"band_difference_ry\":%.17g,\"band_free_difference_ry\":%.17g,"
                    "\"canonical_free_difference_ry\":%.17g,\"charge_error_reference\":%.17g,\"charge_error_current\":%.17g,\"electron_count_passed\":%s,\"spin_muB\":",
                    p.fermi.fermi,p.fermi.charge,d.band_energy,d.band_free_energy,d.canonical_free_energy,
                    d.charge_error_a,d.charge_error_b,d.electron_count_passed?"true":"false");
                if(moments) std::fprintf(f,"[%.17g,%.17g,%.17g]",p.spin[0],p.spin[1],p.spin[2]);
                else std::fputs("null",f);
                std::fprintf(f,",\"raw_spectrum\":\"remag-frozen-%03zu.csv\"",i);
            }
            std::fputc('}',f);
        }
        std::fputs("]}\n",f);output.close();
    }
    void raw(size_t index,const std::vector<double>& e,const std::vector<double>& q,
             const std::vector<double>& spin,const std::vector<double>& scf,const std::vector<double>& weights) const {
        char name[64];std::snprintf(name,sizeof(name),"remag-frozen-%03zu.csv",index);
        File out(name);auto* f=out.value;
        std::fputs("k,band,energy_ry,occupation,spin_x,spin_y,spin_z,generator_scf_energy_ry,k_weight\n",f);
        for(size_t i=0;i<e.size();i++) {
            std::fprintf(f,"%zu,%zu,%.17g,%.17g,",i/fermi.maxe,i%fermi.maxe,e[i],q[i]);
            if(moments) std::fprintf(f,"%.17g,%.17g,%.17g,",spin[3*i],spin[3*i+1],spin[3*i+2]);
            else std::fputs(",,,",f);
            std::fprintf(f,"%.17g,%.17g\n",scf[i],weights[i/fermi.maxe]);
        }
        out.close();
    }
};
}

void frozen_potential_probe(const HamiltonianAssemblyView& view,const double* generator,
        const cuDoubleComplex* offsite,const FermiProblem* problem,const double* weights,
        const double* generator_spectrum,cudaStream_t stream) {
    const char* path=std::getenv("REMAG_FROZEN_ROTATIONS");if(!path) return;
    require(*path,"REMAG_FROZEN_ROTATIONS cannot be empty");
    Probe probe(path);
    try {
        require(problem&&weights&&generator_spectrum,"Missing frozen-potential reference data");
        probe.download(&probe.fermi,problem,1,stream);check(cudaStreamSynchronize(stream));
        const auto& f=probe.fermi;
        require(f.nspinor==2&&f.maxspn==1&&f.nk==view.shape.nk&&f.maxe==2*view.shape.n&&f.ne==f.maxe&&
            std::isfinite(f.temperature)&&f.temperature>=1.e-6&&std::isfinite(f.charge)&&f.charge>0.,
            "Frozen probe requires full spinor bands and supported finite-temperature FD occupations");
        const size_t count=size_t(f.maxe)*f.nk;
        // This qualification probe's raw export is deliberately bounded. It
        // is not a capacity guard on the lower-level spectral executor.
        require(count<=4*1024*1024,"Frozen qualification raw-output limit exceeded");
        FrozenPotentialPlan plan(view,generator,offsite,probe.moments,256ULL*1024*1024,stream);
        probe.workspace=plan.workspace_bytes();
        Buffer<double> occupation(count);Buffer<FermiResult> fd(1);
        std::vector<double> w(f.nk),scf(count),e(count),q(count),spin(probe.moments?3*count:0),ref_e,ref_q;
        probe.download(w.data(),weights,w.size(),stream);probe.download(scf.data(),generator_spectrum,count,stream);
        check(cudaStreamSynchronize(stream));
        for(double value:scf) require(std::isfinite(value),"Nonfinite SCF reference spectrum");
        probe.report("running");
        for(size_t direction=0;direction<probe.rotations.size();direction++) {
            probe.points.emplace_back();auto& p=probe.points.back();
            const auto start=std::chrono::steady_clock::now();
            plan.solve(probe.rotations[direction].data());
            p.jobs=plan.eigensystem_jobs();
            FrozenSpectrumStatus result;
            probe.download(&result,plan.status(),1,stream);check(cudaStreamSynchronize(stream));
            p.status=result.status;p.k=result.k;p.solver_info=result.solver_info;
            require(!p.status,"Frozen eigensystem or spin normalization failed");
            check(fermi_device(problem,plan.energies(),weights,occupation.p,fd.p,stream));
            probe.download(&p.fermi,fd.p,1,stream);
            probe.download(e.data(),plan.energies(),count,stream);probe.download(q.data(),occupation.p,count,stream);
            if(probe.moments) probe.download(spin.data(),plan.spin_expectations(),3*count,stream);
            check(cudaStreamSynchronize(stream));
            p.status=p.fermi.status?10+p.fermi.status:0;
            require(!p.status,"Frozen occupations failed");
            if(!direction) {
                ref_e=e;ref_q=q;
                for(size_t i=0;i<count;i++) probe.identity_error=std::max(probe.identity_error,std::abs(e[i]-scf[i]));
                require(std::isfinite(probe.identity_error),"Invalid reference spectrum discrepancy");
            }
            mae::BandPairSum sum(probe.points[0].fermi.fermi);mae::CompensatedSum moment[3];
            for(size_t i=0;i<count;i++) {
                sum.add(ref_e[i],e[i],ref_q[i],q[i],w[i/f.maxe]);
                if(probe.moments) for(int c=0;c<3;c++) moment[c].add(q[i]*spin[3*i+c]);
            }
            p.difference=sum.finish(f.temperature,probe.points[0].fermi.fermi,p.fermi.fermi,f.charge);
            for(int c=0;c<3;c++) p.spin[c]=moment[c].value();
            require(p.difference.finite_result&&std::all_of(p.spin,p.spin+3,[](double x){return std::isfinite(x);}),
                "Frozen directional reduction failed");
            probe.raw(direction,e,q,spin,scf,w);
            p.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();p.completed=true;
            probe.report("running");
            require(p.difference.electron_count_passed,"Frozen compensated electron-count check failed");
        }
        probe.report("complete");
    } catch(...) {
        if(!probe.points.empty()&&!probe.points.back().completed&&!probe.points.back().status)
            probe.points.back().status=99;
        probe.report("failed");throw;
    }
}
}
