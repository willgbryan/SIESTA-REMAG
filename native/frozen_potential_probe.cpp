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
    size_t matrix_workspace=0;
    bool moments=true,matrices=false,reference_has_soc=true,identity_completed=false;
    Point reference_control{};
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
        if(const char* option=std::getenv("REMAG_FROZEN_MATRIX_SAMPLES")) {
            require(!std::strcmp(option,"0")||!std::strcmp(option,"1"),"Invalid frozen matrix-sample option");
            matrices=!std::strcmp(option,"1");
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
        if(!identity_completed) std::fputs("null",f);
        else std::fprintf(f,"%.17g",identity_error);
        std::fprintf(f,",\"matrix_samples\":%s,\"matrix_capture_workspace_bytes\":%zu",
            matrices?"true":"false",matrix_workspace);
        std::fprintf(f,",\"lattice_operator\":\"%s\"",reference_has_soc?"full_offsite":"spin_traceless_offsite");
        std::fprintf(f,",\"reference_soc_scale\":%d,\"target_soc_scale\":1,\"reference_control\":",reference_has_soc?1:0);
        if(reference_has_soc) std::fputs("null",f);
        else {
            const auto& c=reference_control;
            std::fprintf(f,"{\"soc_scale\":0,\"completed\":%s,\"status\":%d,\"failed_k\":%d,\"solver_info\":%d,\"eigensystem_jobs\":%d,\"seconds\":",
                c.completed?"true":"false",c.status,c.k,c.solver_info,c.jobs);
            if(c.completed) number(f,c.seconds);else std::fputs("null",f);
            std::fputs(",\"raw_spectrum\":\"remag-frozen-reference.csv\"}",f);
        }
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
    void capture(FrozenPotentialPlan& plan,size_t direction,cudaStream_t stream) {
        if(!matrices) return;
        const size_t elements=4*size_t(plan.bands())*plan.bands();
        require(elements<=64ULL*1024*1024/sizeof(cuDoubleComplex),"Frozen matrix capture exceeds 64 MiB");
        const unsigned int endian=1;
        require(*reinterpret_cast<const unsigned char*>(&endian)==1,"Matrix capture requires little endian");
        matrix_workspace=elements*sizeof(cuDoubleComplex);
        Buffer<cuDoubleComplex> device(elements);
        std::vector<cuDoubleComplex> host(elements);
        std::vector<int> samples{0,plan.kpoints()/2,plan.kpoints()-1};
        samples.erase(std::unique(samples.begin(),samples.end()),samples.end());
        const auto& q=rotations[direction];
        for(int k:samples) {
            plan.capture_matrices(k,q.data(),device.p);
            download(host.data(),device.p,elements,stream);check(cudaStreamSynchronize(stream));
            char binary[128],metadata[128];
            std::snprintf(binary,sizeof(binary),"remag-frozen-matrices-%03zu-k%06d.bin",direction,k);
            std::snprintf(metadata,sizeof(metadata),"remag-frozen-matrices-%03zu-k%06d.json",direction,k);
            File data(binary);
            require(std::fwrite(host.data(),sizeof(cuDoubleComplex),elements,data.value)==elements,
                "Failed writing frozen matrix capture");
            data.close();
            File output(metadata);
            std::fprintf(output.value,"{\"schema\":1,\"reference\":\"Final SCF generator Hamiltonian\","
                "\"dtype\":\"<c16\",\"order\":\"F\",\"matrices\":[\"generator\",\"overlap\",\"soc\",\"rotated\"],"
                "\"bands\":%d,\"k\":%d,\"kpoints\":%d,\"direction\":%zu,\"quaternion\":[%.17g,%.17g,%.17g,%.17g],"
                "\"reference_soc_scale\":%d,\"target_soc_scale\":1,\"lattice_operator\":\"%s\",\"matrix_file\":\"%s\",\"raw_spectrum\":\"remag-frozen-%03zu.csv\",\"energy_unit\":\"Ry\"}\n",
                plan.bands(),k,plan.kpoints(),direction,q[0],q[1],q[2],q[3],reference_has_soc?1:0,
                reference_has_soc?"full_offsite":"spin_traceless_offsite",binary,direction);
            output.close();
        }
    }
};
}

void export_magnetic_state(const HamiltonianAssemblyView& view,const double* generator,
        const cuDoubleComplex* offsite,const FermiProblem* problem,const double* weights,
        const double* generator_spectrum,cudaStream_t stream,bool reference_has_soc) {
    // This is a requested scientific output, not an observer or a new run.
    const char* option=std::getenv("REMAG_EXPORT_MAGNETIC_STATE");
    if(!option||!std::strcmp(option,"0")) return;
    require(!std::strcmp(option,"1"),"REMAG_EXPORT_MAGNETIC_STATE must be 0 or 1");
    const unsigned int endian=1;
    require(*reinterpret_cast<const unsigned char*>(&endian)==1,"Magnetic export requires little endian");
    require(problem&&weights&&generator_spectrum,"Missing generator export data");
    FermiProblem f;
    check(cudaMemcpyAsync(&f,problem,sizeof(f),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    require(f.nspinor==2&&f.maxspn==1&&f.nk==view.shape.nk&&f.maxe==2*view.shape.n&&f.ne==f.maxe,
        "Magnetic export requires complete spinor bands");
    // A failure must not leave a previous complete manifest over new data.
    {File metadata("remag-magnetic-export.json");std::fputs("{\"schema\":1,\"complete\":false}\n",metadata.value);metadata.close();}
    // Retain the finite-range generator, not an inverse Fourier transform of
    // the SCF sampling mesh: new k meshes must not inherit coarse-mesh aliasing.
    const auto p=view.shape;
    require(p.kind==3&&p.nh==8&&view.xij,"Real-space magnetic export requires k-space spinors");
    std::vector<HamiltonianPair> pairs(p.pairs);
    size_t sparse_downloaded=0;
    auto download=[&](void* out,const void* in,size_t bytes) {
        check(cudaMemcpyAsync(out,in,bytes,cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        sparse_downloaded+=bytes;transfer(Transfer::spectral_export,false,bytes);
    };
    download(pairs.data(),view.pairs,pairs.size()*sizeof(HamiltonianPair));
    const size_t entries=pairs.back().end;
    std::vector<int> indices(entries),rows(entries),columns(entries);
    std::vector<double> sparse_h(8*size_t(p.ldh)),sparse_s(p.nnz),xij(3*size_t(p.nnz));
    std::vector<cuDoubleComplex> sparse_soc(4*size_t(p.ldh));
    download(indices.data(),view.indices,entries*sizeof(int));
    download(sparse_h.data(),generator,sparse_h.size()*sizeof(double));
    download(sparse_soc.data(),offsite,sparse_soc.size()*sizeof(cuDoubleComplex));
    download(sparse_s.data(),view.overlap,sparse_s.size()*sizeof(double));
    download(xij.data(),view.xij,xij.size()*sizeof(double));
    for(const auto& pair:pairs) for(int j=pair.begin;j<pair.end;j++) {rows[j]=pair.row;columns[j]=pair.col;}
    std::vector<double> compact_x(3*entries),compact_h(8*entries),compact_s(entries);
    std::vector<cuDoubleComplex> compact_soc(4*entries);
    for(size_t j=0;j<entries;j++) {
        const size_t i=indices[j];compact_s[j]=sparse_s[i];
        for(int c=0;c<3;c++) compact_x[3*j+c]=xij[3*i+c];
        for(int c=0;c<8;c++) compact_h[8*j+c]=sparse_h[i+size_t(p.ldh)*c];
        for(int c=0;c<4;c++) compact_soc[4*j+c]=sparse_soc[i+size_t(p.ldh)*c];
    }
    {File raw("remag-magnetic-realspace.bin");
        auto write=[&](const void* source,size_t bytes) {require(std::fwrite(source,1,bytes,raw.value)==bytes,"Real-space export write failed");};
        write(rows.data(),entries*sizeof(int));write(columns.data(),entries*sizeof(int));
        write(compact_x.data(),compact_x.size()*sizeof(double));write(compact_h.data(),compact_h.size()*sizeof(double));
        write(compact_soc.data(),compact_soc.size()*sizeof(cuDoubleComplex));write(compact_s.data(),entries*sizeof(double));raw.close();}
    {File info("remag-magnetic-realspace.json");
        std::fprintf(info.value,"{\"schema\":1,\"complete\":true,\"entries\":%zu,\"spatial_orbitals\":%d,\"device_to_host_bytes\":%zu,"
            "\"reference\":\"Final SCF generator Hamiltonian\",\"layout\":\"rows,columns,xij,generator,offsite,overlap\","
            "\"reference_has_soc\":%s,\"energy_unit\":\"Ry\",\"length_unit\":\"bohr\"}\n",
            entries,p.n,sparse_downloaded,reference_has_soc?"true":"false");info.close();}
    FrozenPotentialPlan plan(view,generator,offsite,false,256ULL*1024*1024,stream,reference_has_soc);
    const size_t matrix=size_t(f.maxe)*f.maxe;
    Buffer<cuDoubleComplex> device(4*matrix);std::vector<cuDoubleComplex> host(4*matrix);
    std::vector<double> spectrum(f.maxe);double weight,kpoint[3],identity[4]={1.,0.,0.,0.};
    File data("remag-magnetic-export.bin");
    auto write=[&](const void* source,size_t bytes) {
        require(std::fwrite(source,1,bytes,data.value)==bytes,"Failed writing magnetic generator export");
    };
    for(int k=0;k<f.nk;k++) {
        plan.capture_matrices(k,identity,device.p);
        check(cudaMemcpyAsync(host.data(),device.p,3*matrix*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(&weight,weights+k,sizeof(double),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(spectrum.data(),generator_spectrum+size_t(f.maxe)*k,f.maxe*sizeof(double),cudaMemcpyDeviceToHost,stream));
        for(double& x:kpoint) x=0.;
        if(view.kpoints) check(cudaMemcpyAsync(kpoint,view.kpoints+3*k,3*sizeof(double),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        write(host.data(),3*matrix*sizeof(cuDoubleComplex));write(kpoint,3*sizeof(double));
        write(&weight,sizeof(double));write(spectrum.data(),spectrum.size()*sizeof(double));
        transfer(Transfer::spectral_export,false,3*matrix*sizeof(cuDoubleComplex)+(spectrum.size()+4)*sizeof(double));
    }
    data.close();
    File metadata("remag-magnetic-export.json");
    std::fprintf(metadata.value,"{\"schema\":1,\"complete\":true,\"reference\":\"Final SCF generator Hamiltonian\","
        "\"kpoints\":%d,\"bands\":%d,\"reference_has_soc\":%s,\"electrons\":%.17g,\"temperature_ry\":%.17g,"
        "\"dtype\":\"<c16\",\"order\":\"F\",\"matrices\":[\"generator\",\"overlap\",\"offsite\"]}\n",
        f.nk,f.maxe,reference_has_soc?"true":"false",f.charge,f.temperature);
    metadata.close();
}

void frozen_potential_probe(const HamiltonianAssemblyView& view,const double* generator,
        const cuDoubleComplex* offsite,const FermiProblem* problem,const double* weights,
        const double* generator_spectrum,cudaStream_t stream,bool reference_has_soc) {
    const char* path=std::getenv("REMAG_FROZEN_ROTATIONS");if(!path) return;
    require(*path,"REMAG_FROZEN_ROTATIONS cannot be empty");
    Probe probe(path);probe.reference_has_soc=reference_has_soc;
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
        FrozenPotentialPlan plan(view,generator,offsite,probe.moments,256ULL*1024*1024,stream,reference_has_soc);
        probe.workspace=plan.workspace_bytes();
        Buffer<double> occupation(count);Buffer<FermiResult> fd(1);
        std::vector<double> w(f.nk),scf(count),e(count),q(count),spin(probe.moments?3*count:0),ref_e,ref_q;
        probe.download(w.data(),weights,w.size(),stream);probe.download(scf.data(),generator_spectrum,count,stream);
        check(cudaStreamSynchronize(stream));
        for(double value:scf) require(std::isfinite(value),"Nonfinite SCF reference spectrum");
        probe.report("running");
        if(!reference_has_soc) {
            // Point zero below restores SOC, so it is NOT the SCF identity
            // control. Preserve and count this extra SOC-free solve explicitly.
            auto& c=probe.reference_control;const auto start=std::chrono::steady_clock::now();
            plan.solve(probe.rotations[0].data(),false);c.jobs=plan.eigensystem_jobs();
            FrozenSpectrumStatus status;
            probe.download(&status,plan.status(),1,stream);
            probe.download(e.data(),plan.energies(),count,stream);check(cudaStreamSynchronize(stream));
            c.status=status.status;c.k=status.k;c.solver_info=status.solver_info;
            require(!c.status,"SOC-free reference eigensystem failed");
            File raw("remag-frozen-reference.csv");
            std::fputs("k,band,energy_ry,generator_scf_energy_ry\n",raw.value);
            for(size_t i=0;i<count;i++) {
                require(std::isfinite(e[i]),"Nonfinite SOC-free identity spectrum");
                probe.identity_error=std::max(probe.identity_error,std::abs(e[i]-scf[i]));
                std::fprintf(raw.value,"%zu,%zu,%.17g,%.17g\n",i/f.maxe,i%f.maxe,e[i],scf[i]);
            }
            raw.close();c.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            c.completed=true;probe.identity_completed=true;probe.report("running");
            require(probe.identity_error*13.605693122994<=1.e-7,"SOC-free SCF identity spectrum differs");
        }
        for(size_t direction=0;direction<probe.rotations.size();direction++) {
            probe.points.emplace_back();auto& p=probe.points.back();
            const auto start=std::chrono::steady_clock::now();
            plan.solve(probe.rotations[direction].data());
            p.jobs=plan.eigensystem_jobs();
            FrozenSpectrumStatus result;
            probe.download(&result,plan.status(),1,stream);check(cudaStreamSynchronize(stream));
            p.status=result.status;p.k=result.k;p.solver_info=result.solver_info;
            if(result.status==3&&result.band>=0) {
                std::vector<cuDoubleComplex> c(size_t(plan.bands())*plan.bands()),s(c.size());
                probe.download(c.data(),plan.failed_vectors(),c.size(),stream);
                probe.download(s.data(),plan.failed_overlap(),s.size(),stream);
                check(cudaStreamSynchronize(stream));
                for(const auto& item: {std::make_pair("remag-failed-frozen-c.bin",&c),
                                      std::make_pair("remag-failed-frozen-s.bin",&s)}) {
                    File output(item.first);
                    require(std::fwrite(item.second->data(),sizeof(cuDoubleComplex),c.size(),output.value)==c.size(),
                        "Failed writing frozen spin diagnostic");
                    output.close();
                }
                File output("remag-frozen-failure.json");
                std::fprintf(output.value,"{\"k\":%d,\"band\":%d,\"norm\":",result.k,result.band);
                number(output.value,result.norm);std::fputs(",\"spin_squared\":",output.value);
                number(output.value,result.spin_squared);std::fputs("}\n",output.value);output.close();
            }
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
                if(reference_has_soc) {
                    for(size_t i=0;i<count;i++) probe.identity_error=std::max(probe.identity_error,std::abs(e[i]-scf[i]));
                    probe.identity_completed=true;
                }
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
            probe.capture(plan,direction,stream);
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
