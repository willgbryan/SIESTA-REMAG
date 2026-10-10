// SPDX-License-Identifier: GPL-3.0-or-later
// Host control-plane metadata. No file writes occur until process shutdown.
#include "execution.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Optional preload-only CUPTI observer. Production runs do not load CUPTI.
extern "C" void remag_activity_phase(int,int,int) __attribute__((weak));
extern "C" void remag_activity_finish() __attribute__((weak));

namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t boundaries = static_cast<size_t>(remag::Transfer::count);
constexpr const char* names[] = {
    "solver_matrices", "solver_eigenvalues", "solver_eigenvectors", "solver_status",
    "grid_topology_basis", "grid_density_input", "grid_potential_input", "grid_sparse_input", "grid_output",
    "dm_topology", "dm_occupations", "dm_energy_weights", "dm_outputs",
    "normalization_overlap", "normalization_status",
    "occupation_setup", "occupation_result", "occupation_outputs", "spectral_export",
    "hs_geometry", "sparse_hamiltonian_input", "hartree_geometry", "hartree_density_input", "hartree_potential_output", "hartree_result",
    "grid_flow_geometry", "grid_flow_result", "sparse_state_setup", "sparse_state_scalars",
    "sparse_state_mixer_status", "sparse_state_export", "solver_graph_setup", "final_density_output", "orbital_basis_setup", "orbital_basis_shadow", "atomic_basis_setup", "atomic_basis_shadow", "two_center_setup", "two_center_shadow", "pair_force_shadow", "nonlocal_geometry", "nonlocal_force_shadow", "pair_geometry", "final_force_output", "population_setup", "population_output", "initial_density_setup", "initial_population_setup", "initial_population_output", "density_history_setup", "density_history_output", "final_spectrum_output", "nonlocal_h0_shadow", "nonlocal_h0", "radial_setup", "initial_mesh_setup", "initial_mesh_shadow"
};
static_assert(sizeof(names)/sizeof(names[0])==boundaries,"Transfer labels must match the accounting enum");
constexpr const char* phases[] = {"initialization", "scf", "post_scf", "analysis_export"};
long long unix_nanoseconds() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
struct Traffic { unsigned long long h2d=0, d2h=0, copies=0; };
struct Interval {
    int phase=0, geometry=0, iteration=0;
    double seconds=0.;
    long long unix_begin_ns=unix_nanoseconds(),unix_end_ns=0;
    std::array<Traffic,boundaries> traffic{};
};
struct Execution {
    uint64_t epoch=0;
    bool marked=false;
    Clock::time_point start=Clock::now();
    std::vector<Interval> intervals{Interval{}};
    void finish_interval() {
        auto now=Clock::now();
        intervals.back().seconds+=std::chrono::duration<double>(now-start).count();
        intervals.back().unix_end_ns=unix_nanoseconds();
        start=now;
    }
    ~Execution() {
        finish_interval();
        if(remag_activity_finish) remag_activity_finish();
        FILE* f=std::fopen("remag-residency.json","w");
        if(!f) return;
        std::fprintf(f,"{\"schema_version\":2,\"phase_markers_seen\":%s,\"geometry_epochs\":%llu,"
            "\"scope\":\"successful explicit bridge copies only\",\"library_transfers\":null,"
            "\"unified_memory_migrations\":null,\"file_io\":null,\"strict_residency_verified\":false,\"intervals\":[",
            marked?"true":"false",static_cast<unsigned long long>(epoch));
        for(size_t i=0;i<intervals.size();i++) {
            const auto& v=intervals[i];
            std::fprintf(f,"%s{\"phase\":\"%s\",\"geometry\":%d,\"iteration\":%d,\"wall_seconds\":%.9f,"
                "\"unix_begin_ns\":%lld,\"unix_end_ns\":%lld,\"boundaries\":{",
                i?",":"",phases[v.phase],v.geometry,v.iteration,v.seconds,v.unix_begin_ns,v.unix_end_ns);
            bool first=true;
            for(size_t j=0;j<boundaries;j++) {
                const auto& t=v.traffic[j];
                if(!t.copies) continue;
                std::fprintf(f,"%s\"%s\":{\"host_to_device_bytes\":%llu,\"device_to_host_bytes\":%llu,\"copy_calls\":%llu}",
                    first?"":",",names[j],t.h2d,t.d2h,t.copies);
                first=false;
            }
            std::fprintf(f,"}}");
        }
        std::fprintf(f,"]}\n");
        std::fclose(f);
    }
};
Execution& execution() { static Execution state; return state; }
}

void remag::transfer(Transfer boundary,bool to_device,size_t bytes,size_t copies) {
    auto& t=execution().intervals.back().traffic[static_cast<size_t>(boundary)];
    if(to_device) t.h2d+=bytes; else t.d2h+=bytes;
    t.copies+=copies;
}
uint64_t remag::geometry_epoch() { return execution().epoch; }
bool remag::in_scf() {return execution().marked&&execution().intervals.back().phase==1;}
extern "C" void remag_geometry_changed() { execution().epoch++; }
extern "C" void remag_trace_finalize() {
    if(remag_activity_finish) remag_activity_finish();
}
extern "C" int remag_defer_scf_log() {
    static const bool enabled=[]() {
        const char* value=std::getenv("REMAG_DEFER_SCF_LOG");
        return value&&std::strcmp(value,"1")==0;
    }();
    return enabled?1:0;
}
extern "C" int remag_phase_enter(int phase,int geometry,int iteration) {
    if(phase<0||phase>3||geometry<0||iteration<0||(phase==1&&iteration==0)) return -140;
    auto& s=execution();
    s.finish_interval();
    Interval interval;
    interval.phase=phase;interval.geometry=geometry;interval.iteration=iteration;
    interval.unix_begin_ns=s.intervals.back().unix_end_ns;
    s.intervals.push_back(interval);
    s.marked=true;
    if(remag_activity_phase) remag_activity_phase(phase,geometry,iteration);
    return 0;
}
