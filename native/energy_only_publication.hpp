// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace remag {
struct EnergyOnlyPublication {
    uint64_t epoch;
    int capacity,spins,iterations;
};

// Preserve the historical single-cycle format. A sweep carries every accepted
// epoch, so its last direction cannot overwrite evidence of earlier directions.
inline void publish_energy_only(const std::vector<EnergyOnlyPublication>& rows) {
    uint64_t previous=0;
    if(rows.empty())throw std::runtime_error("Missing energy-only publication");
    for(const auto& r:rows) {
        if(r.epoch<=previous||r.capacity<=0||r.iterations<=0||
            (r.spins!=1&&r.spins!=2&&r.spins!=4&&r.spins!=8))
            throw std::runtime_error("Invalid or repeated energy-only publication epoch");
        previous=r.epoch;
    }
    const char* temporary="remag-energy-only.json.tmp";
    FILE* f=std::fopen(temporary,"w");
    if(!f)throw std::runtime_error("Cannot record energy-only publication");
    const bool sweep=rows.size()>1;
    if(sweep)std::fputs("{\"schema\":2,\"mode\":\"energy_only\",\"cycles\":[",f);
    for(size_t i=0;i<rows.size();i++) {
        const auto& r=rows[i];
        std::fprintf(f,
            "%s{\"schema\":1,\"mode\":\"energy_only\",\"epoch\":%llu,\"status\":0,\"capacity\":%d,\"spins\":%d,"
            "\"iterations\":%d,\"forces_calculated\":false,\"stress_calculated\":false,"
            "\"hamiltonian_required\":false,\"hamiltonian_cycle_export_bytes\":0}",
            i?",":"",static_cast<unsigned long long>(r.epoch),r.capacity,r.spins,r.iterations);
    }
    std::fputs(sweep?"]}\n":"\n",f);
    const bool failed=std::ferror(f)!=0;
    const int closed=std::fclose(f);
    if(failed||closed!=0||std::rename(temporary,"remag-energy-only.json")!=0)
        throw std::runtime_error("Incomplete energy-only publication record");
}
}
