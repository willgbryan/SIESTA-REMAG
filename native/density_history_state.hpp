// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
namespace remag {
bool density_history_enabled();
bool density_history_reuses();
// The initial owner supplies its exclusive device allocation. Extrapolation
// completes before returning; a host-initialized allocation is not inspected.
void prepare_history_density(double* device_density);
void bind_history_density(uint64_t execution_epoch,const double* device_density);
}
extern "C" int remag_history_enabled();
extern "C" int remag_history_configure(int depth,int mode);
extern "C" int remag_history_count();
extern "C" int remag_history_select(int reuse);
extern "C" int remag_history_geometry(int atoms,int rows,int capacity,int spins,
    const int* num,const int* ptr,const int* col,const int* nsc,const double* xyz);
extern "C" int remag_history_coefficients(int count,double* values);
extern "C" int remag_history_finish(int save);
