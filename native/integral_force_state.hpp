// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
namespace remag {
void finish_integral_grid_force(uint64_t epoch,int status);
}
extern "C" int remag_integral_force_enabled();
extern "C" int remag_integral_force_begin(int atoms,const double* force,const double* stress);
extern "C" int remag_integral_force_finish(int atoms,double* force,double* stress,double* energy);
extern "C" int remag_integral_hartree_stress();
