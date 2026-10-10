// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
namespace remag {
// Rotate only the initial/reused density. H0, including SOC, stays in the
// crystal frame. The pending quaternion is consumed exactly once per direction.
int orientation_density(int capacity,int spins,double* density,cudaStream_t);
}
extern "C" int remag_orientation_set(int direction,const double* quaternion);

extern "C" int remag_orientation_retain_geometry();
