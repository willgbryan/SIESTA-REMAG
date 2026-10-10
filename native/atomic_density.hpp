// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
namespace remag {
// Compact rhooda inputs. Mesh is the fine, sequential mesh. Orbital and unit
// indices retain upstream's one-based representation; ends[0] is zero.
struct AtomicDensityView {
    int mesh[3],nsm,coarse_points,subpoints,entries,orbitals,unit_orbitals;
    const int *ends,*entry_orbital,*unit_orbital;
    const double* occupation;
    const float* phi;
};
// Direct composition of stored-basis rhooda and reord(+1). Each output point
// sums its orbital entries in upstream order. No atomics in density sums.
// One kernel; all arrays are device-owned, with no transfers, allocation,
// synchronization or I/O. Sticky status: 1 invalid mapping, 2 nonfinite data.
cudaError_t initial_atomic_density_device(const AtomicDensityView&,double* sequential_density,
    int* status,cudaStream_t);
}
