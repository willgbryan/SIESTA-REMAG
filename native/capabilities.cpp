// SPDX-License-Identifier: GPL-3.0-or-later
#include "scf_capacity.hpp"
#define REMAG_STRINGIFY_INNER(x) #x
#define REMAG_STRINGIFY(x) REMAG_STRINGIFY_INNER(x)
// Pure metadata: safe to call in the CPU builder; no CUDA or solver call.
extern "C" const char* remag_device_scf_capabilities() {
    return "{\"format\":\"REMAG-DEVICE-SCF-CAPABILITIES/1\","
        "\"max_kpoints\":" REMAG_STRINGIFY(REMAG_SCF_MAX_KPOINTS) ","
        "\"max_spatial_orbitals\":" REMAG_STRINGIFY(REMAG_SCF_MAX_ORBITALS) ","
        "\"max_spinor_dimension\":" REMAG_STRINGIFY(REMAG_SCF_MAX_DIMENSION) "}";
}
