// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
namespace remag {
// Admission profile supported by retained setup evidence. This is a deployment
// restriction, not a claim to repair the driver-620 instantiation fault.
inline bool supported_scf_runtime(int runtime,int driver) noexcept {
    return runtime==12060 && driver==13030;
}
inline const char* scf_runtime_policy() noexcept {
    return "{\"format\":\"REMAG-SCF-RUNTIME-POLICY/1\","
        "\"cuda_runtime\":12060,\"cuda_driver_api\":13030,"
        "\"nvidia_driver_versions\":[\"610.57.04\"]}";
}
}
