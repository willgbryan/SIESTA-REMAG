// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <stdexcept>

// One bounded contract for the adapter, graph planner and CPU-readable build
// metadata. Sampling is dynamic; 128 was an obsolete prototype restriction.
// This ceiling does not promise that every matrix/mesh combination fits a GPU.
#define REMAG_SCF_MAX_KPOINTS 4096
#define REMAG_SCF_MAX_ORBITALS 2048
#define REMAG_SCF_MAX_DIMENSION 4096
namespace remag {
inline constexpr int scf_max_kpoints = REMAG_SCF_MAX_KPOINTS;
inline constexpr int scf_max_orbitals = REMAG_SCF_MAX_ORBITALS;
inline constexpr int scf_max_dimension = REMAG_SCF_MAX_DIMENSION;
constexpr bool scf_shape_supported(int kind,int n,int nh,int nk) {
    return kind>=0&&kind<=3&&n>0&&n<=scf_max_orbitals&&nk>0&&nk<=scf_max_kpoints&&
        (kind%2||nk==1)&&(kind<2?(nh==1||nh==2):(nh==4||nh==8));
}
struct ScfSpectralLayout {
    int dimension,spins,jobs;
    std::size_t vector_bytes;
};
inline ScfSpectralLayout scf_spectral_layout(int kind,int n,int nh,int nk) {
    if(!scf_shape_supported(kind,n,nh,nk))throw std::runtime_error("Unsupported device SCF shape");
    const int dim=n*(kind>=2?2:1),spins=kind<2?nh:1;
    // Cast before multiplication: full retained banks can exceed INT_MAX even
    // though each dimension and the spectral indexing fit signed 32-bit ints.
    return {dim,spins,nk*spins,std::size_t(dim)*dim*(kind?2:1)*sizeof(double)};
}
}
