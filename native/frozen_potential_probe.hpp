// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "frozen_potential.hpp"
#include "occupations.hpp"

namespace remag {
// Development-only connection to a real converged SCF state. Disabled unless
// the experimental build AND REMAG_FROZEN_ROTATIONS are selected. Its explicit
// spectrum downloads and host analysis are qualification boundaries, not a
// production throughput measurement. Outputs never replace SCF observables.
void frozen_potential_probe(const HamiltonianAssemblyView&,const double* generator,
    const cuDoubleComplex* offsite,const FermiProblem*,const double* weights,
    const double* generator_spectrum,cudaStream_t,bool reference_has_soc=true);
}
