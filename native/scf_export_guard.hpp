// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdio>
#include <stdexcept>

namespace remag {
// The executor already validated and downloaded the completed iteration history.
// Keep the residuals in stderr because an upstream abort skips C++ destructors.
template<class Residuals>
void require_converged_final_grid(bool converged, bool exhausted, int geometry,
                                  int iterations, Residuals residuals) {
    if (converged) return;
    std::fprintf(stderr, "REMAG rejected final grid: geometry=%d iterations=%d exhausted=%d\n",
                 geometry, iterations, int(exhausted));
    for (int i = 0; i < iterations; ++i) {
        const double* r = residuals(i);
        std::fprintf(stderr, "REMAG rejected SCF step %d: dDmax=%.17g dHmax_Ry=%.17g dEmax_Ry=%.17g\n",
                     i+1, r[0], r[1], r[2]);
    }
    std::fflush(stderr);
    throw std::runtime_error("Final electronic state requires convergence");
}
}
