// SPDX-License-Identifier: GPL-3.0-or-later
#include "scf_export_guard.hpp"
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const bool accepted = std::strcmp(argv[1], "accepted") == 0;
    double residuals[][3] = {{.5,.25,.125},{.75,.5,.25}};
    try {
        remag::require_converged_final_grid(accepted, !accepted, 7, 2, [&](int i) {
            if (accepted || i < 0 || i >= 2) std::exit(3);
            return residuals[i];
        });
    } catch (const std::runtime_error& e) {
        if (std::strcmp(e.what(), "Final electronic state requires convergence") != 0) return 4;
        std::abort();
    }
    return accepted ? 0 : 5;
}
