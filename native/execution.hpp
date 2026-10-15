// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>

namespace remag {
enum class Transfer {
    solver_matrices, solver_values, solver_vectors, solver_status,
    grid_topology, grid_density, grid_potential, grid_matrix, grid_output,
    dm_topology, dm_occupations, dm_energies, dm_outputs,
    normalization_overlap, normalization_status,
    occupation_setup, occupation_result, occupation_outputs, spectral_export,
    hs_geometry, hs_values, hartree_geometry, hartree_density, hartree_potential, hartree_result,
    grid_flow_geometry, grid_flow_result, sparse_state_setup, sparse_state_scalars,
    sparse_state_mixer_status, sparse_state_export, solver_graph_setup, final_density_output, orbital_basis_setup, orbital_basis_shadow, atomic_basis_setup, atomic_basis_shadow, two_center_setup, two_center_shadow, pair_force_shadow, nonlocal_geometry, nonlocal_force_shadow, pair_geometry, final_force_output, population_setup, population_output, initial_density_setup, initial_population_setup, initial_population_output, density_history_setup, density_history_output, final_spectrum_output, nonlocal_h0_shadow, nonlocal_h0, radial_setup, initial_mesh_setup, initial_mesh_shadow, spin_constraint_setup, spin_constraint_output, count
};
// Accounting of successful explicit copies only; library activity is unmeasured.
void transfer(Transfer boundary, bool to_device, size_t bytes, size_t copies=1);
uint64_t geometry_epoch();
bool in_scf();
}

extern "C" int remag_phase_enter(int phase, int geometry, int iteration);
extern "C" void remag_geometry_changed();
