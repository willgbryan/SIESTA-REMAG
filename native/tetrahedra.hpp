// SPDX-License-Identifier: GPL-3.0-or-later
// Tetrahedron occupations on a Monkhorst-Pack grid. Each grid subcell is cut
// into six tetrahedra sharing its shortest main diagonal. Methods:
//   1 linear: band energies linear inside each tetrahedron;
//   2 Bloechl: linear plus the correction of Bloechl, Jepsen and Andersen,
//     PRB 49, 16223 (1994);
//   3 optimized: linear weights of corner energies fitted to 20 grid points
//     around each tetrahedron, Kawamura, Gohda and Tsuneyuki, PRB 89, 094515
//     (2014), as in libtetrabz and Quantum ESPRESSO's tetrahedra_opt.
// The k list may be the full grid or its time-reversal reduction; every grid
// point must map to exactly one listed k point, by k or by -k. As in QE's
// opt_tetra_weights_only, weights are then averaged over bands degenerate
// within 1e-6 Ry at each k, so the density does not depend on how the solver
// rotates a degenerate subspace. Energies are in Ry.
#pragma once
#include <vector>

namespace remag {
struct TetraGrid {
    int n[3]={0,0,0},corners=0,tetrahedra=0,diagonal=0;
    std::vector<int> point;   // grid point -> listed k index
    std::vector<int> stencil; // corners grid points per tetrahedron
};
enum TetraStatus {
    tetra_ok=0,tetra_bad_input=-401,tetra_off_grid=-402,tetra_grid_incomplete=-403,tetra_grid_conflict=-404,
    tetra_bad_weights=-405,tetra_too_few_states=-406,tetra_top_band_occupied=-407,tetra_no_fermi=-408
};
// cell: lattice vectors as columns (Bohr); kpoints: Cartesian (1/Bohr).
// diagonal -1 picks the shortest main diagonal; 0-3 force (-1,1,1), (1,-1,1),
// (1,1,-1) or (1,1,1) in grid steps. QE's 'tetrahedra' (Bloechl) always uses
// (1,-1,1) and does not average degenerate bands; the checks use both
// settings to reproduce it. SIESTA always uses the defaults.
int tetra_grid(int method,const int* grid,const double* displacement,const double* cell,int nk,
               const double* kpoints,const double* weights,int trs,TetraGrid& out,int diagonal=-1);
// energies, occupations: (maxe,maxspn,nk) as fermid.F. Occupations are the
// k weight times the state occupation (2 per spinless state, else 1). With
// given, *fermi is an input and no Fermi level is searched (checks only).
int tetra_occupations(const TetraGrid& grid,int method,int nspinor,int maxspn,int maxe,int ne,
                      const double* energies,double charge,double* occupations,double* fermi,bool average=true,bool given=false,
                      int threads=1);
}

extern "C" int remag_tetra_occupations(int method,const int* grid,const double* displacement,const double* cell,int nk,
    const double* kpoints,const double* weights,int trs,int nspinor,int maxspn,int maxe,int ne,const double* energies,
    double charge,double* occupations,double* fermi,int* tetrahedra,int threads);
