# Magnetic-response screening

The November 2026 research interface extracts magnetic response from a frozen
electronic state. Its default status is **screening**. Existing full-SCF
outputs, FP64 production defaults and the pristine upstream source are unchanged.
The old MAE qualification campaign remains separate: its measured FePt/FeCo
reference bias is a limitation, not a reason to hide an otherwise usable result.

## Shared state and provenance

`REMAG-MAGNETIC-STATE/1` consists of `magnetic-state.json` and a hashed,
non-pickled `matrices.npz`. It holds complete sampled-k complex FP64 base,
magnetic and SOC operators, overlap, k coordinates/weights and, when available,
the original generator spectrum. Energies are Ry internally; result energies
are eV per evaluated cell. Spinors are interleaved spatial orbital/up/down.

Metadata records the electronic calculation, reference direction, occupations,
operator assumptions, Bloch gauge, geometry, sites/orbitals, settings and raw
evidence. Exact GPAC source and evaluated-geometry hashes are preserved when
supplied. They remain null when unavailable. REMAG never infers a GPAC identity
from a filename or substitutes selected-k captures for a complete k mesh.

The initial decomposition supports a globally collinear exchange field from
XC without Hubbard, constraint or external-field terms. It subtracts the native
spin-traceless off-site SOC operator; a scalar term in SIESTA's lj operator stays
in the base. Unsupported operator contributions fail explicitly. No independent
SCF subtraction is used to identify SOC.

For a fresh REMAG calculation, build with
`-DREMAG_BUILD_FROZEN_POTENTIAL=ON` and set `REMAG_EXPORT_MAGNETIC_STATE=1`.
The complete final **generator** H/S/native off-site operator and spectrum are
exported before the SCF owner releases them. Terminal HSX is not a replacement
for this generator. The export is optional scientific output; it changes no SCF
potential. Attach the actual calculation/geometry/site metadata with:

```sh
remag magnetic-import calculation/ --metadata identity.json --output state/
```

`identity.json` must declare `calculation_id`, `reference_direction`, and
`magnetic_terms` with `xc: rigid_spin_rotation`, and `hubbard`, `constraint`,
`external_field` all `absent`. Geometry, sites and orbital identity are required
for site-resolved response and cell-gauge interpolation. `sites` entries have
`site_id`, `species`, `position_angstrom`; `orbitals` has one entry with `site_id`
per spatial orbital. `geometry.cell_angstrom` stores lattice vectors as rows.
Include `gpac_source_hash`, `evaluated_geometry_hash`, `settings` and `evidence`
from the actual handoff. Missing identities must remain unavailable.

## Frozen angular MFT

Build the standalone CUDA library with the repository's native CMake project:

```sh
cmake -S native -B build/magnetic -DCMAKE_BUILD_TYPE=Release
cmake --build build/magnetic --target remag_magnetic --parallel 8
export REMAG_MAGNETIC_LIBRARY="$PWD/build/magnetic/libremag_magnetic.so"
remag mft state/ --directions directions.json --output screen/
```

`directions.json` contains Cartesian crystal-frame directions, for example
`[[0,0,1],[1,0,1],[1,0,0],[0,1,0]]`. The reference is included first. Actual
crystallographic symmetry must be supplied through the chosen directions;
automatic symmetry discovery is not implemented.

`remag_mft_sweep` uploads the reference once, caches each k-point Cholesky
factor, rotates the magnetic operator relative to fixed SOC, and solves full
spinor Hamiltonians using cuSOLVER. It reuses native REMAG Fermi–Dirac
occupations and compensated paired energy reductions. Generalized residuals,
S-orthogonality, electron count and finite results are checked in the numerical
path. By default, reference-spectrum reconstruction and one SOC-free rotation
are checked where the required evidence exists.

Results include band/free/canonical-free differences, direct electronic torque
from the spin commutator, spin expectations, near-Fermi angular band shifts,
k-resolved free-energy contributions and Mulliken orbital populations. Crossings
can change energy-ordered band identity. Orbital moments and site-resolved torque
remain unavailable. A two-direction difference is not reported as K1.
`--angular-model axial4` or `cubic6` explicitly selects the existing angular fit;
rank-deficient samples fail. The axial model assumes no azimuthal dependence.
Reported easy-axis/easy-plane classifications are model-dependent.

The initial native executor serializes k-point solves in one workspace; it caches
overlaps but does not yet use the production eight-lane solver. Requested raw
spectra and orbital weights are downloaded. Large full-k matrix banks can exceed
the explicit workspace limit; no large-system throughput claim follows.

Already executed probe results are usable without rerunning SCF:

```sh
remag mft-evidence calculation/remag-frozen-potential.json \
  --reference-direction 0 0 1 --output mft.json
```

The importer rejects failed/incomplete solves and preserves unknown quantities
as null. It does not invent torque or residuals absent from historical evidence.

### Dense meshes from the native finite-range generator

New generator exports also retain `remag-magnetic-realspace.{json,bin}`:
active native sparse H/S/SOC entries, orbital pairs and displacement vectors.
This is the actual final generator, not a Fourier reconstruction of the SCF
mesh or a terminal HSX Hamiltonian. Old sampled-only states cannot supply the
missing real-space information; capture it once, then reuse it for any mesh.

```sh
remag mft state/ --generator calculation/ --mesh 16 16 16 \
  --directions directions.json --angular-model axial4 --output dense-screen/
remag exchange state/ --generator calculation/ --mesh 8 8 8 \
  --radius 2.8 --output exchange.json
```

The Python equivalent is `load_realspace(calculation, state.metadata).sample(mesh)`.
Fourier reconstruction runs in GPU batches, retaining the native lower-triangle
semantics and a declared cell-periodic Bloch gauge. `uniform_mesh` uses the
source's Gamma-containing, zero-displacement grid. Denser sampling improves the
frozen-potential integration; it does not recompute its underlying SCF density.
Dense spectra/occupations and named orbital descriptors are retained by default.
Set `retain_orbital_weights=True` in `calculate_mft` only when the complete
k×band×orbital population array is needed; it can dwarf the numerical outputs.

## Optional GPU methods

Install `pip install '.[magnetic]'` on the CUDA host. The optional methods use
CuPy's cuBLAS/cuSOLVER and sparse GPU routines; none implements a CPU numerical
solver. The MFT/SCF path continues to use REMAG's native library.

```python
from siesta_remag.magnetic import (
    load_magnetic_state, calculate_mft, calculate_exchange,
    build_reduced_hamiltonian, calculate_4f_multiplets,
    summarize_magnetic_screen, compare_magnetic_responses,
)

state = load_magnetic_state("state/")
mae = calculate_mft(state, [[0, 0, 1], [1, 0, 1], [1, 0, 0]])
exchange = calculate_exchange(state, radius_angstrom=8.0)
screen = summarize_magnetic_screen(state, mae=mae, exchange=exchange)
```

`GreensFunction.block` evaluates selected blocks of `(zS-H)^-1` through
eigenpairs or complex linear solves. `calculate_exchange` now uses GPU Fourier
transforms of G at continued-fraction Fermi poles, followed by site-pair
contractions in real space. Its k dependence is linear at fixed pole/orbital
count, replacing the nested `(k,k')` loop. The finite-temperature pole expansion
follows [Ozaki, PRB 75, 035123](https://www.openmx-square.org/tech_notes/CF_Fermi.pdf).
The pole count increases only if it fails to represent occupations on the actual
spectrum. The original divided-difference implementation remains available as
`method='spectral_pairs'` for saved comparisons, not as the routine executor.

Local perturbations are an explicitly labeled onsite projection of the magnetic
operator. Nonlocal exchange-potential terms are omitted. Isotropic response is
the initial default; `relativistic=True` combines transverse Hessians from three
globally rotated frozen references. It is not a general noncollinear formalism.
The convention is `E = -sum_unique_pairs e_i.T @ J_ij @ e_j` for unit spins:
positive isotropic J favors parallel alignment. Pair tensors, their symmetric
traceless part and DMI are separate from local single-ion anisotropy. Finite-mesh
aliasing, omitted nonlocal terms and reference dependence remain limitations.
Site `magnetic_role` is explicit: `local_moment`, `induced`, `nonmagnetic` or
`unassigned`. Only pairs of declared local moments enter the independent-spin
network. Pt-like induced responses are retained as electronic response channels;
the code does not infer a freely orientable local spin from an induced moment.
The derivation follows the frozen-potential Green response discussed in
[He et al., TB2J](https://arxiv.org/abs/2009.01910); output pair counting uses
the unordered-pair convention rather than TB2J's double-counted pairs.

`build_reduced_hamiltonian(state, orbital_indices=[...], electrons=...)` selects
spatial PAOs, orthogonalizes the selected subspace on GPU, and projects all
magnetic/SOC operators in the same gauge. A truncated model requires an explicit
retained electron count. Fourier interpolation requires a complete uniform
`settings.kmesh`; its Hermitian trigonometric interpolation retains finite-mesh
aliasing. The original orbital mapping is preserved. Call `.spectrum(kpoints)`
or `.magnetic_state(kpoints, weights)` to evaluate the model. It never replaces
the full PAO calculation automatically or claims a saving without measurement.

`import_wannier90("seedname_hr.dat", metadata=...)` imports spinor Wannier90
Hamiltonians with degeneracy factors and checks real-space Hermiticity. It
supports GPU spectral interpolation. A Hamiltonian alone cannot supply a
magnetic/SOC decomposition; separately projected operators are needed before
using it for MFT. Existing SIESTA/Wannier90 tooling remains responsible for
projections, neighboring-k overlaps, localization and disentanglement.

## Localized 4f multiplets

`solve_4f_multiplet(parameters, directions=..., temperature_k=..., levels=...)`
works independently of DFT parameter extraction. `MultipletModel` caches the
fixed-occupancy Fock basis, atomic Coulomb/SOC and moment operators for repeated
environments. Custom CUDA constructs fermionic sparse matrix elements, then
GPU dense or sparse eigensolvers supply the spectrum. Eigenpair residuals and
Hermiticity are checked. The sign of the electronic magnetic moment is
`-muB*(L+2S)`; the supplied exchange energy couples as `-2 B_ex dot S`.

Required supplied parameters are `species`, `ionic_valence`, `occupancy` (0–14),
`slater_ev` (`0`, `2`, `4`, `6`), `soc_ev`, `crystal_field_convention`,
`crystal_field_ev`, `exchange_field_ev` and `double_counting`. Crystal fields use
explicit complex Wybourne `C^k_q` coefficients, **not interchangeable Stevens
coefficients**. A term has `k`, `q`, `real_ev`, optional `imag_ev`; conjugate terms
must make the matrix Hermitian. An explicit empty list specifies zero CF.
Directions and coefficients currently share the supplied crystal/local axes.
The angular Coulomb convention follows
[Quanty's spherical-harmonic formulation](https://www.quanty.org/physics_chemistry/coulomb_repulsion_jh_and_slater_integrals).

The partition function uses the available levels. If a truncated spectrum cannot
bound the omitted thermal weight adequately, free energy is unavailable until
more levels are requested. This is local multiplet thermodynamics, not the full
magnet's finite-temperature anisotropy. `double_counting` must be
`local_model_only` or `separate_open_core_contribution`; no term is automatically
added to a DFT result containing the same 4f electrons. General crystal-field
parameter extraction, automatic ionic-state assignment and relaxed substitution
predictions are not implemented. Missing parameters produce `incomplete` output.

## Model-facing results and comparisons

`REMAG-MAGNETIC-SCREEN/1` provides a compact `summary`, method results,
approximation labels, numerical sensitivity, missing observables and evidence
references. `compare_magnetic_responses(parent, child)` retains exact source and
geometry identities, site interventions, common-direction anisotropy changes and
moment changes, with explicit numerical comparability limitations. It does not
convert a correlation into a causal electronic explanation. TWINS now registers these operations in its real physics tool catalog. `magnetic.authoring` now supplies the narrow
GPAC-to-REMAG handoff and a TWINS Notebook adapter:

```python
from gpac import ir, parametric
from twins.memory import Notebook
from siesta_remag.magnetic.authoring import (
    export_authored_structure, prepare_authored_input, state_metadata, record_screen,
)

# GPAC owns exact source creation, validation, editing and orbit correspondence.
program = ir.load_program(source_bytes)
authored = export_authored_structure(program, "candidate/")
prepare_authored_input(authored, "existing-electronic-template/", "native-input/")
# After the existing run_case calculation, state_metadata binds the actual HSX
# geometry/site and PAO quantum numbers to the authored structure; H still comes
# from the converged generator export, not HSX.
notebook = Notebook("workspace/research.sqlite")
# hypothesis_id / experiment_id use Notebook's existing prospective APIs.
record_screen(notebook, experiment_id, screen, "screen.json")
notebook.close()
```

Exact GPAC program bytes, evaluated Cartesian geometry, site IDs and parent/edit
correspondence are retained separately. The numerical geometry frame is the
lower Cholesky factor of the exact Gram matrix in Angstrom. No GPAC compiler,
identity service or separate research-memory system is reimplemented. The Python workflow and direct chat tools use the installed GPAC facade and
TWINS Notebook. No separate notebook or model service is introduced.

Near-Fermi outputs now name Fe/Pt s/p/d PAO groups, signed projected DOS,
window centroids, angular energy-ranked shifts, and intersite band co-weights.
The latter is an absolute-Mulliken-weight normalized mixing diagnostic, not a
unique bond energy or proof of a mechanism. Paired comparisons report changes
in these quantities alongside common-direction anisotropy and exchange paths.

`contract_radial_manifolds(realspace, training_state, radial_rank=2)` constructs
a genuinely smaller, fixed local radial PAO map from occupied and nearby SOC
spinor states. It averages within complete angular shells and projects every
real-space H/S/magnetic/SOC operator with the same map, so subsequent dense-k
evaluation uses fewer states. The returned model/report does not automatically
replace the original: compare the intended angular and exchange response and
retain full PAOs if the contraction changes them materially. This is local
variational compression, not exact elimination or Wannier disentanglement.

In the FePt example, 96 → 72 spinor contraction reduced the 8³ three-direction
MFT time from 5.10 to 3.87 seconds with 0.051 meV maximum anisotropy error.
Its nearest Fe–Fe exchange changed by 70.6%, so this compact model is
provisional for anisotropy only; use full PAOs for exchange.

## Measured screening evidence

The [discovery follow-up report](../benchmarks/evidence/discovery-response-20261123/REPORT.md)
records native 8³/12³/16³ frozen interpolation, stable positive FePt anisotropy
at 12³/16³, the faster real-space exchange engine, and the completed GPAC/TWINS
parent–child feedback loop. A fixed-volume c+2% edit increased the 16³
anisotropy from 2.737 to 3.323 meV/cell and Fe-d/Pt-d co-weight from 0.402 to
0.432, contradicting the prospectively recorded prediction. The next exact
GPAC structure separates c stretch from in-plane compression; the [completed follow-up](../benchmarks/evidence/followup-response-20261123/REPORT.md)
measured 2.919 meV/cell and 0.401 co-weight, within the prospective intervals. Frozen-density and exchange mesh-sensitivity limits remain.

The [23 November A100 report](../benchmarks/evidence/magnetic-response-20261123/REPORT.md)
records a complete FePt six-direction MFT calculation in 1.93 seconds after a
13.82-second reference SCF. Its x–z band free-energy difference is −0.680 meV/cell
versus −0.542 meV/cell from directional SCF: the 0.138 meV bias remains explicit.
This is a provisional approximation comparison, not a matched upstream-GPU
speedup. The report retains the all-band upstream regression failure, passing
occupied-spectrum/density/moment checks, raw artifacts and optional GPU operator
results. These coarse settings are not physical convergence.

## Direct authoring-model workflow

TWINS exposes `prepare_magnetic_screen`, `inspect_magnetic_screen`,
`compare_magnetic_screens` and `record_magnetic_outcome` through `tool-catalog`,
`tool-call` and the `tools` JSON-lines transport. An explicitly selected operator
runtime additionally enables `run_magnetic_screen`. The browser IDE is unchanged.
Prepare accepts either GPAC's verified `GPAC-PHYSICAL/1` export or the existing
REMAG authored handoff; it re-executes exact source, retains the site mapping and
rejects changed geometry. Source withholding and workspace scope also apply.

A prepare call uses `structure`, a fresh `output` directory, optional `meshes`
(default `[[8,8,8]]`), Cartesian `directions`, `angular_model` and an optional
`exchange_mesh` equal to one requested mesh. Mesh dimensions are bounded at 16;
one request permits up to four meshes and twelve directions. The run call takes
the saved `request` and a fresh `output`. It performs one converged SOC-free
reference SCF, retains the native finite-range generator, then evaluates all
requested frozen responses. Inspection/comparison only read retained evidence.

Operator runtime schema (`REMAG-MAGNETIC-RUNTIME/1`):

- `repository`, `python`: absolute paths to this checkout and a Modal-enabled Python.
- `image`: pinned, prebuilt Modal image containing the native executable, magnetic
  library and optional GPU dependencies; the tool does not rebuild binaries.
- `volume`, `template_volume`, `template_relative_path`: existing result/template
  volumes and same-composition Cartesian-Angstrom input template.
- `executable`, `settings`, `electronic_settings`: native binary, string-valued
  environment and scientific settings retained with results.
- `timeout_seconds`, `scf_timeout_seconds`: positive allocation/SCF limits up to
  900 seconds, with the SCF limit no larger than the allocation limit.

This is a trusted operator file, selected at host startup with
`--magnetic-runtime`; model request JSON cannot choose commands or resources.
`scripts/run_modal.py` retains its account-wide allocation and upstream-pin
checks. A failed run saves native evidence; collection failures retain the
owned app ID and must be reconciled before another launch. There is no automatic
scientific retry. `magnetic-run.json` binds converged reference identity, exact
screen paths/hashes and scientific settings. The result volume also retains
raw generator matrices and spectra; local tool responses collect compact
reports/logs. Free-energy, moment and named mixing outcomes are written through
TWINS' existing source-bound Notebook APIs.

The [focused TB2J comparison](../benchmarks/evidence/exchange-comparison-20261123/REPORT.md)
measured 29.16 seconds including first-call GPU interpolation/exchange versus
150.96 seconds for matched full-basis TB2J on the saved 9³ FePt generator, with
4.50e−14 eV maximum interaction difference. This 5.18× result is one workload;
the report retains CPU parallelism, default-window accuracy, packaging and
sampling limits. No OpenMX `jx` result or general cross-code scaling is claimed.
