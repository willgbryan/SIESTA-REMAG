# GPU magnetic-response screening, 23 November 2026

The initial screening stack now shares one complex-FP64 magnetic state across
native CUDA MFT, CuPy Green response, PAO interpolation and supplied-parameter
local 4f multiplets. This report records usable results and current limits;
it does not qualify a material or claim a matched upstream-GPU speedup.

## FePt frozen angular response

The two-atom FePt cell uses TZDP, nonperturbative polarization, SplitNorm 0.25,
400 Ry mesh, a complete 4×4×4 k mesh and 300 K electronic temperature: 96 spinor
states at each of 64 k points. A converged SOC-free generator retains the
scalar trace of the native off-site lj operator. Its spin-traceless SOC operator
is restored in the fixed crystal frame for the frozen angular calculations.
The native solver's lower-triangle convention is preserved on import.

All GPU jobs ran sequentially on one Modal A100-SXM4-40GB through the serialized
launcher. Native CUDA 12.6.3, CuPy 14.2.0 and the pinned pristine SIESTA source
were used. The disposable integration source/build and Python code are preserved
in each raw run's `implementation.tar.gz`; no upstream source was edited.

| Calculation | Seconds | Status |
|---|---:|---|
| SOC-free reference SCF, including requested export/probe | 13.8194 | Completed, converged |
| Six-direction MFT, after the reference | 1.9286 | Numerical checks passed |
| Full-SOC directional SCF z | 13.0969 | Completed, converged |
| Full-SOC directional SCF x | 13.7015 | Completed, converged |

Each MFT orientation's native solve/reduction took approximately 0.206 seconds;
reported native workspace was 52,627,748 bytes. These are individual timings,
not repeated performance statistics. MFT and SCF calculate different
approximations, and no speedup ratio is asserted. Setup, cloud cost and
unmeasured transfer/stage quantities are not inferred.

The frozen result gives **F(x)−F(z) = −0.680212 meV/cell**; the fresh directional
SCF controls give **−0.541786 meV/cell**. The frozen bias is **−0.138426 meV/cell**.
The model therefore remains screening, including its easy-plane classification
at these coarse settings. It is not a physically converged prediction for FePt.
An explicitly selected axial fourth-order fit gives K1 = −0.750114 and
K2 = 0.070124 meV/cell; these coefficients assume the selected angular model.

Generator-spectrum reconstruction differs by at most 7.01e−13 eV, SOC-free
rotation by 2.76e−11 eV, and x/y free energies by 3.34e−10 eV. Maximum generalized
residual is 2.62e−15 and S-orthogonality error 4.20e−12. The spin direction is
within 0.0213 degrees of the requested direction. A saved-matrix torque check
agrees with a fixed-count corrected free-energy derivative within 2.11e−10
eV/rad; its uncorrected derivative was sensitive to occupation-count tolerance
and is retained in the raw diagnostic.

Reusing the existing pristine-SIESTA CPU results, total/free energies, moments,
charges, RHO and DRHO pass the existing checks. **The full regression fails**
the all-band eigenvalue tolerance: maximum 3.441e−4 eV versus 1e−4 eV at an
unoccupied band near 61.77 eV. Occupied bands agree within 1e−7 eV. The original
tolerance is unchanged; this comparison does not establish an upstream-GPU gain.

## Other GPU methods

Reusing the saved FePt state, a 2.74 Å interaction radius produced 12 unique
isotropic interactions in 52.24 seconds: Fe–Fe approximately 28.93 meV,
Fe–Pt 6.905 meV and Pt–Pt 1.200 meV, all ferromagnetic in the declared
onsite frozen-field model. Symmetry-related Fe–Pt values agree to approximately
1.2e−12 eV. These are coarse-mesh screening outputs, without a TB2J material
comparison or radius/k convergence. They do not establish a Curie temperature.

Full-basis PAO orthogonalization/Fourier reconstruction on the same 64-point
mesh preserved the z/x/y MFT free-energy differences within 2.83e−12 eV. Setup
took 3.02 seconds; setup plus three MFT directions took 5.33 seconds. The
dimension remains 96: this demonstrates consistent operator projection and
response reuse, not a compact-model performance gain. An explicitly supplied
illustrative Ce3+ model at 300 K (SOC 0.1 eV, Wybourne B20=0.01 eV, exchange
field 0.02 eV) gives a 0.193 meV local angular difference. These coefficients
were not extracted from a material and are not added to FePt's result.

Focused operator calculations passed on the same single-GPU workflow:

- Spectral versus selected-solve Green block error: 5.40e−15.
- Two-site exchange versus frozen grand-potential finite difference:
  3.91e−7 eV; sign and unique-pair convention are explicit.
- Full-basis analytic PAO interpolation spectrum error: zero at stored precision.
- One-electron 4f SOC splitting error: 1.11e−16 eV (six plus eight states).
- Two-electron Hund state: S(S+1)=2 and L(L+1)=30.
- A noncentrosymmetric two-site spin-dependent hopping model produces a nonzero
  antisymmetric exchange tensor. This is an operator example, not a material
  DMI benchmark or a TB2J validation.

The initial exchange method retains only onsite magnetic perturbations and has
O(nk²) spectral contractions. SOC tensors combine transverse responses from
three frozen global references. General noncollinear response and local
single-ion anisotropy extraction are unavailable. PAO selection has no energy
disentanglement; Wannier90 import supports spectra but H alone cannot supply
magnetic/SOC operators. No compact-model speedup is claimed. The 4f solver needs
explicit ionic occupancy, Coulomb, SOC, crystal-field and exchange parameters;
missing parameters remain incomplete. No SmCo5/Gd materials qualification or
automatic parameter extraction has been performed.

## Evidence and reproducibility

- [Summary and scope](summary.json), [individual timings](fept-ledger.json),
  [MFT surface and descriptors](fept-mft.json).
- [Native component check](native-component.json),
  [optional GPU calculations](optional-response-checks.json),
  [saved upstream comparison](saved-upstream-comparison.json).
- [FePt exchange](fept-exchange.json), [PAO response reuse](downfold-response.json),
  [illustrative local 4f environment](ce-local-environment.json),
  [combined FePt screening package](magnetic-screen.json),
  [material-response run](material-run-summary.json).
- [Run settings, source hashes and GPU identity](fept-run-metadata.json),
  [artifact digests](evidence-sha256.json),
  [retained unsuccessful attempts](retained-failed-attempts.json).
- Full state, raw generator, input/output, spectra and source snapshot:
  `results/magnetic-screen-20261123T171929Z/`; optional checks:
  `results/magnetic-response-components-20261123T172242Z/`.
  Material response reuse is in
  `results/magnetic-material-response-20261123T172802Z/`.
  All are retained under the same run IDs in Modal volume
  `siesta-remag-magnetic-results`.

The main run's original summary remains `failed` because optional-check report
writing encountered a NumPy-boolean JSON serialization error after the material
calculations completed. Its four calculation ledger entries are valid. A
separate optional-only run fixed report serialization and passed, without
repeating SCF. Earlier build, startup, full-volume, triangle-convention,
charge-derivative, incompatible-build and export-accounting attempts remain
preserved in local `build/magnetic-screen-*` logs and raw results. The old Modal
volume had exhausted its inode quota and was left intact.

The 87 focused local tests passed. Exact GPAC/evaluated-geometry identity was
not supplied for this historical input and remains null. The API accepts these
identities when supplied; this repository has no live TWINS/GPAC connector.
Orbital moments, site torque and causal orbital/SOC intervention explanations
remain unavailable rather than fabricated. Production FP64 defaults and the
separate paused MAE qualification campaign are unchanged.
