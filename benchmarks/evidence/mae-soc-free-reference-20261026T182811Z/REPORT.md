# SOC-free MAE diagnostic — 26 October 2026

**Stopped on an implementation failure.** Two fresh full-SOC FePt calculations
passed; the third, SOC-free reference converged to a nonmagnetic state and
failed the required final-moment direction gate. Six planned FeCo calculations
were unvisited. No valid SOC-free MAE or reference-direction comparison was
established. The original failed attempt remains intact.

Run `mae-soc-free-reference-20261026T182811Z`, completed attempt
`9bbd21df4c3341cd814e423ec13027e8`. A completed attempt marker records evidence
publication, not scientific success. Remote summary is `validation_failed`.
The seven comparisons executed before the failure passed; they do not erase
the failed third ledger row. Local qualification likewise retains two passes
and one failure.

## Cause and correction

The recorded nonlocal owner uses `split=0`. Pinned upstream
`Src/nlefsm.F:calc_Vj_offsiteSO` puts the entire lj nonlocal operator into
`V_so` in this mode, including scalar ionic terms. Zeroing that buffer removed
the scalar potential as well as SOC. Its half spin trace reaches 17.276615 Ry
in the selected saved matrices. FePt's atomic net charges changed from
approximately +0.365/−0.365 electrons to −4.318/+4.318; the spin collapsed.

The local correction decomposes each complex offsite spin block as
`V = ½ Tr_spin(V) I + V_spin`. SCF retains the scalar component and frozen
targets restore only the spin-traceless remainder. This keeps the original
projectors and the separately stored scalar operator, including scalar terms
that the original offsite buffer held. It is a fixed-projector spin decomposition,
not equivalence to a regenerated scalar-relativistic pseudopotential or the
optional Hemstreet projector split. The ordinary full-SOC path is unchanged.

107 focused tests and host C++ syntax checks pass. Host-compiled shared
decomposition arithmetic was replayed on six saved dense captures: scalar
extraction agrees exactly with the independent half trace; reconstruction error
is at most 1.39e-17 Ry, residual spin trace 1.78e-15 Ry, and the equivalent
ordinary full-SOC rotation differs by at most 7.11e-15 Ry. Corrected CUDA
compilation and SCF execution are **pending**. These local checks do not repair
or qualify the failed calculation, and no retry was launched.

## Retained numerical results

All energies in the following table are µeV/f.u. at **300 K electronic
smearing**, for FePt x minus z. The full-SCF anchors are both fresh. Frozen
values here use the ordinary SOC-containing reference, which passed its
historical regression; they are not results of the rejected SOC-free mode.

| Estimator | E | F |
| --- | ---: | ---: |
| Full SCF: total / free | −1357.583279 | −541.785301 |
| Ordinary frozen: band / band-free | −1357.944259 | −579.581617 |
| Frozen minus full SCF | −0.360980 | −37.796316 |

The ordinary approximation still misses the unchanged 5 µeV/f.u. free-energy
target. Full-SCF spin magnitudes are 3.378620 and 3.382100 µB/f.u. for z/x.
The rejected SOC-free SCF prints zero moment; its restored frozen targets have
only 2.07e-10 and 4.44e-10 µB, so their nearly zero directional differences
are not admissible magnetic anisotropy estimates.

| Calculation | SCF iterations | Process seconds | Result |
| --- | ---: | ---: | --- |
| FePt z, ordinary frozen probe | 57 | 14.207863 | Pass |
| FePt x, full-SOC anchor | 74 | 14.322032 | Pass |
| FePt z, attempted SOC-free probe | 27 | 13.703600 | Rejected: zero moment / wrong operator |

Executed scope: two-atom FePt, 48 spatial / 96 spinor basis functions, TZDP,
non-perturbative polarization, global SplitNorm 0.25, PBE, fixed full offsite
pseudopotentials/geometry, 0.01 Ry confinement, requested 400 Ry grid (actual
443.309 Ry), zero offsets, 4³ = 64 actual k points and 64 GPU eigensystem jobs
per full-mesh solve. FP64, upstream radial setup, eight SCF lanes, DM 1e-9,
H 1e-8 eV, 300-SCF cap; frozen evaluation uses one streamed dense lane.
The new SOC-free identity solve also recorded 64 jobs. The planned FeCo
16³ / 4096-k scope was not executed. Production admission remains 128; staged
32768 is only an admission ceiling.

Twelve selected matrix captures from the two executed probe calculations pass
84 algebra/spectrum checks: maximum matrix error 9.77e-15 Ry and spectral
error 1.34e-10 eV. Those checks validate the supplied operator's algebra;
they did not detect removal of the wrong physical component. The correction
adds explicit scalar-preservation metadata and a spin-trace check. Old captures
retain their original unknown operator-definition metadata.

## Budget and evidence

The single H100 app `ap-R8ElUpDqYH8382WYhRlEDe` is STOPPED with zero account
containers. Conservative app lifetime was **131.598391 seconds (2.193307
minutes)**, including image building, within the approved 1200 seconds. It
stopped at 18:29:48.172 UTC, before the fixed 18:47:32.233 UTC deadline.
Function work before packaging took 71.556346 seconds; actual billed GPU time
is unavailable. The allowance is closed; no follow-on allocation or extension.

The exact completed attempt was used to fetch 27 density/matrix output files.
Raw spectra, capture binaries, logs, density/matrix outputs and implementation
archive remain under this run in local `results/` and the Modal volume.
`raw-evidence-manifest.json` records hashes of the local evidence. The compact
tracked copy retains the ledger, failed validation, operator diagnosis, inputs,
allocation release and local reports. `postprocessing/` preserves the original
and corrected local analyzers: zero-moment angles now remain null rather than
causing a JSON-serialization error, and invalid probes are excluded from
method-bias estimates. Remote sources and original records are unchanged.

No CPU timing run, useful MAE speedup, upstream-GPU speedup, zero-temperature
MAE, physical convergence or cross-package agreement is claimed. Spin moments
do not establish total saturation magnetization. Basis/confinement, grid,
k/smearing, geometry, independent pseudopotential/semicore, angular/torque and
orbital qualification remain open.
