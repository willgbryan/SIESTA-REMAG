# Final population consumers

The authoritative integral-force path keeps D/E on the GPU through kinetic,
nonlocal/SOC, grid and overlap forces. Compact terminal exports are qualified in
run `20261007T011339Z`; they eliminate 37,308,608 bytes of H(Dout) and Dold
copies across the 17-geometry corpus (no Eold export was required there). D and E remain compatibility exports.

The current regression inputs already print total spin from the device SCF
diagnostics. Their optional `state_analysis` geometry-spin branch is disabled;
replacing that inactive branch would not remove a current density consumer.
Actual remaining uses include final Mulliken population analysis in
`siesta_analysis`, requested DM files and `DM_history` in `post_scf_work`.
The latter supplies subsequent relaxation guesses through `new_DM`.
E is used by optional TSDE and NetCDF state exports; those settings must retain
valid output when terminal downloads become conditional.

## Population arithmetic reference

`scripts/check_population.py` extracts the original collinear and spinor
population loops from pinned `mulliken.F` and links unmodified `dm_charge` from
the pristine upstream library. Its serial orbital mapping is identity, matching
the supported one-process material route. It does not emulate MPI distribution.

The 39 retained fixtures in `results/preflight-population-v80/check.json` cover
spin components 1/2/4/8, empty atoms and sparse rows, holes in capacity, nonfinite
unused entries, and 131,073 rows. An independent NumPy sparse accumulation
agrees to 4.150e-12. SOC components 3/7 and 4/8 are averaged; components 5/6
do not enter these populations. Every matrix entry in a selected CSR span
contributes, including duplicate orbital columns.

## GPU candidate and acceptance

`native/population.{hpp,cu}` computes orbital, atomic and total spin-charge
populations from device D/S and topology. It initializes outputs, contracts
rows, reduces atoms and reduces total charge in four kernels. It performs no
allocation, copy, synchronization or I/O. Nonzero status invalidates the whole
result; owner integration must prevent publication/reuse of a failed result.
The standalone probe isolates this arithmetic. Run `20261007T015902Z` qualifies its CUDA arithmetic
and captured execution. Run `20261007T024355Z` subsequently qualifies its
retained-state integration across the nine-material corpus.

That run's optional `--probe-population` Modal check passed 184 executions:
direct and captured operation, one/three replays with poisoned output storage,
shape/span/atom errors, consumed nonfinite values, arithmetic overflow, ignored
SOC components and a last-row/partial-block tail fault. The 131,073-row fixture
does not cross the kernel's 1,024-block × 256-thread stride; a larger fixture is
required to validate a second stride iteration. Fixture transfers are outside capture.
The check is a separate library and does not alter production populations.
CUDA compilation, execution and independent retained-array reanalysis pass.
The independent NumPy reconstruction's maximum absolute discrepancy is
2.221e-15 (rounded upward); original pristine-Fortran comparison gates also pass.

The integrated owner stages atom boundaries before SCF and binds the
population consumer to the retained final density epoch.
The electronic-cycle plan releases its private buffers at terminal publication,
so borrowing its former D/S/topology pointers is invalid. The population owner
retains its own device copies of S/topology, and reads D from the final-state
owner. Analysis is permitted only after all force consumers publish
successfully; stale epochs and incomplete or failed force states are rejected.
Empty atoms and SOC conventions are preserved.

Integrate computed orbital/atomic values into the existing requested output
format, including spin-vector conversion and species/total reductions. Validate
against raw upstream output and every original material/MAE gate. Requested
overlap populations (options 2/3), orbital moments and additional analysis modes
need explicit routes; the option-1 primitive does not qualify them. Migrate
relaxation density history and make remaining matrix exports conditional on
actual consumers before claiming that D/E download removal is complete.

## Integration boundary identified from current consumers

`siesta_analysis.F90` calls `dm_charge` on the final D before optional spin and
Mulliken output. Replace both arithmetic consumers together; changing the
disabled geometry-spin branch alone would not remove this read. Preserve
upstream output ordering and the `spnvec` convention. The primitive currently
produces spin-charge components; vector magnitudes and species reductions are
additional observable computations that still need a device route.

The qualified owner stages atom boundaries before SCF and retains topology and
overlap from the cycle with device-to-device copies before the private cycle
storage is freed. Population contraction can run after the final force kernels,
before the existing shared status publication. Merge its status into that
publication so a failed population cannot become readable and no separate
status download is needed. Mark the population epoch readable only after all
force owners accept publication; stale, failed and incomplete epochs must fail.
This owner connection and the Fortran option-1 consumers are qualified and
disabled by default. Modal `--population` enables the retained consumer and
its owner checks. In run `20261007T024355Z`, all 48 shared-status/owner scenarios,
198 primitive executions and nine material calculations pass independent
audits, including the new 262,145-row second-stride fixture. Both directional
MAE comparisons pass. Across 17 geometries, setup is 740 bytes and requested
population exports are 20,400 bytes in 43 copies; repeated reads use cached
exports. All 358 SCF iterations retain zero explicit host/device copies.
The integration does not
remove D/E compatibility exports or migrate spin-vector/species reductions.
See the [population integration qualification](../results/20261007T024355Z/POPULATION.md).

The current corpus requests `Write.DM.End.Of.Cycle true` and `SaveHS true`.
Its requested matrix exports must therefore remain valid. Across run
`20261007T011339Z`, terminal H, D and E exports are respectively 18,654,304,
18,654,304 and 16,890,848 bytes. The immutable
`remaining-boundaries.json` records all measured explicit boundaries. Removing
unrequested matrix copies requires an additional input policy with DM/HSX
disabled, a fresh matching upstream reference, and migration of actual
relaxation-history consumers. The history remaps sparse patterns and may
extrapolate multiple coordinate/density pairs; replacing it with a different
initial guess would change the baseline.

## Performance follow-up

Run `20261007T004619Z` spends 37.120 of 42.989 seconds inside Fe16's complete
device electronic cycle (43 iterations). Fe54 spends 12.508 of 18.897 seconds
there (51 iterations). These are single observations using reused CPU
references, not qualified speedups or an attribution to a particular kernel.
Fresh matched repetitions and stage/kernel profiling remain necessary.

The subsequent compact-export run records 39.759 of 45.230 seconds in Fe16's
device cycle (46 iterations). Run `20261007T015902Z` includes fresh
same-allocation references, three candidate repetitions and a cache-disabled
control. Caching retains the first eigensolver's vectors for sampled k-points
instead of solving H/S again after occupation calculation. Its configured
vector-bank size for Fe16 is 10,616,832 bytes; Gamma already retains every block.
Local mode/shape checks pass, including 144 mutated records rejected. All 36
GPU calculations, 53 raw scientific comparisons, requested HSX outputs and cache
configuration/control audits pass. Fe16's median is 24.202 s versus 41.752 s
without the cache and 127.227 s for the serial CPU reference, all at 43 SCF
iterations. These are three candidate repetitions versus one CPU and one control
measurement on the same allocation. See the [fresh performance report](../results/20261007T015902Z/PERFORMANCE.md).

## Remaining energy-density export consumer

The supported force route consumes retained GPU E. Upstream `state_init.F`
creates fresh EDM storage and zeros it for each non-Transiesta geometry, so
relaxation history retains D rather than E. Optional E output consumers remain:
`save_density_matrix.F90` writes it when `TS_DE_save` requests TSDE, and
`m_ncdf_siesta.F90:cdf_save_state` writes it for the `EDM` dictionary key.
Both requested-output routes must stay correct before E downloads become
conditional. This is a source audit, not a validated export-policy change.

Derived population work must preserve signed collinear spin, the upstream
off-diagonal spin convention, cumulative totals across species in `mulliken.F`,
and the separate atom-order totals in `partial_charges.F90`. A small nine-fixture
CPU reference using pristine `spnvec` agrees with independent arithmetic to
2.221e-16 (rounded upward). Two corresponding CUDA kernels are drafted outside
the production tree; they are not compiled, integrated or GPU-qualified.

### Derived observables qualified (v82)

The production owner now prepares immutable species IDs and valences before
SCF and derives the upstream spin/charge views on the force stream, before
shared status publication. Schema 2 records seven export kinds: raw orbital,
raw atomic, raw total, orbital view, atomic view, cumulative species view and
summary. Views contain respectively 5, 6 and 5 doubles per item; the summary
contains 10 doubles. Initial staging is `36 + 8*atoms + 8*species` bytes in five
copies. Output downloads occur on the first requested read of each kind.

The final-analysis spin printer and Mulliken partial-charge writer select
these values without recomputing electronic reductions on the host. The
non-collinear/SOC table no longer exports raw orbital/atomic components.
Initialization's spin printer and non-Mulliken partition methods keep their
existing paths. D/E matrix downloads are still present.

Local tests and C++/Fortran compilation pass; Modal CUDA compilation also
passes. Run `20261007T033144Z` passes all nine materials, both MAE comparisons,
34 derived-output checks and 56 owner scenarios. All 225 source and 1,749
component-evidence hashes verify. Across 391 SCF iterations, explicit copies
remain zero. Requested population output is 28,128 bytes after SCF; D/E
compatibility exports and complete memory/I/O observation remain unfinished.
The allocation is stopped. Performance remains the fresh matched baseline in
`results/20261007T015902Z/PERFORMANCE.md`; this run reused CPU references.
See the [qualified run](../results/20261007T033144Z/REPORT.md).

The later [v84 qualification](../results/20261007T044323Z/qualification-summary.json)
revalidates the population route across 54 GPU calculations, including all
requested TSDE/NetCDF variants. It also qualifies omission of E download when
those files are not requested. H/D/spectral compatibility exports, initialization
and history remain. [Fresh matched repeated timings](../results/20261007T044323Z/PERFORMANCE.md)
now supersede v80 as the current performance baseline; the batch does not isolate
an incremental population or E-export timing effect.
