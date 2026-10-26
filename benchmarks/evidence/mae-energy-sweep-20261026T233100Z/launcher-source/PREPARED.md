# Energy-only full-SCF orientation reuse validation

Prepared after commits ab85cd7a, ba86b8d9, 1f23991e and 80acbab8, with the
subsequent accepted-electronic-snapshot reset correction. The user approved a new 1200-second allowance. The preceding diagnostic
allowance is closed. The fixed deadline is recorded in run-status.json.

One exact H100, requested ceiling 1200 seconds including image preparation and a
120-second evidence reserve. The deadline must be fixed when explicitly
approved, before importing the Modal app; it cannot reset on retry. Serial
launch through `scripts/run_modal.py` only. The app refuses existing run IDs and
does not rerun after an interruption or numerical/execution failure. Confirm
the prior app is stopped and account containers are empty before launch.

Fourteen serial processes, twenty-one independent SCF directions:

- FePt z/x/z return-to-start correctness check, excluded from timing ratios.
- Three matched FePt repetitions: two cold processes (z and x) versus one
  cached z/x sweep. Alternate contender order; retain each repetition.
- One FeCo matched set: cold [001], [110], [111] versus one cached three-axis
  sweep. One sample cannot establish repeatable performance.

Both modes use independent atomic guesses, FP64, eight solver lanes, full
offsite SOC, upstream radial setup, DM 1e-9/H 1e-8 eV and a 300-iteration cap.
Physical inputs remain the saved FePt 4³/400 Ry/TZDP/0.01 Ry/300 K and FeCo
16³/2000 Ry/TZDP/0.00125 Ry/300 K settings. No new CPU timing runs. Reuse the
saved pristine FePt oracle and REMAG FeCo controls, with unchanged density,
population, spectral, DM/HSX, signed final-direction and direct directional
E/F gates (5 µeV/f.u.). These inputs are not physical convergence studies.

The native patch applies to the exact prior recorded image sources. Only the
disposable Fortran source admits energy-only cached orientation loops; the
production guard stays closed. The staged 32768-k admission ceiling is carried
forward; production stays at 128 and no general capacity is claimed. The
experimental frozen code remains compiled but all frozen runtime options are
absent, and no frozen-potential calculation is requested.

Each process preserves its execution record, stdout, requested outputs and
failed/partial state before comparison. A sweep keeps the original labelled
files plus ordinary per-direction comparison views. No time is invented for
individual directions. Both contenders use the same 100-ms CPU and 200-ms GPU
observers, and whole-process time includes requested electronic exports.
Comparisons are incremental REMAG measurements, not upstream-GPU speedup.

Local checks: 54 focused tests pass; the native overlay applies with zero fuzz
to the previous implementation archive; the staged Fortran edits apply once
to disposable pinned integration source and preserve valid line lengths; all
five saved cold cases and both input/rotation families pass the staged gates.
Replaying the old force-producing z/x/z sweep retains all three directions and
correctly rejects it as energy-only evidence. CUDA compilation/execution is
still pending. Prior app ap-xrKTXHTfEM0cctD4nZyvoX is authoritatively STOPPED
with zero account containers (see prior-allocation-release.json).

After terminal completion, inspect ledger.json/current-comparisons.json and
summary.json, confirm allocation release, then fetch with the exact completed
attempt ID:

```
.venv/bin/python scripts/fetch_evidence.py <run> --outputs-only --density --matrices --workers 1 --attempt-id <id>
.venv/bin/python build/mae-energy-sweep-20261026/analyze_local.py <run>
```

Preserve failures and incomplete timing pairs. Do not promote the input guard
or claim a useful runtime improvement unless the scientific checks pass.
