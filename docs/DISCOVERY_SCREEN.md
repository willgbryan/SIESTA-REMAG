# Fast discovery screening

`remag-discover` turns a periodic candidate into a relaxed, magnetic directional-energy
screen using three finite stages:

1. **Orb:** unconstrained positions and full cell, conservative Orb v3 MPA model,
   FIRE, one deterministic 0.01 Å positional perturbation to release exact symmetry.
   This is the cheap stage that does most of the structural work.
2. **REMAG DFT:** variable-cell Broyden relaxation with fully relativistic PBE PSML,
   collinear magnetism and practical SCF/geometry tolerances.
3. **SOC:** separate self-consistent z/x/y runs at the accepted DFT geometry, each
   starting from its own rotated copy of the same accepted scalar density.

The discovery loop receives `report.json`, with a terminal status, directional
energy signals, magnetization, checks, hashes and separate timings. Failed or
unfinished relaxations never feed SOC. No convergence sweeps, angular fits,
exchange/Curie-temperature calculations, reference calculations or automatic
retries are part of this tool.

## Defaults and bounds

| Setting | Default |
| --- | --- |
| Orb force / stress targets | 0.02 eV/Å maximum force-vector norm / 0.1 GPa maximum stress component |
| Orb bounds | 400 steps, 300 seconds including worker startup |
| DFT force / stress targets | 0.04 eV/Å maximum force-vector norm / 0.5 GPa maximum stress component |
| DFT relaxation bounds | 30 ionic steps, 1,800 seconds |
| SCF tolerances, scalar and SOC | DM = 1e-4; H = 1e-3 eV; both must pass |
| SCF bounds | 100 iterations per ionic step or static calculation |
| Mixing | Pulay Hamiltonian, weight 0.03, history 8, all spin components |
| Basis / grid / electronic temperature | DZP / 300 Ry / 300 K |
| k sampling | Integer mesh from reciprocal-vector spacing ≤0.35 Å⁻¹ (includes 2π); ≤512 full-mesh points |
| SOC bounds | 600 seconds per direction |
| Whole DFT + SOC bound | 3,600 seconds, shared across stages |
| Input size | ≤200 atoms |
| Skip-SOC criterion | Relaxed scalar polarization Js <0.4 T |

These are upper bounds, not promised runtimes. Failed convergence stops the queue
without an extension. Native Broyden tests force components: the input uses
0.04/√3 eV/Å to enforce the documented 0.04 eV/Å vector gate. Stress is explicitly
set to 0.5 GPa. The DM/H thresholds are SIESTA defaults, not micro-eV MAE thresholds.
See the [SIESTA manual](https://docs.siesta-project.org/projects/siesta/en/5.4/reference/siesta.html)
and [Orb model documentation](https://github.com/orbital-materials/orb-models).

Use `remag-discover defaults` to print every setting. A `--settings policy.json`
file can override named settings when preparing a new bundle. Unknown keys and
nonpositive/nonfinite values are rejected. A larger structure or mesh is held
before DFT instead of being silently coarsened. Input constraints are rejected.

## Installation and local use

Use Python 3.12+ on Linux/macOS. The command needs a built REMAG executable;
installing the Python package does not build SIESTA or CUDA code.

```sh
python -m pip install -e '.[discovery,modal]'
remag-discover prepare candidate.cif \
  --pseudos /path/to/fully-relativistic-PBE-PSML \
  --orb-weights /path/to/orb-v3-conservative-inf-mpa-20250404.ckpt \
  --bundle runs/candidate-bundle
remag-discover dft --bundle runs/candidate-bundle \
  --runtime runtime.json --output runs/candidate-screen
```

Orb runs on the caller's CPU by default, with four threads and compilation
disabled. `--orb-device cuda` is available on a machine where Orb is installed
and GPU use is authorized. It finishes before DFT starts. Weights must already
be cached locally: no implicit model download or model substitution occurs.
Use a clean virtual environment if a mixed conda/pip installation loads conflicting
OpenMP libraries; do not bypass OpenMP's duplicate-runtime safety check.

The pseudo directory must contain `Element.psml` for each element, with Dirac
relativity and PBE exchange/correlation. The tool preserves explicitly supplied
collinear initial moments (e.g. in extxyz); otherwise it uses a simple magnetic
seed for Fe/Co/Ni/Mn/Cr and zero for other elements. This does not search competing
magnetic states or certify the ground state.

`runtime.json` identifies the executable and the reviewed REMAG environment:

```json
{
  "executable": "/opt/remag-candidate/Src/siesta",
  "executable_sha256": "REPLACE_WITH_ACTUAL_SHA256",
  "threads": 1,
  "environment": {
    "LD_LIBRARY_PATH": "/opt/hopper-build:/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
    "REMAG_DEVICE_SCF": "0"
  }
}
```

The inline example is a schema, **not a complete GPU runtime configuration**.
[The A100 host-controlled runtime example](../examples/discovery/a100-host-runtime.json)
contains the complete environment and binary hash used by the integration test.
Use it only with that matching validated binary/image. Otherwise reuse the
complete validated environment for your binary/image from the REMAG build's
experiment records; see [RUNNING.md](RUNNING.md). The executable checksum is
required and checked before execution. The report records supplied and inherited
REMAG controls. Remote runs clear inherited REMAG controls first. No new binary
build or pristine validation is launched by discovery.

To run both stages on one appropriately configured machine, replace `prepare`
with `run` and add `--runtime` and `--output`. Each stage runs serially.

## One A100 on Modal

Prepare Orb locally first. This avoids installing/loading Torch on the paid DFT
allocation and permits a lightweight native image. The native image must already
contain the validated REMAG binary, NumPy, SciPy and psutil. The helper adds one
cached CPU-only ASE dependency layer, then copies source, bundle and runtime
configuration. It never compiles a new solver or builds on a GPU.

From the checkout root, with the upstream source clean and pinned as required by
the repository launch guard:

```sh
export REMAG_DISCOVERY_BUNDLE=/absolute/path/to/runs/candidate-bundle
export REMAG_DISCOVERY_RUNTIME=/absolute/path/to/runtime.json
export REMAG_DISCOVERY_IMAGE=im-YOUR_VALIDATED_CACHED_NATIVE_IMAGE
python scripts/run_modal.py --module-file scripts/discovery_modal.py \
  --receipt runs/candidate-submission.json --output runs/candidate-collected
```

Always use `scripts/run_modal.py`. It checks account-wide allocations and refuses
GPU overlap. The helper uses one A100-80GB, one container, no warm pool and zero
retries. It adds 120 seconds to the DFT deadline for initialization/packaging; the
per-calculation and shared solver bounds still apply. CPU image-building and
allocation overhead are distinct from solver time and should be charged to the
caller's compute ledger.

A submission receipt is written before submission and updated with the function
call ID. If the client disconnects, retrieve that **existing call**:

```sh
remag-discover collect --receipt runs/candidate-submission.json \
  --output runs/candidate-collected --timeout 30
```

Collection launches no calculation. Do not submit again to recover output. A
receipt stuck at `submitting` has an ambiguous submission and requires inspection
of the recorded app before any new launch. A remotely interrupted run remains
on the `siesta-remag-results` volume under the receipt's run ID; never treat it as
a completed report. Use Modal's app/container inventory to verify allocation
shutdown before releasing the next candidate.

## Discovery-loop contract

The Python API is `prepare_bundle(...)` then `run_dft(...)` in
`siesta_remag.discovery.pipeline`; the CLI prints JSON. Exit status 0 means a
completed screening/preparation outcome; status 2 means a held scientific stage.
Invalid inputs raise an error and return nonzero. `status` reads a file and starts
nothing. All output paths are owned by one invocation, protected by a local lock.

| Terminal status | Interpretation / next discovery action |
| --- | --- |
| `prepared` | Orb targets passed; portable bundle ready for DFT |
| `screened` | Relaxation and all three SOC directions passed screening checks; rank the signal |
| `screened_low_moment` | Scalar relaxation passed; SOC skipped below the configured Js threshold |
| `held_orb` | Orb did not produce an accepted geometry; no DFT started |
| `held_dft-relax` | Native SCF, optimizer completion or structural gate failed; no SOC started |
| `held_soc-z/x/y` | A static calculation failed native completion or axis checks; no directional result accepted |
| `held_magnetic_branch` | Site moments/net magnetization changed beyond the branch policy; hold interpretation |
| `held_error` | Other execution/provenance/geometry/budget error; inspect the reason and preserved files |

An identical completed local invocation returns its saved report after artifact
checks. Changed inputs, policy, executable, environment or code refuse reuse.
An interrupted directory refuses automatic continuation. Keep the evidence and
make any new attempt explicit. Compact collected archives omit DM/HSX/PSML bytes;
those remain on the remote volume with recorded hashes. Compact collections are
for review, not a substitute for the full directory in `run_dft` reuse.

The report includes per-stage SCF counts, solver wall times, native mesh/options,
force/stress, final-axis deviations, site-moment branch checks and artifact hashes.
A separate collection receipt records archive hash and unpack time. Its
`collection_call_seconds` includes waiting for the function to finish plus
download; download-only time is unknown, not mislabeled as that full interval. Orb worker,
DFT pipeline and solver times are separate; transfer/build/allocation overhead is
not silently assigned to solver time. Raw failures are retained.

`screen.delta_E_MJ_m3` and `screen.delta_F_MJ_m3` are x−z/y−z differences. z is
parallel to relaxed c; x is the perpendicular projection of a; y = z × x. At low
symmetry these directions are a diagnostic frame, not a complete easy-axis
search. `axial` means both sampled transverse directions lie above z in both
energy definitions; it is a **preliminary sign**, without an error bar. No minimum
signal size can replace a numerical accuracy check. The relaxed scalar density
and z/x/y branches must agree within 0.2 μB site RMS and 5% net magnitude, with no
robust-site sign reversal; the final net axis must stay within 1°.

**`K1`, `K2`, and `kappa` remain null.** Ordinary SCF tolerances do not qualify small
MAE differences. The screen does not establish stability against decomposition,
magnetic ordering, synthesis feasibility, an angular anisotropy fit or physical
k/grid/basis/smearing convergence. It provides fast, consistently labeled evidence
for deciding which candidates deserve that later work.

## Validation

The [Fe2 integration record](../benchmarks/evidence/discovery-fe2-smoke.json)
records a complete CPU Orb → A100 REMAG relaxation → xyz SOC run. Orb took
3.32 s, DFT relaxation 90.06 s, and the three SOC cases 35.50 s combined. The
DFT pipeline including local checkpoint/review overhead took 145.26 s; the
GPU task allocation was 155.65 s. These are one tiny-cell smoke test, not a
performance forecast or an accuracy benchmark. Failed development submissions
are described separately in the record. The final reviewer was replayed against
the saved native evidence without new computation.

The local tests exercise failed/unfinished relaxation, magnetic-branch and axis
changes, native spin mode and density loading, mismatched grids, corrupt seeds,
process timeout, archive tampering, idempotent reuse and refusal to retry an
interrupted run. Run `python -m pytest -q` after installing `.[dev]`.
