# Contributing

Use Python 3.12 or newer and install the development dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,netcdf,modal]'
python -m pytest -q
```

The integration-patch test skips when `upstream/siesta` is absent. Run
`python scripts/build_upstream.py` to fetch and build the pinned CPU reference,
then rerun the tests to exercise that seam. Unit tests require no GPU or Modal
credentials. GPU execution requires separate Modal setup; always launch through
`scripts/run_modal.py` and keep at most one GPU instance active.

Keep `upstream/siesta` pristine. Make integration changes in the patch and
native sources, applying the patch only to disposable source copies. Changes
to numerical code need representative upstream comparisons with unchanged
scientific tolerances, including the magnetic observables relevant to the
change. Reuse existing checks and record failures as well as passing runs.

For performance work, compare complete, converged calculations on equivalent
inputs. State hardware, solver/thread settings, source revision, repetitions,
SCF counts and timing scope. Upstream GPU SIESTA is the primary competitor;
CPU comparisons and incremental REMAG comparisons must be labeled separately.
Do not infer a broad speedup from one system or multiply historical gains.

Keep generated builds, credentials, raw outputs and scratch files out of Git.
Small benchmark records intended for publication belong in
`benchmarks/evidence/`; retain the full local evidence separately. Do not
regenerate expensive GPU runs for a documentation or formatting change.

Keep AI-assistant instructions, prompts, conversation history and working
journals local. The ignore rules preserve these files on disk while excluding
them from version control; the source manifest also excludes the local planning
documents. Technical documentation, regression tests and scientific evidence
belong in the release regardless of which tools helped create them.

Describe the concrete change, relevant validation and known limits in a pull
request. Contributions use the repository's existing license; preserve all
upstream and third-party notices.
