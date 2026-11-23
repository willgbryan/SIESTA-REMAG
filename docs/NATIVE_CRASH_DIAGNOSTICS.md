# Native crash evidence

The closed r6 attempt failed inside `cudaGraphInstantiate`. Offline symbolization
identifies that call but cannot distinguish a driver defect, earlier memory
corruption or an environment difference. The affected executable/library pair
remains held before paid execution. No repaired CUDA runtime is established.

New runs retain two additional artifacts through the existing hash-checked
collection path:

- `remag-scf-setup.log`: JSON lines written and closed before and after graph
  instantiation, upload and synchronization. Each line records graph dimensions,
  node count, solver lanes, API status and CUDA version-query results. A crash
  inside a call leaves its begin record. Returned API errors stop the sequence;
  there is no retry. Checkpoints occur during setup, outside graph capture and
  execution. They do not change graph nodes, numerical settings or solver order.
- `native-process.json`: the last successful sample of the owned Linux child's
  memory-map metadata and resource limits, plus NVIDIA driver version text when
  available. Sampling is bounded to twice per second with bounded file sizes.
  File offsets, address ranges and mapped paths support later symbolization;
  process memory and environment variables are not read. Missing, oversized,
  unavailable and exit-racing observations remain explicit. A failed later read
  preserves the earlier sample. Traced runs do not misidentify the tracer parent
  as the solver. Non-Linux runs report that procfs observation is unavailable.

The runner saves exit status and process context **before** parsing scientific
output, so a truncated output record cannot erase those diagnostics. A fast
native exit before CPU monitoring starts still has a waitable process status.
The maps are sampled metadata, not a core dump or hashes of mapped library bytes.
Flushed setup records survive process termination, not necessarily power loss.
An unwritable checkpoint reports to stderr without masking an API failure.

`driver_cuda_version` means the CUDA version supported by the driver, not its
installed driver-package version; failed queries retain an error code and `-1`.
The separate procfs text supplies driver-package information when available.
These distinctions follow NVIDIA's [version-query API documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART____VERSION.html).

Regression tests compile the production checkpoint helper against a CPU API
double. Deliberate SIGSEGVs and returned failures at each boundary verify retained
records and no subsequent calls. A real local child crash exercises runner,
collection and tamper rejection; Linux CI additionally observes the child's real
mapping. A historical-runner comparison demonstrates the previously lost status
when output parsing raises. These tests prove diagnostic behavior. They do not
reproduce the CUDA crash or establish a causal native repair, GPU acceptance,
latency, scientific accuracy or permission to spend the remaining budget.
