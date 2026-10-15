#!/bin/bash
# Records the machine and binaries, then runs every benchmark group in sequence.
cd ~/bench
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv
  nvcc --version 2>/dev/null | tail -2; /usr/local/cuda-12.9/bin/nvcc --version | tail -1
  lscpu | grep -E "Model name|^CPU\(s\)|Socket|Thread"; echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max)"
  free -g | head -2; uname -r
  sha256sum ~/candidate-kl/Src/siesta ~/candidate-kl/remag-native/*.so ~/reference/Src/siesta ~/reference-mpi/Src/siesta ~/bench/bench.py
  dpkg -l | grep -E "libopenblas0|libopenmpi3|libscalapack-openmpi2|gfortran-13 |libnetcdff7" | awk '{print $2,$3}'
  mpirun --version | head -1
} > machine.txt 2>&1
~/venv/bin/python -u bench.py profile lanes scaling-gpu methods-gpu threads mpi-ranks scaling-cpu methods-host scaling-serial
