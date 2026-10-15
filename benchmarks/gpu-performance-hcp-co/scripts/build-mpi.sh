#!/bin/bash
# Restore the toolchain (the image resets on wake), then build pristine SIESTA 5.4.2 with
# MPI + ScaLAPACK (the multicore upstream baseline) from the same pinned source as ~/reference.
set -uxo pipefail
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq > ~/apt.log 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gfortran g++ cmake make pkg-config \
  libopenblas-dev libnetcdff-dev libnetcdf-dev bc libopenmpi-dev openmpi-bin libscalapack-openmpi-dev >> ~/apt.log 2>&1
echo APT_RC=$?
dpkg -l | grep -E "libopenblas0|libopenmpi3|libscalapack-openmpi2|gfortran-13 |libnetcdff7" | awk '{print $2,$3}'
ldd ~/reference/Src/siesta ~/candidate-kl/Src/siesta | grep "not found"
cd ~
FLAGS="-DSIESTA_WITH_MPI=ON"
for f in ELSI DFTD3 FLOOK OPENMP FFTW HOST_OPTIMIZATION; do FLAGS="$FLAGS -DSIESTA_WITH_$f=OFF"; done
FLAGS="$FLAGS -DSIESTA_WITH_NETCDF=ON -DSIESTA_WITH_NCDF=ON"
rm -rf reference-mpi
cmake -S reference-src -B reference-mpi -DCMAKE_BUILD_TYPE=Release $FLAGS \
  -DSCALAPACK_LIBRARY=/usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so > ~/cmake-mpi.log 2>&1
echo CMAKE_RC=$?
cmake --build reference-mpi --target SIESTA.siesta --parallel 24 > ~/make-mpi.log 2>&1
echo MPI_BUILD_RC=$?
ls -l reference-mpi/Src/siesta
grep -E "SIESTA_WITH_MPI:|SCALAPACK_LIBRARY:" reference-mpi/CMakeCache.txt
