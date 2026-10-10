#!/usr/bin/env python3
"""Retain pinned partial-core/neutral-atom force contraction fixtures.

Literal upstream arithmetic is checked against NumPy, including tensor layout,
diagonal-spin averaging and accumulation into existing force/stress. Radial
derivative evaluation, real-material indexing and GPU execution are not tested.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(upstream, build, folder):
    text = (upstream/'Src/meshsubs.F').read_text()
    start = text.index('C                   Calculate gradients of PCC')
    core = text[start:text.index('                  else', start)]
    start = text.index('                if (Gradients) then', text.index('      subroutine NeutralAtomOnMesh'))
    start = text.index('\n', start)+1
    neutral = text[start:text.index('                else', start)]
    wrapper = '''      program reference
      use iso_fortran_env, only: dp=>real64
      implicit none
      integer kind,nentries,ntpl,na,nsd,ifa,istr,fi,fo,k,ip,iua,i,ispin
      integer,allocatable :: points(:),atoms(:)
      real(dp) dvol,volume,vxc,dx(3),dfa(3),grrho(3),grva(3)
      real(dp),allocatable :: gradient(:,:),displacement(:,:),
     &  Vscf(:,:),Vaux(:),drho(:),fal(:,:),stressl(:,:)
      logical Forces,Stress
      character(2048) input,output
      call get_command_argument(1,input)
      call get_command_argument(2,output)
      open(newunit=fi,file=trim(input),access='stream',
     &     form='unformatted',status='old')
      read(fi) kind,nentries,ntpl,na,nsd,ifa,istr
      allocate(points(nentries),atoms(nentries),gradient(3,nentries),
     & displacement(3,nentries),Vscf(ntpl,nsd),Vaux(ntpl),
     & drho(ntpl),fal(3,na),stressl(3,3))
      read(fi) dvol,volume,points,atoms,gradient,displacement,
     &         Vscf,Vaux,drho,fal,stressl
      close(fi)
      Forces=ifa==1
      Stress=istr==1
      do k=1,nentries
        ip=points(k)
        iua=atoms(k)
        dx=displacement(:,k)
        grrho=gradient(:,k)
        grva=gradient(:,k)
        if(kind==1) then
'''+core+'''        else
'''+neutral+'''        endif
      enddo
      open(newunit=fo,file=trim(output),access='stream',
     &     form='unformatted',status='replace')
      write(fo) fal,stressl
      close(fo)
      end program
'''
    code = folder/'reference.F'
    code.write_text(wrapper)
    exe = folder/'reference'
    compiler = shlex.split((build/'Src/CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())[0]
    subprocess.run([compiler, '-O2', '-ffixed-line-length-none', str(code), '-o', str(exe)], check=True)
    return exe, wrapper, hashlib.sha256((core+neutral).encode()).hexdigest()


def run(upstream, build, output):
    upstream, build, output = upstream.resolve(), build.resolve(), output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(985103)
    checks, arrays = [], {}
    with tempfile.TemporaryDirectory(prefix='remag-atomic-grid-force-') as temp:
        folder = Path(temp)
        exe, wrapper, blocks = reference(upstream, build, folder)
        for kind, nsd in ((1, 1), (1, 2), (2, 1)):
            for nentries, points_count, atoms_count in ((0, 8, 4), (1, 1, 1), (257, 23, 4)):
                points = rng.integers(1, points_count+1, nentries, dtype=np.int32)
                atoms = rng.integers(1, atoms_count+1, nentries, dtype=np.int32)
                gradient = rng.normal(size=(3, nentries))
                displacement = rng.normal(size=(3, nentries))
                potential = rng.normal(size=(points_count, nsd))
                auxiliary = rng.normal(size=points_count)
                deformation = rng.normal(size=points_count)
                initial_force = rng.normal(size=(3, atoms_count))
                initial_stress = rng.normal(size=(3, 3))
                dvol, volume = .017, 13.7
                values = (gradient, displacement, potential, auxiliary, deformation, initial_force, initial_stress)
                for ifa, istr in ((1, 1), (1, 0), (0, 1)):
                    label = f'kind{kind}_nsd{nsd}_entries{nentries}_force{ifa}_stress{istr}'
                    inp, out = folder/'input.bin', folder/'output.bin'
                    inp.write_bytes(np.array([kind, nentries, points_count, atoms_count, nsd, ifa, istr], np.int32).tobytes()+
                        np.array([dvol, volume]).tobytes()+points.tobytes()+atoms.tobytes()+
                        b''.join(a.tobytes(order='F') for a in values))
                    subprocess.run([str(exe), str(inp), str(out)], check=True, capture_output=True)
                    actual = np.fromfile(out)
                    scalar = ((potential.mean(axis=1)-auxiliary)[points-1] if kind == 1 else deformation[points-1])
                    df = dvol*gradient*scalar
                    force, stress = initial_force.copy(), initial_stress.copy()
                    if ifa:
                        for entry, atom in enumerate(atoms):
                            force[:, atom-1] += df[:, entry]
                    if istr:
                        stress += displacement@df.T/volume
                    expected = np.r_[force.ravel(order='F'), stress.ravel(order='F')]
                    error = float(np.max(np.abs(expected-actual)))
                    assert np.isfinite(actual).all() and error < 1e-12, (label, error)
                    for name, value in zip(('gradient', 'displacement', 'potential', 'auxiliary', 'deformation', 'initial_force', 'initial_stress'), values):
                        arrays[label+'_'+name] = value
                    arrays[label+'_points'], arrays[label+'_atoms'], arrays[label+'_result'] = points, atoms, actual
                    checks.append(dict(fixture=label, kind=kind, nsd=nsd, entries=nentries, points=points_count,
                        atoms=atoms_count, forces=ifa, stress=istr, dvol=dvol, volume=volume,
                        input_sha256=sha(inp), max_absolute_error=error, passed=True))
    artifact = output.with_suffix('.npz')
    np.savez_compressed(artifact, **arrays)
    source = output.with_suffix('.F')
    source.write_text(wrapper)
    report = dict(scope=__doc__, checks=checks, passed=all(c['passed'] for c in checks), gpu_validation=None,
        source_sha256=sha(upstream/'Src/meshsubs.F'), literal_blocks_sha256=blocks,
        wrapper_file=source.name, wrapper_sha256=sha(source), script_sha256=sha(Path(__file__)),
        arrays_file=artifact.name, arrays_sha256=sha(artifact))
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('Atomic grid-force oracle:', len(checks), 'cases; maximum error', max(c['max_absolute_error'] for c in checks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'build', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream, args.build, args.output)
