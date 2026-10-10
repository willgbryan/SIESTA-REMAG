#!/usr/bin/env python3
"""Retain pristine orbital/atomic populations and total spin charge.

Extract the literal serial Mulliken population loops and link unmodified
dm_charge. This covers population option 1, including SOC hermitization;
orbital/atomic overlap-population options 2/3 and orbital moments are separate.
No GPU implementation or material substitution is qualified by this oracle.
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
    text = (upstream/'Src/mulliken.F').read_text()
    start = text.index('                 qa = 0.0_dp\n', text.index('C             Compute charge in each orbital'))
    collinear = text[start:text.index('                 qtot = qtot + qa', start)]
    branch = text.index('        elseif (spin%DM .ge. 4) then')
    start = text.index('          do io = 1,nbasis\n', branch)
    spinor = text[start:text.index('#ifdef MPI', start)]
    wrapper = '''      program reference
      use precision, only: dp
      use m_spin, only: spin
      use dm_charge_m, only: dm_charge
      implicit none
      integer natoms,nbasistot,nbasis,maxnh,nh,fi,fo,
     & ia,is,io,nao,itot,in,ind,Node,Nodes
      integer,allocatable :: numh(:),listhptr(:),lasto(:),iaorb(:)
      real(dp),allocatable :: dm(:,:),s(:),qo(:),qos(:,:),
     & partial_qa(:,:),charge(:)
      real(dp) qa,dm_aux(4)
      character(2048) input,output
      call get_command_argument(1,input)
      call get_command_argument(2,output)
      open(newunit=fi,file=trim(input),access='stream',
     & form='unformatted',status='old')
      read(fi) natoms,nbasistot,maxnh,nh
      spin%DM=nh
      spin%Grid=min(nh,4)
      spin%SO=nh==8
      nbasis=nbasistot
      Node=0
      Nodes=1
      allocate(numh(nbasis),listhptr(nbasis),lasto(0:natoms),
     & iaorb(nbasis),dm(maxnh,nh),s(maxnh),qo(nbasis),
     & qos(spin%Grid,nbasis),partial_qa(spin%Grid,natoms),
     & charge(spin%Grid))
      read(fi) numh,listhptr,lasto,iaorb,s,dm
      close(fi)
      qos=0._dp
      partial_qa=0._dp
      if(nh<=2) then
        do is=1,nh
          do ia=1,natoms
'''+collinear+'''            qos(is,lasto(ia-1)+1:lasto(ia))=
     &        qo(1:lasto(ia)-lasto(ia-1))
          enddo
        enddo
      else
'''+spinor+'''      endif
      call dm_charge(spin,nbasis,numh,listhptr,s,dm,charge)
      open(newunit=fo,file=trim(output),access='stream',
     & form='unformatted',status='replace')
      write(fo) qos,partial_qa,charge
      close(fo)
      contains
      subroutine GlobalToLocalOrb(global,node,nodes,local)
      integer,intent(in) :: global,node,nodes
      integer,intent(out) :: local
      if(node/=0.or.nodes/=1) stop 2
      local=global
      end subroutine
      subroutine LocalToGlobalOrb(local,node,nodes,global)
      integer,intent(in) :: local,node,nodes
      integer,intent(out) :: global
      if(node/=0.or.nodes/=1) stop 2
      global=local
      end subroutine
      end program
'''
    source, executable = folder/'reference.F', folder/'reference'
    source.write_text(wrapper)
    src = build/'Src'
    link = shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())
    libraries = link[next(i for i, s in enumerate(link) if Path(s).name == 'libsiesta.a'):]
    flags = (src/'CMakeFiles/SIESTA.libsiesta.dir/flags.make').read_text()
    includes = next(line.split('=', 1)[1].strip() for line in flags.splitlines() if line.startswith('Fortran_INCLUDES ='))
    subprocess.run([link[0], '-O2', '-ffixed-line-length-none', *shlex.split(includes),
                    '-I'+str(src), str(source), '-o', str(executable), *libraries], cwd=src, check=True)
    return executable, wrapper, hashlib.sha256((collinear+spinor).encode()).hexdigest()


def run(upstream, build, output):
    upstream, build, output = upstream.resolve(), build.resolve(), output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays, checks = {}, []
    rng = np.random.default_rng(615247)
    with tempfile.TemporaryDirectory(prefix='remag-population-') as temporary:
        folder = Path(temporary)
        executable, wrapper, extracted = reference(upstream, build, folder)
        for spins in (1, 2, 4, 8):
            for rows in (1, 5, 257, *((131073,262145) if spins == 8 else ())):
                # Empty atoms/rows, holes, a partial block and an actual second grid stride.
                atoms, capacity = 4, rows*8+3
                lasto = np.array([0, rows//3, rows//3, 2*rows//3, rows], np.int32)
                atom = np.repeat(np.arange(1, atoms+1, dtype=np.int32), np.diff(lasto))
                num = np.resize(np.array([0, 1, 3, 2, 5], np.int32), rows)
                num[-1] = 5
                ptr = np.arange(rows, dtype=np.int32)*8+1
                row = np.repeat(np.arange(rows), num)
                active = np.concatenate([np.arange(p, p+n) for p, n in zip(ptr, num)])
                density = rng.normal(size=(spins, capacity))*.3
                overlap = rng.normal(size=capacity)
                for mode in ('finite_padding', 'poisoned_padding', 'empty_rows'):
                    label = f'spin{spins}_rows{rows}_{mode}'
                    counts = num.copy()
                    d, s = density.copy(), overlap.copy()
                    if mode == 'poisoned_padding':
                        unused = np.ones(capacity, dtype=bool)
                        unused[active] = False
                        d[:, unused] = np.nan
                        s[unused] = np.nan
                    elif mode == 'empty_rows':
                        counts[:] = 0
                        d[:] = np.nan
                        s[:] = np.nan
                    config = np.array([atoms, rows, capacity, spins], np.int32)
                    fields = dict(num=counts, ptr=ptr, lasto=lasto, atom=atom, overlap=s, density=d)
                    raw = config.tobytes()+b''.join(a.tobytes() for a in fields.values())
                    inp, out = folder/'input.bin', folder/'output.bin'
                    inp.write_bytes(raw)
                    subprocess.run([str(executable), str(inp), str(out)], check=True, capture_output=True)
                    actual = np.fromfile(out, dtype='f8')
                    components = min(spins, 4)
                    orbital, atomic, charge = np.zeros((rows, components)), np.zeros((atoms, components)), np.zeros(components)
                    if mode != 'empty_rows':
                        values = d[:components, active].copy()
                        if spins == 8:
                            values[2:] = .5*(values[2:]+d[6:8, active])
                        values *= s[active]
                        for spin in range(components):
                            orbital[:, spin] = np.bincount(row, weights=values[spin], minlength=rows)
                            atomic[:, spin] = np.bincount(atom[row]-1, weights=values[spin], minlength=atoms)
                        charge = values.sum(axis=1)
                    expected = np.concatenate((orbital.ravel(), atomic.ravel(), charge))
                    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)
                    arrays[label+'_config'] = config
                    arrays[label+'_expected'] = actual
                    for name, value in fields.items():
                        arrays[label+'_'+name] = value
                    checks.append(dict(fixture=label, rows=rows, atoms=atoms, capacity=capacity, spins=spins,
                                       active_entries=int(counts.sum()), input_sha256=hashlib.sha256(raw).hexdigest(),
                                       max_absolute_error=float(np.max(np.abs(actual-expected))), passed=True))
    artifact, source = output.with_suffix('.npz'), output.with_suffix('.F')
    np.savez_compressed(artifact, **arrays)
    source.write_text(wrapper)
    result = dict(scope=__doc__, passed=True, gpu_validation=None, checks=checks,
                  arrays_file=artifact.name, arrays_sha256=sha(artifact), wrapper_file=source.name, wrapper_sha256=sha(source),
                  literal_blocks_sha256=extracted, reference_library_sha256=sha(build/'Src/libsiesta.a'),
                  source_sha256={name: sha(upstream/'Src'/name) for name in ('mulliken.F', 'dm_charge.F90', 'm_spin.F90')},
                  script_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Pristine population oracle:', len(checks), 'passed=True; GPU validation=None')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'build', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream, args.build, args.output)
