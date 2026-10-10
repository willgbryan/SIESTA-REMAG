#!/usr/bin/env python3
"""Compose literal pristine projector and sparse-force blocks over compact geometry.

The CPU reference derives sparse selection directly from CSR, independently of
the GPU lookup. Includes mixed projector blocks, unused storage, excluded local
rows, periodic atom identities, duplicate sparse entries and the strided tail.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import numpy as np
from check_nonlocal_force import blocks, sha
from check_soc_projector import block as projector

FIELDS = ('centers', 'neighbors', 'block_indices', 'epsilon', 'num', 'ptr', 'list',
          'lookup', 'integrals', 'xyz', 'density', 'initial')


def reference(upstream, build, folder):
    sd, sf, od, zf, af = blocks(upstream)
    wrapper = '''      module pristine_soc
      use iso_fortran_env, only: dp=>real64
      implicit none
      contains
'''+projector(upstream)+'''      end module
      program reference
      use pristine_soc
      implicit none
      logical,parameter :: matrix_elements_only=.false.
      logical split_sr_so
      integer nn,nc,nb,nq,na,nr,no,cap,components,spinor,soc,
     & split,cycles,fi,fo,cycle,c,b,base,np,nnc,q,p,
     & iio,j,ind,jo,ispin,ia,kua,ix,jx,nspin,ino,jno,koa,ikb,l
      integer,allocatable :: centers(:,:),neighbors(:,:),
     & indices(:,:),numh(:),listhptr(:),listh(:),lookup(:,:)
      logical,allocatable :: listed(:)
      real(dp),allocatable :: Dscf(:,:),Di(:),integrals(:,:),
     & epsilon(:,:),xyz(:,:),xno(:,:),fa(:,:),Ski(:,:),
     & grSki(:,:,:)
      real(dp) volume,Enl,stress(3,3),epsk(2),Vit,Vit_saved,
     & Cijk,CVj,fik,er(4),ei(4)
      complex(dp),allocatable :: Ds(:,:,:)
      complex(dp) E_offsiteSO(4),V_sot(2,2),F_so(3,2,2)
      character(2048) input,output
      call get_command_argument(1,input)
      call get_command_argument(2,output)
      open(newunit=fi,file=trim(input),access='stream',
     & form='unformatted',status='old')
      read(fi) nn,nc,nb,nq,na,nr,no,cap,components,spinor,soc,
     & split,cycles
      allocate(centers(7,nc),neighbors(4,nn),indices(2,nb),
     & epsilon(2,nb),numh(nr),listhptr(nr),listh(cap),
     & lookup(no,nr),integrals(4,nq),xyz(3,nq),
     & Dscf(cap,components),Di(no),Ds(2,2,no),listed(no),
     & fa(3,na))
      read(fi) volume,centers,neighbors,indices,epsilon,
     & numh,listhptr,listh,lookup,integrals,xyz,Dscf,
     & Enl,er,ei,fa,stress
      close(fi)
      E_offsiteSO=cmplx(er,ei,dp)
      nspin=components
      if(soc==0) nspin=spinor
      split_sr_so=split==1
      do cycle=1,cycles
       do c=1,nc
        base=centers(1,c)
        nnc=centers(2,c)
        np=centers(5,c)
        kua=centers(7,c)
        allocate(Ski(np,nnc),grSki(3,np,nnc),xno(3,nnc))
        do ino=1,nnc
         q=centers(6,c)+(ino-1)*np
         xno(:,ino)=xyz(:,q+1)
         do p=1,np
          Ski(p,ino)=integrals(1,q+p)
          grSki(:,p,ino)=integrals(2:4,q+p)
         enddo
        enddo
        do ino=1,nnc
         iio=neighbors(1,base+ino)
         if(iio==0) cycle
         ia=neighbors(3,base+ino)
         Di=0.0_dp
         listed=.false.
         if(soc==0) then
'''+sd+'''         else
'''+od+'''          do j=1,numh(iio)
           listed(listh(listhptr(iio)+j))=.true.
          enddo
         endif
         do jno=1,nnc
          jo=neighbors(2,base+jno)
          if(.not.listed(jo)) cycle
          do b=centers(3,c)+1,centers(3,c)+centers(4,c)
           koa=indices(1,b)+1
           ikb=koa
           l=indices(2,b)
           epsk=epsilon(:,b)
           if(soc==0) then
            call apply_scalar()
           else if(l==0) then
            Vit_saved=epsk(1)*Ski(koa,ino)*Ski(koa,jno)
            Vit=0.0_dp
            if(split_sr_so) Vit=Vit_saved
'''+zf+'''           else
            call calc_Vj_offsiteSO(l,epsk,Ski(koa,ino),
     &       Ski(koa,jno),grSki(1,koa,ino),grSki(1,koa,jno),
     &       Vit,V_sot,F_so,split_sr_so)
'''+af+'''           endif
          enddo
         enddo
        enddo
        deallocate(Ski,grSki,xno)
       enddo
      enddo
      open(newunit=fo,file=trim(output),access='stream',
     & form='unformatted',status='replace')
      write(fo) Enl,real(E_offsiteSO,dp),aimag(E_offsiteSO),
     & fa,stress
      close(fo)
      contains
      subroutine apply_scalar()
      real(dp) epsk,Sik,Sjk
      epsk=epsilon(1,b)
      Sik=Ski(ikb,ino)
      Sjk=Ski(ikb,jno)
'''+sf+'''      end subroutine
      end program
      subroutine die(message)
      character(*) message
      write(*,*) message
      stop 1
      end subroutine
'''
    source, executable = folder/'reference.F', folder/'reference'
    source.write_text(wrapper)
    compiler = shlex.split((build/'Src/CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())[0]
    subprocess.run([compiler, '-O2', '-fcheck=bounds', '-ffixed-line-length-none', str(source), '-o', str(executable)], check=True, cwd=folder)
    return executable, wrapper


def signatures():
    base = [(0, c, 0, 0, n) for c in (1, 2, 4, 8) for n in (0, 1, 19)]
    base += [(1, 8, s, l, n) for s in (0, 1) for l in range(6) for n in (1, 19)]
    for soc, components, split, l, n in base:
        for cycles in (1, 3):
            yield soc, components, split, l, n, cycles
    yield 0, 8, 0, 0, 131073, 1
    for split in (0, 1):
        yield 1, 8, split, 1, 131073, 1


def fixture(rng, soc, components, split, l, n, cycles):
    na, nr, no, cap = 4, 5, 7, 63
    num = np.array([5, 0, 7, 4, 9], np.int32)
    ptr = np.arange(nr, dtype=np.int32)*12+1
    columns = np.zeros(cap, np.int32)
    lookup = np.full((nr, no), -1, np.int32)
    for row in range(nr):
        columns[ptr[row]:ptr[row]+num[row]] = [1, 1, 3, 6, 6, 7, 2, 3, 4][:num[row]]
        for i in range(ptr[row], ptr[row]+num[row]):
            lookup[row, columns[i]-1] = i
    centers, neighbors, indices, epsilon = [], [], [], []
    tail = n == 131073
    counts = [1]*n if tail else [3, 0, 4, 1, 2, 9] if n == 19 else [n] if n else []
    queries = 0 if tail else 2
    signs = [(1.3, .8), (-.4, -.9), (.7, -1.1), (-.6, .9), (0., .7), (0., 0.)]
    for c, count in enumerate(counts):
        width = 2*(2*l+1) if l else 1
        descriptors = [(0, l)] if tail else [(2, l), (width+3, 0)] if soc else [(4, 0), (1, 0), (3, 0)]
        projectors = width if tail else width+5 if soc else 6
        base = len(neighbors)
        centers.append([base, count, len(indices), len(descriptors), projectors, queries, 2 if tail else c % na+1])
        for i in range(count):
            neighbors.append([int(rng.integers(0, nr+1)), int(rng.integers(1, no+1)), int(rng.integers(1, na+1)), c])
        for offset, angular in descriptors:
            indices.append([offset, angular])
            epsilon.append(signs[0 if tail else c % len(signs)])
        queries += count*projectors+(0 if tail else 2)
    data = dict(centers=np.array(centers, np.int32).reshape(-1, 7), neighbors=np.array(neighbors, np.int32).reshape(-1, 4),
                block_indices=np.array(indices, np.int32).reshape(-1, 2), epsilon=np.array(epsilon, 'd').reshape(-1, 2),
                num=num, ptr=ptr, list=columns, lookup=lookup,
                integrals=rng.normal(size=(queries, 4))*.1, xyz=rng.normal(size=(queries, 3)),
                density=rng.normal(size=(components, cap))*.3, initial=rng.normal(size=3*na+18))
    if n:
        data['neighbors'][0, :3] = [1, 1, 1]
        data['neighbors'][-1, :3] = [1, 1, 1]
    if n == 19:
        # Neighbor 1 is right-only; its atom may be a supercell image.
        data['neighbors'][1, :3] = [0, 1, na+17]
    for base, count, _, _, projectors, query, _ in data['centers']:
        for i in range(count):
            first = query+i*projectors
            data['xyz'][first:first+projectors] = data['xyz'][first]
    config = np.array([n, len(centers), len(indices), queries, na, nr, no, cap, components, min(2, components), soc, split, cycles], np.int32)
    return config, 13.7, data


def binary(config, volume, data):
    return config.tobytes()+np.array([volume], 'd').tobytes()+b''.join(data[k].tobytes() for k in FIELDS)


def run(upstream, build, output):
    upstream, build, output = upstream.resolve(), build.resolve(), output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(613981)
    checks, arrays = [], {}
    with tempfile.TemporaryDirectory(prefix='remag-nonlocal-compact-') as temporary:
        folder = Path(temporary)
        executable, wrapper = reference(upstream, build, folder)
        for soc, components, split, l, n, cycles in signatures():
            label = f'soc{soc}_components{components}_split{split}_l{l}_neighbors{n}_cycles{cycles}'
            config, volume, data = fixture(rng, soc, components, split, l, n, cycles)
            inp, out = folder/'input.bin', folder/'output.bin'
            raw = binary(config, volume, data)
            inp.write_bytes(raw)
            subprocess.run([str(executable), str(inp), str(out)], check=True, capture_output=True)
            expected = np.fromfile(out)
            assert expected.shape == data['initial'].shape and np.isfinite(expected).all()
            arrays[label+'_config'], arrays[label+'_volume'], arrays[label+'_expected'] = config, np.array([volume]), expected
            for k, v in data.items(): arrays[label+'_'+k] = v
            if n:
                reduced = {**data, 'neighbors': data['neighbors'].copy()}
                reduced['neighbors'][-1, 0] = 0
                inp.write_bytes(binary(config, volume, reduced))
                subprocess.run([str(executable), str(inp), str(out)], check=True, capture_output=True)
                arrays[label+'_without_last'] = np.fromfile(out)
            else:
                arrays[label+'_without_last'] = expected.copy()
            checks.append(dict(fixture=label, input_sha256=hashlib.sha256(raw).hexdigest(), soc=soc, components=components,
                               split=split, l=l, neighbors=n, cycles=cycles, passed=True))
    archive, source = output.with_suffix('.npz'), output.with_suffix('.F')
    np.savez_compressed(archive, **arrays)
    source.write_text(wrapper)
    snippets = ''.join(blocks(upstream))+projector(upstream)
    report = dict(scope=__doc__, passed=True, gpu_validation=None, checks=checks, arrays_file=archive.name, arrays_sha256=sha(archive),
                  wrapper_file=source.name, wrapper_sha256=sha(source), source_sha256=sha(upstream/'Src/nlefsm.f'),
                  literal_blocks_sha256=hashlib.sha256(snippets.encode()).hexdigest(), script_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('Pristine compact nonlocal oracle:', len(checks), 'fixtures', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'build', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream, args.build, args.output)
