#!/usr/bin/env python3
"""Retain literal scalar, l=0 SOC and angular-SOC sparse force contractions.

Projector generation and material neighbor selection are separate gates.
These fixtures preserve scalar duplicate accumulation and SOC last-entry
assignment, both energy channels, complex phases and force/stress signs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import numpy as np

FIELDS = ('qr', 'qc', 'atoms', 'num', 'ptr', 'list', 'scalar', 'operators', 'xyz', 'density', 'initial')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blocks(upstream):
    text = (upstream/'Src/nlefsm.f').read_text()
    def extract(start, stop, offset=0):
        begin = text.index(start, offset)
        return text[begin:text.index(stop, begin)].rstrip()+'\n'
    scalar_density = extract('          do j = 1,numh(iio)', '\n\n! Find matrix elements')
    scalar_force = extract('                   Cijk = Di(jo) * epsk', '\n                endif')
    soc_density = extract('           Ds(1:2,1:2,1:no) =', '\n          endif')
    zero_force = extract('               if (split_sr_so) then', '\n              endif\n              ko = ko + 1', text.index('              if (.not. matrix_elements_only) then', text.index('             if ( l.eq.0 ) then')))
    angular_force = extract('               Enl = Enl +  Di(jo) * Vit', '\n              endif\n              ko = ko+2', text.index('c------------ Forces & SO contribution'))
    return scalar_density, scalar_force, soc_density, zero_force, angular_force


def reference(upstream, build, folder):
    sd, sf, od, zf, af = blocks(upstream)
    wrapper = '''      program reference
      use iso_fortran_env, only: dp=>real64
      implicit none
      logical,parameter :: matrix_elements_only=.false.
      logical split_sr_so
      integer nq,na,nr,no,cap,components,spinor,cycles,kind,split,
     & fi,fo,q,cycle,iio,j,ind,jo,ispin,ia,kua,ix,jx,nspin,
     & ino,jno,koa,ikb
      integer,allocatable :: qr(:),qc(:),atoms(:,:),numh(:),
     & listhptr(:),listh(:)
      logical,allocatable :: listed(:)
      real(dp),allocatable :: Dscf(:,:),Di(:),values(:,:),
     & ops(:,:),xyz(:,:),xno(:,:),fa(:,:),Ski(:,:),grSki(:,:,:)
      real(dp) volume,Enl,stress(3,3),epsk(2),Vit,Vit_saved,
     & Cijk,CVj,fik,er(4),ei(4)
      complex(dp),allocatable :: Ds(:,:,:)
      complex(dp) E_offsiteSO(4),V_sot(2,2),F_so(3,2,2)
      character(2048) input,output
      call get_command_argument(1,input)
      call get_command_argument(2,output)
      open(newunit=fi,file=trim(input),access='stream',
     & form='unformatted',status='old')
      read(fi) nq,na,nr,no,cap,components,spinor,cycles,kind,split
      allocate(qr(nq),qc(nq),atoms(2,nq),numh(nr),listhptr(nr),
     & listh(cap),Dscf(cap,components),Di(no),Ds(2,2,no),
     & listed(no),values(6,nq),ops(33,nq),xyz(3,nq),
     & xno(3,2*nq),fa(3,na),Ski(1,2*nq),grSki(3,1,2*nq))
      read(fi) volume,qr,qc,atoms,numh,listhptr,listh,values,
     & ops,xyz,Dscf,Enl,er,ei,fa,stress
      close(fi)
      E_offsiteSO=cmplx(er,ei,dp)
      nspin=components
      if(kind==0) nspin=spinor
      split_sr_so=split==1
      do q=1,nq
        Ski(1,2*q-1)=values(2,q)
        Ski(1,2*q)=values(3,q)
        grSki(:,1,2*q-1)=values(4:6,q)
        grSki(:,1,2*q)=0.0_dp
        xno(:,2*q-1)=xyz(:,q)
        xno(:,2*q)=0.0_dp
      enddo
      do cycle=1,cycles
        do q=1,nq
          iio=qr(q)
          Di=0.0_dp
          listed=.false.
          if(kind==0) then
'''+sd+'''          else
'''+od+'''          endif
          jo=qc(q)
          ia=atoms(1,q)
          kua=atoms(2,q)
          ino=2*q-1
          jno=2*q
          koa=1
          ikb=1
          if(kind==0) then
            call apply_scalar()
          else if(kind==1) then
            epsk(1)=values(1,q)
            Vit_saved=epsk(1)*Ski(koa,ino)*Ski(koa,jno)
            Vit=0.0_dp
            if(split_sr_so) Vit=Vit_saved
'''+zf+'''          else
            Vit=ops(1,q)
            V_sot=reshape(cmplx(ops(2:5,q),ops(6:9,q),dp),
     &                    [2,2])
            F_so=reshape(cmplx(ops(10:21,q),ops(22:33,q),dp),
     &                   [3,2,2])
'''+af+'''          endif
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
      epsk=values(1,q)
      Sik=Ski(ikb,ino)
      Sjk=Ski(ikb,jno)
'''+sf+'''      end subroutine
      end program
'''
    source, executable = folder/'reference.F', folder/'reference'
    source.write_text(wrapper)
    compiler = shlex.split((build/'Src/CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())[0]
    subprocess.run([compiler, '-O2', '-ffixed-line-length-none', str(source), '-o', str(executable)], check=True, cwd=folder)
    return executable, wrapper


def binary(config, volume, data):
    return config.tobytes()+np.array([volume], dtype='d').tobytes()+b''.join(data[k].tobytes() for k in FIELDS)


def run(upstream, build, output):
    upstream, build, output = upstream.resolve(), build.resolve(), output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(613947)
    checks, arrays = [], {}
    signatures = [(0, c, 0, n) for c in (1, 2, 4, 8) for n in (0, 1, 257)]
    signatures += [(kind, 8, split, n) for kind in (1, 2) for split in (0, 1) for n in (0, 1, 257)]
    signatures += [(kind, 8, 1 if kind else 0, 131073) for kind in (0, 1, 2)]
    with tempfile.TemporaryDirectory(prefix='remag-nonlocal-force-') as temporary:
        folder = Path(temporary)
        executable, wrapper = reference(upstream, build, folder)
        for kind, components, split, n in signatures:
            na, nr, nc, cap = 4, 5, 7, 63
            num = np.array([5, 0, 7, 4, 9], np.int32)
            ptr = np.arange(nr, dtype=np.int32)*12+1
            columns = np.zeros(cap, np.int32)
            for i in range(nr):
                columns[ptr[i]:ptr[i]+num[i]] = [1, 1, 3, 6, 6, 7, 2, 3, 4][:num[i]]
            data = dict(qr=rng.integers(1, nr+1, n, dtype=np.int32), qc=rng.integers(1, nc+1, n, dtype=np.int32),
                        atoms=rng.integers(1, na+1, (n, 2), dtype=np.int32), num=num, ptr=ptr, list=columns,
                        scalar=rng.normal(size=(n, 6))*.2, operators=rng.normal(size=(n, 33))*.1,
                        xyz=rng.normal(size=(n, 3)), density=rng.normal(size=(components, cap))*.3,
                        initial=rng.normal(size=3*na+18))
            if kind == 2 and not split:
                data['operators'][:, 0] = 0.
            if n:
                data['qr'][-1] = data['qc'][-1] = 1
                data['atoms'][-1] = [1, 2]
            for cycles in (1, 3):
                label = f'kind{kind}_components{components}_split{split}_queries{n}_cycles{cycles}'
                config = np.array([n, na, nr, nc, cap, components, min(2, components), cycles, kind, split], np.int32)
                volume = 13.7
                inp, out = folder/'input.bin', folder/'output.bin'
                raw = binary(config, volume, data)
                inp.write_bytes(raw)
                subprocess.run([str(executable), str(inp), str(out)], check=True, capture_output=True)
                expected = np.fromfile(out)
                assert expected.shape == (3*na+18,) and np.isfinite(expected).all()
                arrays[label+'_config'], arrays[label+'_volume'], arrays[label+'_expected'] = config, np.array([volume]), expected
                for k, v in data.items(): arrays[label+'_'+k] = v
                if n:
                    reduced = {**data, 'scalar': data['scalar'].copy(), 'operators': data['operators'].copy()}
                    reduced['scalar'][-1] = 0.
                    reduced['operators'][-1] = 0.
                    inp.write_bytes(binary(config, volume, reduced))
                    subprocess.run([str(executable), str(inp), str(out)], check=True, capture_output=True)
                    arrays[label+'_without_last'] = np.fromfile(out)
                else:
                    arrays[label+'_without_last'] = expected.copy()
                checks.append(dict(fixture=label, input_sha256=hashlib.sha256(raw).hexdigest(), kind=kind,
                                   components=components, split=bool(split), queries=n, cycles=cycles, passed=True))
    archive, source = output.with_suffix('.npz'), output.with_suffix('.F')
    np.savez_compressed(archive, **arrays)
    source.write_text(wrapper)
    report = dict(scope=__doc__, passed=True, gpu_validation=None, checks=checks, arrays_file=archive.name, arrays_sha256=sha(archive),
                  wrapper_file=source.name, wrapper_sha256=sha(source), source_sha256=sha(upstream/'Src/nlefsm.f'),
                  literal_blocks_sha256=hashlib.sha256(''.join(blocks(upstream)).encode()).hexdigest(), script_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('Pristine nonlocal-force oracle:', len(checks), 'fixtures')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'build', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream, args.build, args.output)
