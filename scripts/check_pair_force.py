#!/usr/bin/env python3
"""Retain literal upstream kinetic/overlap sparse contraction oracles.

Neighbor selection, radial integrals and material ownership are separate gates.
The wrappers use the original sparse diagonal-spin summation and force blocks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reference(upstream,build,folder):
    kinetic=(upstream/'Src/kinefsm.F').read_text();overlap=(upstream/'Src/overfsm.f90').read_text()
    start=kinetic.index('               do j = 1,numd(iio)');kd=kinetic[start:kinetic.index('            endif',start)]
    start=overlap.index('          do j = 1,numd(iio)');od=overlap[start:overlap.index('          do jn',start)]
    start=kinetic.index('                     Ekin = Ekin + Di(jo) * Tij');kf=kinetic[start:kinetic.index('                  endif',start)]
    start=overlap.index('                do ix = 1,3');of=overlap[start:overlap.index('              end if',start)]
    wrapper='''      program reference
      use iso_fortran_env, only: dp=>real64
      implicit none
      type SpinType
        integer spinor
      end type
      type(SpinType) spin
      integer nq,na,nr,nc,cap,components,nspin,cycles,kind,fi,fo,
     &        q,cycle,iio,j,ind,jo,ispin,ia,jua,ix,jx,jn
      integer,allocatable :: qr(:),qc(:),atoms(:,:),numd(:),
     &        listdptr(:),listd(:)
      real(dp),allocatable :: Dscf(:,:),Escf(:,:),Di(:),values(:,:),
     &        xij(:,:),fa(:,:)
      real(dp) volume,Ekin,stress(3,3),Tij,grTij(3),grSij(3),fij(3)
      character(2048) input,output
      call get_command_argument(1,input)
      call get_command_argument(2,output)
      open(newunit=fi,file=trim(input),access='stream',
     &     form='unformatted',status='old')
      read(fi) nq,na,nr,nc,cap,components,nspin,cycles,kind
      allocate(qr(nq),qc(nq),atoms(2,nq),numd(nr),listdptr(nr),
     & listd(cap),Dscf(cap,components),Escf(cap,components),Di(nc),
     & values(4,nq),xij(3,nq),fa(3,na))
      read(fi) volume,qr,qc,atoms,numd,listdptr,listd,values,xij,
     &         Dscf,Ekin,fa,stress
      close(fi)
      Escf=Dscf
      spin%spinor=nspin
      do cycle=1,cycles
        do q=1,nq
          iio=qr(q)
          Di=0._dp
          if(kind==1) then
'''+kd+'''          else
'''+od+'''          endif
          jo=qc(q)
          ia=atoms(1,q)
          jua=atoms(2,q)
          jn=q
          Tij=values(1,q)
          grTij=values(2:4,q)
          grSij=values(2:4,q)
          if(kind==1) then
'''+kf+'''          else
'''+of+'''          endif
        enddo
      enddo
      open(newunit=fo,file=trim(output),access='stream',
     &     form='unformatted',status='replace')
      write(fo) Ekin,fa,stress
      close(fo)
      end program
'''
    code=folder/'reference.F';code.write_text(wrapper);exe=folder/'reference'
    compiler=shlex.split((build/'Src/CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())[0]
    subprocess.run([compiler,'-O2','-ffixed-line-length-none',str(code),'-o',str(exe)],check=True)
    return exe,wrapper,hashlib.sha256((kd+od+kf+of).encode()).hexdigest()

FIELDS=('qr','qc','atoms','num','ptr','list','integrals','xyz','density','initial')
def binary(config,kind,volume,d):
    return np.r_[config,np.int32(kind)].astype(np.int32).tobytes()+np.array([volume],dtype='d').tobytes()+b''.join(d[k].tobytes(order='C') for k in FIELDS)

def run(upstream,build,output):
    upstream,build,output=upstream.resolve(),build.resolve(),output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(613207);checks=[];arrays={}
    with tempfile.TemporaryDirectory(prefix='remag-pair-force-') as tmp:
        folder=Path(tmp);exe,wrapper,blocks=reference(upstream,build,folder)
        for components in (1,2,4,8):
            spinor=min(components,2)
            for n in (0,1,257,*((131073,) if components==8 else ())):
                na=4;nr=5;nc=7;capacity=nr*12+3
                num=np.array([5,0,7,4,9],np.int32);ptr=np.arange(nr,dtype=np.int32)*12+1;cols=np.zeros(capacity,np.int32)
                for i in range(nr):cols[ptr[i]:ptr[i]+num[i]]=[1,1,3,6,6,7,2,3,4][:num[i]]
                d=dict(qr=rng.integers(1,nr+1,n,dtype=np.int32),qc=rng.integers(1,nc+1,n,dtype=np.int32),atoms=rng.integers(1,na+1,(n,2),dtype=np.int32),
                       num=num,ptr=ptr,list=cols,integrals=rng.normal(size=(n,4))*.2,xyz=rng.normal(size=(n,3)),density=rng.normal(size=(components,capacity))*.3,
                       initial=rng.normal(size=3*na+10))
                if n:d['qr'][-1]=1;d['qc'][-1]=1;d['atoms'][-1]=[1,2]
                for kind in (1,4):
                    for cycles in (1,3):
                        label=f'kind{kind}_components{components}_queries{n}_cycles{cycles}';config=np.array([n,na,nr,nc,capacity,components,spinor,cycles],np.int32);volume=13.7
                        inp=folder/'input.bin';out=folder/'output.bin';raw=binary(config,kind,volume,d);inp.write_bytes(raw)
                        subprocess.run([str(exe),str(inp),str(out)],check=True,capture_output=True);actual=np.fromfile(out);assert actual.shape==(3*na+10,) and np.isfinite(actual).all()
                        arrays[label+'_config']=config;arrays[label+'_parameters']=np.array([kind,volume]);arrays[label+'_expected']=actual
                        for k,v in d.items():arrays[label+'_'+k]=v
                        # Retain a pristine reference excluding the final query for strided-tail fault probes.
                        if n:
                            reduced={**d,'integrals':d['integrals'].copy()};reduced['integrals'][-1]=0.;inp.write_bytes(binary(config,kind,volume,reduced))
                            subprocess.run([str(exe),str(inp),str(out)],check=True,capture_output=True);arrays[label+'_without_last']=np.fromfile(out)
                        else:arrays[label+'_without_last']=actual.copy()
                        checks.append(dict(fixture=label,input_sha256=hashlib.sha256(raw).hexdigest(),kind=kind,queries=n,components=components,spinor=spinor,cycles=cycles,passed=True))
    archive=output.with_suffix('.npz');np.savez_compressed(archive,**arrays);source=output.with_suffix('.F');source.write_text(wrapper)
    report=dict(scope=__doc__,passed=True,gpu_validation=None,checks=checks,arrays_file=archive.name,arrays_sha256=sha(archive),wrapper_file=source.name,wrapper_sha256=sha(source),
                literal_blocks_sha256=blocks,source_sha256={name:sha(upstream/'Src'/name) for name in ('kinefsm.F','overfsm.f90')},script_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(report,indent=2)+'\n');print('Pristine pair-force oracle:',len(checks),'fixtures')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','build','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.upstream,a.build,a.output)
