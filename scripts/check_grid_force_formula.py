#!/usr/bin/env python3
"""Extract the pinned dfscf force/stress contraction before porting it.

This checks local contraction arithmetic with FP64 orbital values/derivatives.
It does not validate basis evaluation, sparse/periodic lookup, other force terms,
GPU execution, or the full final-force workflow.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(upstream,build,folder):
    path=upstream/'Src/dfscf.f';text=path.read_text()
    start=text.index('         DM_spbherm(:,1)');spin=text[start:text.index('      else',start)]
    start=text.index('C  Find range of a single array');limits=text[start:text.index('C  Initialise variables',start)]
    start=text.index('C  If stress required. Generate stress derivatives')
    stress=text[start:text.index('        enddo',text.index('          endif',start))]
    start=text.index('C     Copy potential to a double precision array')
    contraction=text[start:text.index('C  End of mesh point loop',start)]
    wrapper='''      program reference
      use precision, only: dp
      implicit none
      integer nc,nsp,nh,nspin,na,ifa,istr,fi,fo,ic,isp,ii,ix,iy,
     &        ix1,ix2,ip,imp,i,iu,ia,iua,ib,ispin,jc,jb
      integer,allocatable :: endpht(:),lstpht(:),indxuo(:),iaorb(:),
     &                       indxua(:),ibc(:)
      real(dp) dvol,VolCel,rvol,Dji
      real(dp),allocatable :: Dscf(:,:),DM_spbherm(:,:),D(:,:,:),
     & C(:,:),gC(:,:,:),xgC(:,:,:),displacement(:,:,:),dxsp(:,:),
     & Vscf(:,:,:),Vatm(:,:),V(:,:),Datm(:),Fal(:,:),Stressl(:),
     & gCi(:,:),CD(:),CDV(:),DF(:)
      character(2048) input,output
      call get_command_argument(1,input)
      call get_command_argument(2,output)
      open(newunit=fi,file=trim(input),access='stream',
     &     form='unformatted',status='old')
      read(fi) nc,nsp,nh,na,ifa,istr
      nspin=min(nh,4)
      allocate(endpht(0:1),lstpht(nc),indxuo(nc),iaorb(nc),
     & indxua(na),ibc(nc),Dscf(nc*nc,nh),DM_spbherm(nc*nc,nspin),
     & D(nc,nc,nspin),C(nsp,nc),gC(3,nsp,nc),xgC(9,nsp,nc),
     & displacement(3,nsp,nc),dxsp(3,nsp),Vscf(nsp,1,nspin),
     & Vatm(nsp,1),V(nsp,nspin),Datm(nc),Fal(3,na),Stressl(9),
     & gCi(12,nsp),CD(nsp),CDV(nsp),DF(12))
      read(fi) dvol,VolCel,iaorb,Dscf,Datm,C,gC,displacement,
     &         Vscf,Vatm,Fal,Stressl
      close(fi)
      if(nh==8) then
'''+spin+'''      else
        DM_spbherm=Dscf
      endif
      D=reshape(DM_spbherm,shape(D))
      endpht=[0,nc]
      do i=1,nc
        lstpht(i)=i
        indxuo(i)=i
        ibc(i)=i
      enddo
      do i=1,na
        indxua(i)=i
      enddo
      xgC=0._dp
      rvol=1._dp/VolCel
'''+limits+'''      do ic=1,nc
        dxsp=displacement(:,:,ic)
'''+stress+'''      enddo
      ip=1
'''+contraction+'''      open(newunit=fo,file=trim(output),access='stream',
     &     form='unformatted',status='replace')
      write(fo) Fal,Stressl
      close(fo)
      end program
'''
    code=folder/'reference.F';code.write_text(wrapper);exe=folder/'reference';src=build/'Src'
    link=shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())
    subprocess.run([link[0],'-O2','-ffixed-line-length-none','-I'+str(src),str(code),'-o',str(exe)],check=True)
    return exe,wrapper,hashlib.sha256((spin+limits+stress+contraction).encode()).hexdigest()


def run(upstream,build,output):
    upstream=upstream.resolve();build=build.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(641630);checks=[];arrays={}
    with tempfile.TemporaryDirectory(prefix='remag-grid-force-formula-') as tmp:
        tmp=Path(tmp);exe,wrapper,blocks=reference(upstream,build,tmp)
        for nh in (1,2,4,8):
            for nc,nsp in ((1,1),(5,8),(17,8)):
                ng=min(nh,4);na=min(nc,4)
                atom=np.arange(nc,dtype=np.int32)%na+1
                density=rng.normal(size=(nc,nc,nh));density=(density+density.swapaxes(0,1))*.05
                phi=rng.normal(size=(nsp,nc));gradient=rng.normal(size=(3,nsp,nc))*.3
                displacement=rng.normal(size=(3,nsp,nc));datm=rng.random(nc)
                potential=rng.normal(size=(nsp,ng));aux=rng.normal(size=(nsp,1));dvol=.017;volume=13.7
                start_force=rng.normal(size=(3,na));start_stress=rng.normal(size=9)
                for ifa,istr in ((1,1),(1,0),(0,1)):
                    label=f'spin{nh}_orbitals{nc}_subpoints{nsp}_force{ifa}_stress{istr}'
                    values=(density,datm,phi,gradient,displacement,potential,aux,start_force,start_stress)
                    inp=tmp/'input.bin';out=tmp/'output.bin'
                    inp.write_bytes(np.array([nc,nsp,nh,na,ifa,istr],np.int32).tobytes()+np.array([dvol,volume]).tobytes()+
                        atom.tobytes()+b''.join(a.tobytes(order='F') for a in values))
                    subprocess.run([str(exe),str(inp),str(out)],check=True,capture_output=True)
                    actual=np.fromfile(out);assert actual.shape==(3*na+9,) and np.isfinite(actual).all()
                    d=density[:,:,:ng].copy()
                    if nh==8:d[:,:,2:]=.5*(density[:,:,2:4]+density[:,:,6:8])
                    v=potential.copy()
                    if ng==4:v[:,2:]*=2.
                    cdv=sum((phi@d[:,:,spin])*v[:,spin,None] for spin in range(ng))
                    cdv=2*dvol*(cdv-phi*datm*aux)
                    force=start_force.copy();stress=start_stress.copy()
                    if ifa:
                        for orbital in range(nc):force[:,atom[orbital]-1]+=gradient[:,:,orbital]@cdv[:,orbital]
                    if istr:stress+=np.einsum('xpi,ypi,pi->xy',gradient,displacement,cdv).ravel()/volume
                    expected=np.r_[force.ravel(order='F'),stress]
                    error=float(np.max(np.abs(expected-actual)))
                    assert error<1.e-12,(label,error)
                    for name,value in zip(('density','atomic_occupation','phi','gradient','displacement','potential','auxiliary','initial_force','initial_stress'),values):arrays[label+'_'+name]=value
                    arrays[label+'_atom']=atom;arrays[label+'_result']=actual
                    checks.append(dict(fixture=label,spins=nh,nc=nc,nsp=nsp,atoms=na,forces=ifa,stress=istr,dvol=dvol,volume=volume,input_sha256=sha(inp),max_absolute_error=error,passed=True))
    artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**arrays)
    source=output.with_suffix('.F');source.write_text(wrapper)
    report=dict(scope=__doc__,checks=checks,passed=all(c['passed'] for c in checks),gpu_validation=None,
        source_sha256=sha(upstream/'Src/dfscf.f'),literal_blocks_sha256=blocks,wrapper_file=source.name,wrapper_sha256=sha(source),
        script_sha256=sha(Path(__file__)),arrays_file=artifact.name,arrays_sha256=sha(artifact))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('Grid-force contraction oracle:',len(checks),'cases; maximum error',max(c['max_absolute_error'] for c in checks))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.upstream,a.build,a.output)
