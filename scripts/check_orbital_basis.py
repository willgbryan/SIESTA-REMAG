#!/usr/bin/env python3
"""Pristine all_phi/rad_get/rlylm oracle for GPU force-derivative generation.

The executable links the unmodified upstream library. Synthetic species tables
cover complete angular shells through l=5, multiple radial cutoffs, empty radial
functions, mesh-node boundaries, the origin and the general near-axis branch.
This is component evidence; material topology is a separate validation gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import numpy as np

WRAPPER='''program reference
 use precision, only: dp
 use units, only: pi
 use atm_types, only: species,nspecies,maxnorbs
 use atmfuncs, only: all_phi
 implicit none
 integer fi,fo,is,ir,nr,no,nq,i,nphi,j
 integer,allocatable :: qs(:),qo(:)
 real(dp),allocatable :: xyz(:,:),result(:,:),phi(:),grad(:,:)
 real(dp) r2cut,r2
 character(2048) input,output
 call get_command_argument(1,input);call get_command_argument(2,output)
 open(newunit=fi,file=trim(input),access='stream',form='unformatted',status='old')
 read(fi) nspecies,nq
 allocate(species(nspecies),qs(nq),qo(nq),xyz(3,nq),result(4,nq),phi(maxnorbs),grad(3,maxnorbs))
 do is=1,nspecies
   read(fi) nr,no
   species(is)%n_orbnl=nr;species(is)%norbs=no;species(is)%z=26
   allocate(species(is)%orbnl(nr))
   do ir=1,nr
     read(fi) species(is)%orbnl(ir)%n,species(is)%orbnl(ir)%cutoff,species(is)%orbnl(ir)%delta
     i=species(is)%orbnl(ir)%n
     allocate(species(is)%orbnl(ir)%f(i),species(is)%orbnl(ir)%d2(i))
     read(fi) species(is)%orbnl(ir)%f,species(is)%orbnl(ir)%d2
   enddo
   read(fi) species(is)%orb_l(1:no),species(is)%orb_m(1:no),species(is)%orb_index(1:no)
 enddo
 read(fi) qs,qo,xyz
 close(fi)
 do i=1,nq
   is=qs(i);r2cut=0._dp
   do j=1,species(is)%norbs
     r2cut=max(r2cut,species(is)%orbnl(species(is)%orb_index(j))%cutoff**2)
   enddo
   r2=xyz(1,i)**2+xyz(2,i)**2+xyz(3,i)**2
   phi=0._dp;grad=0._dp
   if(r2<r2cut) call all_phi(is,1,xyz(:,i),maxnorbs,nphi,phi,grad)
   result(1,i)=phi(qo(i));result(2:4,i)=grad(:,qo(i))
 enddo
 open(newunit=fo,file=trim(output),access='stream',form='unformatted',status='replace')
 write(fo) pi,result
 close(fo)
end program
'''


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def build_reference(build,tmp):
    src=build.resolve()/'Src';wrapper=tmp/'reference.f90';wrapper.write_text(WRAPPER);exe=tmp/'reference'
    link=shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())
    libs=link[next(i for i,v in enumerate(link) if Path(v).name=='libsiesta.a'):]
    flags=(src/'CMakeFiles/SIESTA.libsiesta.dir/flags.make').read_text()
    includes=next(v.split('=',1)[1] for v in flags.splitlines() if v.startswith('Fortran_INCLUDES ='))
    subprocess.run([link[0],'-O2',*shlex.split(includes),str(wrapper),'-o',str(exe),*libs],cwd=src,check=True)
    return exe


def fixtures():
    rng=np.random.default_rng(67125)
    for lmax in range(6):
        species=[]
        for variant in range(2):
            radials=[];orbitals=[]
            for l in range(lmax+1):
                count=81;delta=.075;cutoff=4.7-.35*l-.1*variant
                x=np.arange(count)*delta
                f=np.exp(-(.2+.05*l)*x*x)*(1.+.03*variant*x)
                d2=rng.normal(scale=.2,size=count)
                # The oracle covers the literal interpolation, not how atomic
                # initialization generates these input second derivatives.
                radials.append(dict(n=count,delta=delta,cutoff=cutoff,f=f,d2=d2))
                orbitals.extend((l,m,l) for m in range(-l,l+1))
            # One extra s radial with a different cutoff and, for variant 1,
            # no radial values. It still participates in upstream cutoff logic.
            radials.append(dict(n=0 if variant else 2,delta=.6,cutoff=.55,
                                f=np.array([]) if variant else np.array([.3,.7]),
                                d2=np.array([]) if variant else np.array([-.2,.1])))
            orbitals.append((0,0,len(radials)-1))
            species.append((radials,np.array(orbitals,dtype=np.int32)))
        qs=[];qo=[];xyz=[]
        for s,(radials,orbs) in enumerate(species):
            points=[np.zeros(3),np.array([0.,0.,1.]),np.array([0.,0.,-1.]),
                    np.array([1.e-16,0.,1.]),np.array([1.e-15,-1.e-15,2.e-15]),
                    np.array([1.e-12,0.,0.]),np.array([.3,.4,.5])]
            for radial in radials:
                for r in (radial['delta'],np.nextafter(radial['delta'],0),np.nextafter(radial['delta'],np.inf),
                          radial['cutoff'],np.nextafter(radial['cutoff'],0),np.nextafter(radial['cutoff'],np.inf)):
                    points.append(np.array([r,0.,0.]))
            points.extend(rng.normal(size=(23,3))*1.3)
            points.append(np.array([8.,9.,10.]))
            for point in points:
                for orb in range(len(orbs)):
                    qs.append(s);qo.append(orb);xyz.append(point)
        yield 'shells_l'+str(lmax),species,np.array(qs,np.int32),np.array(qo,np.int32),np.array(xyz,np.float64)


def pack(species,qs,qo,xyz):
    chunks=[np.array([len(species),len(qs)],np.int32).tobytes()]
    for radials,orbs in species:
        chunks.append(np.array([len(radials),len(orbs)],np.int32).tobytes())
        for r in radials:
            chunks.extend((np.array([r['n']],np.int32).tobytes(),np.array([r['cutoff'],r['delta']],np.float64).tobytes(),r['f'].tobytes(),r['d2'].tobytes()))
        chunks.extend((orbs[:,0].tobytes(),orbs[:,1].tobytes(),(orbs[:,2]+1).tobytes()))
    chunks.extend(((qs+1).tobytes(),(qo+1).tobytes(),xyz.tobytes()))
    return b''.join(chunks)


def flatten(species):
    # Species [first orbital, count]; radial descriptors per expanded orbital
    # [n, sample offset, l, m] and [cutoff,delta], preserving upstream order.
    sp=[];ints=[];reals=[];values=[];second=[];offset=0;begin=0;radii=[]
    for radials,orbs in species:
        sp.append((begin,len(orbs)));begin+=len(orbs);local=[]
        for r in radials:
            local.append(offset);offset+=r['n'];values.extend(r['f']);second.extend(r['d2'])
        for l,m,ir in orbs:
            r=radials[ir];ints.append((r['n'],local[ir],l,m));reals.append((r['cutoff'],r['delta']))
        radii.append(max(r['cutoff']**2 for r in radials))
    return dict(species=np.array(sp,np.int32),orbitals=np.array(ints,np.int32),radial_parameters=np.array(reals,np.float64),
                values=np.array(values,np.float64),second=np.array(second,np.float64),radius2=np.array(radii,np.float64))


def run(upstream,build,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);checks=[];arrays={}
    with tempfile.TemporaryDirectory(prefix='remag-orbital-basis-') as tmp:
        tmp=Path(tmp);exe=build_reference(build,tmp)
        for name,species,qs,qo,xyz in fixtures():
            inp=tmp/'input.bin';out=tmp/'output.bin';inp.write_bytes(pack(species,qs,qo,xyz))
            p=subprocess.run([str(exe),str(inp),str(out)],cwd=tmp,capture_output=True,text=True)
            if p.returncode:raise RuntimeError(p.stdout+p.stderr)
            raw=np.fromfile(out,np.float64);assert raw[0]==np.pi and raw.size==1+4*len(qs)
            expected=raw[1:].reshape(-1,4);assert np.isfinite(expected).all()
            data=dict(flatten(species),query_species=qs,query_orbital=qo,xyz=xyz,expected=expected)
            for k,v in data.items():arrays[name+'_'+k]=v
            checks.append(dict(fixture=name,queries=len(qs),species=len(species),orbitals=len(data['orbitals']),samples=len(data['values']),input_sha256=sha(inp),passed=True))
    archive=output.with_suffix('.npz');np.savez_compressed(archive,**arrays);wrapper=output.with_suffix('.f90');wrapper.write_text(WRAPPER)
    report=dict(scope=__doc__,passed=True,gpu_validation=None,checks=checks,pi=float(np.pi),
                source_sha256={n:sha(upstream/'Src'/n) for n in ('atmfuncs.f','radial.f','spher_harm.f','interpolation.f90')},
                reference_library_sha256=sha(build/'Src/libsiesta.a'),script_sha256=sha(Path(__file__)),
                arrays_file=archive.name,arrays_sha256=sha(archive),wrapper_file=wrapper.name,wrapper_sha256=sha(wrapper))
    output.write_text(json.dumps(report,indent=2)+'\n');print('Pristine orbital oracle:',len(checks),'fixtures,',sum(v['queries'] for v in checks),'queries')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('upstream','build','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.upstream.resolve(),a.build.resolve(),a.output)
