#!/usr/bin/env python3
"""Pristine chcore_sub/phiatm(io=0) oracle, including outer mesh radius selection.

Links unmodified libsiesta. Retains all radial tables, coordinates, full outputs
and an original reord inverse map for later device-geometry qualification.
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
 use precision, only: dp,grid_p
 use atm_types, only: species,nspecies
 use radial, only: rad_func
 use atmfuncs, only: chcore_sub,phiatm,rcore,rcut
 implicit none
 integer fi,fo,nq,kind,mesh_filter,shape(3),nsm,is,ir,i,j,n
 integer,allocatable :: qs(:),active(:),inverse(:)
 real(dp),allocatable :: xyz(:,:),result(:,:)
 real(grid_p),allocatable :: map(:)
 type(rad_func),pointer :: radial
 real(dp) radius,cutoff
 character(2048) input,output
 call get_command_argument(1,input);call get_command_argument(2,output)
 open(newunit=fi,file=trim(input),access='stream',form='unformatted',status='old')
 read(fi) nspecies,nq,kind,mesh_filter,shape,nsm
 if(grid_p/=dp.or.any(modulo(shape,nsm)/=0)) error stop 'bad oracle mesh'
 allocate(species(nspecies),qs(nq),active(nq),xyz(3,nq),result(4,nq))
 do is=1,nspecies
   read(fi) species(is)%z
   do ir=1,2
     if(ir==1) then
       radial=>species(is)%core
     else
       radial=>species(is)%vna
     endif
     read(fi) radial%n,radial%cutoff,radial%delta
     allocate(radial%f(radial%n),radial%d2(radial%n))
     read(fi) radial%f,radial%d2
   enddo
 enddo
 read(fi) qs,xyz
 close(fi)
 result=0._dp;active=1
 do i=1,nq
   is=qs(i)
   if(mesh_filter==1) then
     radius=sqrt(dot_product(xyz(:,i),xyz(:,i)))
     if(kind==1) then
       cutoff=rcore(is)
       if(.not.(cutoff>1.e-12_dp.and.radius<cutoff)) active(i)=0
     else
       cutoff=rcut(is,0)
       if(.not.(radius<cutoff)) active(i)=0
     endif
   endif
   if(active(i)==0) cycle
   if(kind==1) then
     call chcore_sub(is,xyz(:,i),result(1,i),result(2:4,i))
   else
     call phiatm(is,0,xyz(:,i),result(1,i),result(2:4,i))
   endif
 enddo
 n=product(shape);allocate(map(n),inverse(n))
 do i=1,n
   map(i)=i
 enddo
 call reord(map,map,shape/nsm,nsm,1)
 do i=1,n
   inverse(nint(map(i)))=i
 enddo
 open(newunit=fo,file=trim(output),access='stream',form='unformatted',status='replace')
 write(fo) result,active,inverse
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


def species_fixtures():
    rng=np.random.default_rng(92645)
    def radial(n,delta,cutoff):
        x=np.arange(n)*delta
        return dict(n=n,delta=delta,cutoff=cutoff,f=np.exp(-.21*x*x)*(1.+.04*x),d2=rng.normal(scale=.2,size=n))
    smooth=[(26,[radial(81,.05,3.4),radial(97,.075,5.1)]),(78,[radial(93,.0625,4.2),radial(121,.0625,6.4)])]
    ghost=( -26,[dict(n=1,delta=np.nan,cutoff=3.,f=np.array([np.nan]),d2=np.array([np.nan])) for _ in range(2)])
    empty=(26,[dict(n=0,delta=np.nan,cutoff=2.,f=np.array([]),d2=np.array([])) for _ in range(2)])
    linear=(26,[dict(n=2,delta=2.,cutoff=1.4,f=np.array([.3,.7]),d2=np.array([-.2,.1])) for _ in range(2)])
    yield 'smooth',smooth+[ghost]
    yield 'empty',[empty,ghost]
    yield 'no_samples',[empty]
    yield 'two_samples',[linear,ghost]
    yield 'endpoint',[(26,[dict(n=2,delta=.6,cutoff=.6000002,f=np.array([.3,.7]),d2=np.array([-.2,.1])) for _ in range(2)])]
    yield 'tiny_cutoffs',[(26,[radial(2,4.e-12,c),radial(2,4.e-12,c)]) for c in (0.,1.e-12,np.nextafter(1.e-12,np.inf),2.e-12)]
    yield 'stride',smooth+[ghost,empty]


def fixtures():
    rng=np.random.default_rng(256901)
    for name,species in species_fixtures():
        qs=[];xyz=[]
        for ispec,(_,radials) in enumerate(species):
            points=[np.zeros(3),np.array([1.e-20,0.,0.]),np.array([0.,0.,-1.e-20]),
                np.array([1.e-30,-1.e-30,1.e-30]),np.array([.3,.4,.5]),np.array([0.,0.,1.])]
            for radial in radials:
                for base in (radial['cutoff'],radial['cutoff']-1.e-12,radial['delta']):
                    if np.isfinite(base):
                        for x in (base,np.nextafter(base,-np.inf),np.nextafter(base,np.inf)):
                            points.append(np.array([x,0.,0.]))
            points.extend(rng.normal(size=(43,3))*1.4)
            if name=='tiny_cutoffs':points.extend(rng.normal(size=(43,3))*1.e-12)
            for point in points:qs.append(ispec);xyz.append(point)
        qs=np.array(qs,np.int32);xyz=np.array(xyz,np.float64)
        if name=='stride':
            count=131073
            qs=np.resize(qs,count);xyz=np.resize(xyz,(count,3));qs[-1]=0;xyz[-1]=[.7,.8,.9]
        yield name,species,qs,xyz


def flatten(species):
    z=[];index=[];parameters=[];values=[];second=[]
    for atomic_number,radials in species:
        z.append(atomic_number)
        for r in radials:
            index.append((r['n'],len(values)));parameters.append((r['cutoff'],r['delta']));values.extend(r['f']);second.extend(r['d2'])
    return dict(atomic_number=np.array(z,np.int32),radial_index=np.array(index,np.int32),
        radial_parameters=np.array(parameters,np.float64),values=np.array(values,np.float64),second=np.array(second,np.float64))


def pack(species,qs,xyz,kind,mesh_filter,shape=(1,1,1),nsm=1):
    chunks=[np.array([len(species),len(qs),kind,mesh_filter,*shape,nsm],np.int32).tobytes()]
    for z,radials in species:
        chunks.append(np.array([z],np.int32).tobytes())
        for r in radials:
            chunks.extend((np.array([r['n']],np.int32).tobytes(),np.array([r['cutoff'],r['delta']],np.float64).tobytes(),r['f'].tobytes(),r['d2'].tobytes()))
    chunks.extend(((qs+1).tobytes(),xyz.tobytes()));return b''.join(chunks)


def oracle(executable,tmp,species,qs,xyz,kind,mesh_filter,shape=(1,1,1),nsm=1):
    inp=tmp/'input.bin';out=tmp/'output.bin';inp.write_bytes(pack(species,qs,xyz,kind,mesh_filter,shape,nsm))
    done=subprocess.run([str(executable),str(inp),str(out)],cwd=tmp,capture_output=True,text=True)
    if done.returncode:raise RuntimeError(done.stdout+done.stderr)
    raw=out.read_bytes();n=len(qs);npnt=int(np.prod(shape));assert len(raw)==36*n+4*npnt
    value=np.frombuffer(raw,np.float64,count=4*n).reshape(n,4).copy()
    active=np.frombuffer(raw,np.int32,count=n,offset=32*n).copy();inverse=np.frombuffer(raw,np.int32,count=npnt,offset=36*n).copy()
    assert np.isfinite(value).all() and np.isin(active,[0,1]).all() and np.array_equal(np.sort(inverse),np.arange(1,npnt+1))
    return value,active,inverse,sha(inp)


def run(upstream,build,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);checks=[];arrays={}
    with tempfile.TemporaryDirectory(prefix='remag-atomic-basis-') as temporary:
        tmp=Path(temporary);exe=build_reference(build,tmp)
        for name,species,qs,xyz in fixtures():
            for kind in (1,2):
                for mesh_filter in (0,1):
                    label=f'{name}_kind{kind}_mesh{mesh_filter}'
                    expected,active,inverse,input_hash=oracle(exe,tmp,species,qs,xyz,kind,mesh_filter)
                    for key,value in dict(flatten(species),query_species=qs,xyz=xyz,expected=expected,active=active,inverse=inverse).items():arrays[label+'_'+key]=value
                    checks.append(dict(fixture=label,source=name,kind=kind,mesh_filter=mesh_filter,queries=len(qs),input_sha256=input_hash,passed=True))
    archive=output.with_suffix('.npz');np.savez_compressed(archive,**arrays);wrapper=output.with_suffix('.f90');wrapper.write_text(WRAPPER)
    report=dict(scope=__doc__,passed=True,gpu_validation=None,checks=checks,
        source_sha256={name:sha(upstream/'Src'/name) for name in ('atmfuncs.f','radial.f','interpolation.f90','meshsubs.F','reord.f')},
        reference_library_sha256=sha(build/'Src/libsiesta.a'),script_sha256=sha(Path(__file__)),
        arrays_file=archive.name,arrays_sha256=sha(archive),wrapper_file=wrapper.name,wrapper_sha256=sha(wrapper))
    output.write_text(json.dumps(report,indent=2)+'\n');print('Pristine atomic radial oracle:',len(checks),'fixtures,',sum(v['queries'] for v in checks),'queries')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','build','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.upstream.resolve(),a.build.resolve(),a.output)
