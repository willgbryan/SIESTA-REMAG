#!/usr/bin/env python3
"""Build real pinned S/T two-center tables and retain pristine get_matel outputs.

This is a CPU oracle for the next GPU primitive, not production integration.
The unmodified registry, Fourier/radial table builder, spherical harmonics and
spline evaluator all come from the pinned libsiesta. Table generation is setup.
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
 use radial, only: rad_func
 use m_matel_registry, only: register_in_rf_pool,evaluate
 use matel_ylm_m, only: spher_harm_t
 use matel_table_m, only: matel_t
 use matel_params_m, only: NRTAB,RMAX
 use interpolation, only: spline
 implicit none
 type(spher_harm_t),target :: spha
 type(matel_t) table
 type(rad_func),pointer :: radial
 integer fi,fo,nreg,nq,mode,lo1,hi1,lo2,hi2,l,m,id,i,j
 integer,allocatable :: pairs(:,:)
 real(dp),allocatable :: cutoffs(:),xyz(:,:),result(:,:)
 character(2048) input,output
 call get_command_argument(1,input);call get_command_argument(2,output)
 open(newunit=fi,file=trim(input),access='stream',form='unformatted',status='old')
 read(fi) nreg,nq,mode,lo1,hi1,lo2,hi2
 allocate(cutoffs(nreg),pairs(2,nq),xyz(3,nq),result(4,nq))
 do i=1,nreg
   allocate(radial)
   read(fi) l,m,radial%n,radial%cutoff,radial%delta
   cutoffs(i)=radial%cutoff
   allocate(radial%f(radial%n),radial%d2(radial%n));read(fi) radial%f
   call spline(radial%delta,radial%f,radial%n,0._dp,0._dp,radial%d2)
   call register_in_rf_pool(radial,l,m,'orb',[i],id)
   if(id/=i) error stop 'Registry mismatch'
 enddo
 read(fi) pairs,xyz;close(fi)
 call spha%compute_spha(1,nreg,evaluate,0)
 call table%init(mode,spha,lo1,hi1,spha,lo2,hi2)
 do i=1,nq
   call table%get_matel(pairs(1,i),pairs(2,i),xyz(:,i),result(1,i),result(2:4,i))
 enddo
 open(newunit=fo,file=trim(output),access='stream',form='unformatted',status='replace')
 write(fo) NRTAB,table%NFFR,table%NFFY,table%MAT_ROWS,table%MAT_COLS, &
   table%BASE_SPHA1,table%BASE_SPHA2,nreg,nq
 write(fo) RMAX/NRTAB,cutoffs
 write(fo) spha%ind(1:nreg+1),table%INDFFY(1:table%MAT_ROWS*table%MAT_COLS+1), &
   table%ILMFF(1:table%NFFY),table%INDFFR(1:table%NFFY)
 write(fo) table%FFR(0:NRTAB,1:2,1:table%NFFR),table%FFY(1,1:table%NFFY),result
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
    rng=np.random.default_rng(602611)
    families=(('s',[(0,0)],(1,1,1,1)),
        ('spd',[(l,m) for l in range(3) for m in range(-l,l+1)],(1,9,1,9)),
        ('high',[(l,m) for l in (3,4,5) for m in (-l,-1,0,l)],(1,12,1,12)),
        ('subset',[(0,0),(1,0),(2,-2),(3,1),(4,0),(5,-5),(2,2),(1,-1)],(3,6,2,7)))
    for name,lm,bounds in families:
        functions=[]
        for i,(l,m) in enumerate(lm):
            n=201;delta=.03;cutoff=4.5+.15*(i%4);r=np.arange(n)*delta
            f=np.exp(-(.6+.03*(i%3))*r*r)*np.maximum(0,1-r/cutoff)**4
            functions.append(dict(l=l,m=m,n=n,cutoff=cutoff,delta=delta,f=f))
        pairs=[];xyz=[]
        for a in range(bounds[0]-1,bounds[1]):
            for b in range(bounds[2]-1,bounds[3]):
                cut=functions[a]['cutoff']+functions[b]['cutoff'];dr=np.pi/100
                points=[[0.,0.,0.],[1e-20,-1e-20,-1e-20],[1e-12,0.,0.],[0.,0.,-1e-12],
                    [0.,0.,1.],[1e-16,-1e-16,-1.],[.3,.4,.5],[-.3,-.4,-.5]]
                for radius in (1e-12,dr,31*dr,cut):
                    for x in (np.nextafter(radius,-np.inf),radius,np.nextafter(radius,np.inf)):points.append([x,0.,0.])
                points.extend(rng.normal(size=(11,3))*2.)
                for axis in range(3):
                    for sign in (-1,1):
                        point=np.array([.3,.4,.5]);point[axis]+=sign*1.e-5;points.append(point)
                pairs.extend([[a+1,b+1]]*len(points));xyz.extend(points)
        for mode in (1,2):yield name+('_S' if mode==1 else '_T'),functions,bounds,mode,np.array(pairs,np.int32),np.array(xyz,np.float64)

def pack(functions,bounds,mode,pairs,xyz):
    chunks=[np.array([len(functions),len(pairs),mode,*bounds],np.int32).tobytes()]
    for f in functions:
        chunks.extend((np.array([f['l'],f['m'],f['n']],np.int32).tobytes(),np.array([f['cutoff'],f['delta']],np.float64).tobytes(),f['f'].tobytes()))
    return b''.join(chunks+[pairs.tobytes(),xyz.tobytes()])

def unpack(raw):
    pos=0
    def read(dtype,count):
        nonlocal pos
        a=np.frombuffer(raw,dtype,count=count,offset=pos).copy();pos+=a.nbytes;return a
    config=read(np.int32,9);nr,nf,ny,rows,cols,base1,base2,nreg,nq=map(int,config)
    data=dict(config=config,delta=read(np.float64,1),cutoffs=read(np.float64,nreg),
        ind=read(np.int32,nreg+1),indffy=read(np.int32,rows*cols+1),ilm=read(np.int32,ny),indffr=read(np.int32,ny),
        radial=read(np.float64,(nr+1)*2*nf).reshape(nf,2,nr+1),coefficients=read(np.float64,ny),expected=read(np.float64,4*nq).reshape(nq,4))
    if pos!=len(raw):raise ValueError('Oracle output length differs')
    return data

def run(upstream,build,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);checks=[];arrays={}
    with tempfile.TemporaryDirectory(prefix='remag-two-center-') as temporary:
        tmp=Path(temporary);exe=build_reference(build,tmp)
        for label,functions,bounds,mode,pairs,xyz in fixtures():
            inp=tmp/'input.bin';out=tmp/'output.bin';inp.write_bytes(pack(functions,bounds,mode,pairs,xyz))
            p=subprocess.run([str(exe),str(inp),str(out)],cwd=tmp,capture_output=True,text=True,timeout=180)
            (output.parent/(label+'.log')).write_text(p.stdout+p.stderr)
            if p.returncode:raise RuntimeError(p.stdout+p.stderr)
            data=unpack(out.read_bytes());data.update(bounds=np.array(bounds,np.int32),pairs=pairs,xyz=xyz)
            cut=data['cutoffs'][pairs[:,0]-1]+data['cutoffs'][pairs[:,1]-1]
            outside=np.linalg.norm(xyz,axis=1)>cut
            if not np.isfinite(data['expected']).all() or not np.all(data['expected'][outside]==0):raise ValueError('Invalid oracle/cutoff result')
            for k,v in data.items():arrays[label+'_'+k]=v
            checks.append(dict(fixture=label,mode=mode,queries=len(pairs),input_sha256=sha(inp),output_sha256=sha(out),
                radial_functions=int(data['config'][1]),angular_terms=int(data['config'][2]),max_l=int(np.sqrt(data['ilm'].max()-1)),passed=True))
            print(label,len(pairs),'queries, radial',data['config'][1],'angular',data['config'][2],flush=True)
    archive=output.with_suffix('.npz');np.savez_compressed(archive,**arrays);wrapper=output.with_suffix('.f90');wrapper.write_text(WRAPPER)
    sources=('matel.F90','matel_table.F90','matel_ylm.F90','matel_registry.F90','matel_params.f90','spher_harm.f','radial.f','radfft.f','interpolation.f90')
    report=dict(scope=__doc__,passed=True,gpu_validation=None,checks=checks,source_sha256={n:sha(upstream/'Src'/n) for n in sources},
        reference_library_sha256=sha(build/'Src/libsiesta.a'),script_sha256=sha(Path(__file__)),arrays_file=archive.name,arrays_sha256=sha(archive),wrapper_file=wrapper.name,wrapper_sha256=sha(wrapper))
    output.write_text(json.dumps(report,indent=2)+'\n');print('Pristine two-center oracle:',len(checks),'fixtures,',sum(c['queries'] for c in checks),'queries')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('upstream','build','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.upstream.resolve(),a.build.resolve(),a.output)
