#!/usr/bin/env python3
"""Retain pristine SCF energy, charge and residual oracles for device consumers.

This links unmodified compute_energies, dm_charge and compute_max_diff with
explicit sparse state. It does not run full SCF, construct H or validate CUDA.
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
 use m_spin, only: spin
 use sparse_matrices, only: H,Hold,Dscf,Dold,Escf,Eold,S,maxnh
 use siesta_options, only: mixH,mix_charge,Temp
 use m_ts_global_vars, only: TSrun
 use m_energies
 use m_compute_energies, only: compute_energies
 use m_compute_max_diff, only: compute_max_diff
 use dm_charge_m, only: dm_charge
 implicit none
 integer :: fi,fo,n,nh,eh,rows
 integer,allocatable :: num(:),ptr(:)
 real(dp) :: seeds(12),result(13)
 character(2048) :: input,output
 call get_command_argument(1,input);call get_command_argument(2,output)
 open(newunit=fi,file=trim(input),access='stream',form='unformatted',status='old')
 read(fi) n,nh,rows
 spin%H=nh;spin%DM=nh;spin%Grid=min(nh,4);spin%EDM=min(nh,4);spin%spinor=min(nh,2)
 spin%none=nh==1;spin%Col=nh==2;spin%NCol=nh==4;spin%SO=nh==8
 eh=spin%EDM;maxnh=n
 allocate(H(n,nh),Hold(n,nh),Dscf(n,nh),Dold(n,nh),Escf(n,eh),Eold(n,eh),S(n),num(rows),ptr(rows))
 read(fi) num,ptr,seeds,H,Hold,Dscf,Dold,Escf,Eold,S
 close(fi)
 call init_Energies()
 Ena=seeds(1);Ekin=seeds(2);Enl=seeds(3);Eso=seeds(4);Eions=seeds(5)
 Enascf=seeds(6);Enaatm=seeds(7);DUscf=seeds(8);Exc=seeds(9)
 Entropy=seeds(10);Temp=seeds(11);Emad=seeds(12)
 mixH=.true.;mix_charge=.false.;TSrun=.false.
 call compute_energies(1)
 result=0
 result(1:6)=(/Ebs,DEharr,Etot,Eharrs,FreeE,DEna/)
 call compute_max_diff(Dold,Dscf,result(7))
 call compute_max_diff(Hold,H,result(8))
 call compute_max_diff(Eold,Escf,result(9))
 call dm_charge(spin,rows,num,ptr,S,Dscf,result(10:9+spin%Grid))
 open(newunit=fo,file=trim(output),access='stream',form='unformatted',status='replace')
 write(fo) result
 close(fo)
end program
'''


def build_reference(build,tmp):
    src=build.resolve()/'Src';wrapper=tmp/'reference.f90';wrapper.write_text(WRAPPER);exe=tmp/'reference'
    link=shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())
    libraries=link[next(i for i,s in enumerate(link) if Path(s).name=='libsiesta.a'):]
    flags=(src/'CMakeFiles/SIESTA.libsiesta.dir/flags.make').read_text()
    includes=next(line.split('=',1)[1].strip() for line in flags.splitlines() if line.startswith('Fortran_INCLUDES ='))
    subprocess.run([link[0],'-O2',*shlex.split(includes),'-I'+str(src),str(wrapper),'-o',str(exe),*libraries],cwd=src,check=True)
    return exe


def evaluate(exe,tmp,num,ptr,seeds,h,hold,d,old,e,eold,overlap):
    n,nh=h.shape;inp=tmp/'input';out=tmp/'output'
    inp.write_bytes(np.array([n,nh,len(num)],dtype=np.int32).tobytes()+num.tobytes()+ptr.tobytes()+
        seeds.tobytes()+b''.join(a.tobytes(order='F') for a in (h,hold,d,old,e,eold,overlap)))
    completed=subprocess.run([str(exe),str(inp),str(out)],cwd=tmp,capture_output=True,text=True)
    if completed.returncode:
        print(completed.stdout,completed.stderr,flush=True);completed.check_returncode()
    result=np.fromfile(out,dtype=np.float64);assert result.shape==(13,) and np.isfinite(result).all()
    return result,hashlib.sha256(inp.read_bytes()).hexdigest()


def run(upstream,build,output):
    upstream=upstream.resolve();build=build.resolve();src=build/'Src'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(gpu_validation=None,fixture_script_sha256=sha(Path(__file__)),
                reference='Pristine compute_energies (Hamiltonian mixing), dm_charge and compute_max_diff',
                source_sha256={name:sha(upstream/'Src'/name) for name in
                    ('compute_energies.F90','m_energies.F90','dm_charge.F90','compute_max_diff.F90','m_spin.F90')},
                reference_library_sha256=sha(src/'libsiesta.a'),checks=[])
    arrays={};rng=np.random.default_rng(59103)
    with tempfile.TemporaryDirectory(prefix='remag-scf-scalars-') as tmp:
        tmp=Path(tmp);exe=build_reference(build,tmp)
        for nh in (1,2,4,8):
            for n in (19,128,513):
                label=f'spin{nh}_capacity{n}';eh=min(nh,4)
                num=np.array([2,3,1,4],dtype=np.int32);ptr=np.array([0,4,9,n-7],dtype=np.int32)
                active=np.concatenate([np.arange(p,p+c) for p,c in zip(ptr,num)])
                # Nonzero unused capacity distinguishes dense-capacity energy /
                # residual loops from sparse-topology-only charge reduction.
                h=np.asfortranarray(rng.normal(size=(n,nh)))
                hold=np.asfortranarray(h+rng.normal(size=h.shape)*.01)
                d=np.asfortranarray(rng.normal(size=(n,nh))*.1)
                old=np.asfortranarray(d+rng.normal(size=d.shape)*.003)
                e=np.asfortranarray(rng.normal(size=(n,eh))*.2)
                eold=np.asfortranarray(e+rng.normal(size=e.shape)*.007)
                overlap=rng.normal(size=n)
                seeds=np.array([.7,1.3,-.4,.02,2.1,-.8,-.9,.13,-1.7,.025,.01,.12])
                result,input_hash=evaluate(exe,tmp,num,ptr,seeds,h,hold,d,old,e,eold,overlap)
                weights=np.array([1.,1.,2.,2.]) if nh==4 else np.ones(nh)
                band=float(np.sum(h*d*weights));harris=float(np.sum(h*(d-old)*weights))
                ena,kinetic,nonlocal_energy,soc,ions,enascf,enaatm,hartree,xc,entropy,temp,madelung=seeds
                total=ena+kinetic+nonlocal_energy+soc-ions+(enascf-enaatm)+hartree+xc+madelung
                expected=np.zeros(13)
                expected[:9]=[band,harris,total,total+harris,total-temp*entropy,enascf-enaatm,
                              np.max(np.abs(d-old)),np.max(np.abs(h-hold)),np.max(np.abs(e-eold))]
                charge=d[active,:min(nh,4)].copy()
                if nh==8: charge[:,2:]=.5*(d[active,2:4]+d[active,6:8])
                expected[9:9+min(nh,4)]=np.sum(charge*overlap[active,None],axis=0)
                np.testing.assert_allclose(result,expected,atol=1.e-11,rtol=1.e-12)
                for name,a in (('h',h),('hold',hold),('density',d),('old_density',old),('energy_density',e),
                               ('old_energy_density',eold),('overlap',overlap),('num',num),('ptr',ptr),('energy_seeds',seeds),('result',result)):
                    arrays[label+'_'+name]=a
                report['checks'].append(dict(fixture=label,capacity=n,spin_components=nh,used_entries=len(active),
                    input_sha256=input_hash,max_absolute_error=float(np.max(np.abs(result-expected))),passed=True))
    output.parent.mkdir(parents=True,exist_ok=True);artifact=output.with_suffix('.npz')
    np.savez_compressed(artifact,**arrays)
    report.update(arrays_file=artifact.name,arrays_sha256=sha(artifact),passed=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"SCF scalar reference scenarios: {len(report['checks'])}; passed=True; GPU validation=None")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,default=Path('upstream/siesta'))
    p.add_argument('--build',type=Path,default=Path('build/upstream'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.upstream,args.build,args.output)
