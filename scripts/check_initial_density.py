#!/usr/bin/env python3
"""Pinned first-geometry normalization oracle, using actual normalize_dm."""
from pathlib import Path
import argparse,hashlib,json,shlex,subprocess
import numpy as np

WRAPPER='''program reference
 use precision, only: dp
 use m_spin, only: spin
 use sparse_matrices, only: Dscf,Escf,S,maxnh
 use siesta_options, only: mix_scf_first,normalize_dm_during_scf
 use atomlist, only: qtot
 use m_normalize_dm, only: normalize_dm,dm_norm
 use parallel, only: IOnode
 implicit none
 integer :: nh,mix,fi,fo
 real(dp) :: charge
 character(2048) :: input,output
 call get_command_argument(1,input);call get_command_argument(2,output)
 open(newunit=fi,file=trim(input),access='stream',form='unformatted',status='old')
 read(fi) maxnh,nh,mix,qtot
 spin%DM=nh;spin%EDM=min(nh,4);spin%spinor=min(nh,2)
 allocate(Dscf(maxnh,nh),Escf(maxnh,spin%EDM),S(maxnh))
 read(fi) S,Dscf;close(fi)
 Escf=0._dp;mix_scf_first=mix==1;normalize_dm_during_scf=.true.;IOnode=.true.
 call dm_norm(charge)
 call normalize_dm(first=.true.)
 Escf=0._dp
 open(newunit=fo,file=trim(output),access='stream',form='unformatted',status='replace')
 write(fo) charge,qtot/charge,merge(1,0,mix_scf_first),Dscf,Escf
 close(fo)
end program
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build_reference(build,folder):
    src=build.resolve()/'Src';link=shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text());libs=link[next(i for i,x in enumerate(link) if Path(x).name=='libsiesta.a'):]
    flags=(src/'CMakeFiles/SIESTA.libsiesta.dir/flags.make').read_text();includes=shlex.split(next(line.split('=',1)[1] for line in flags.splitlines() if line.startswith('Fortran_INCLUDES =')))
    code=folder/'reference.f90';code.write_text(WRAPPER);exe=folder/'reference'
    subprocess.run([link[0],'-O0','-fcheck=all',*includes,'-I'+str(src),str(code),'-o',str(exe),*libs],cwd=src,check=True)
    return exe

def evaluate(exe,folder,label,overlap,density,charge,mix_first):
    capacity,spins=density.shape;inp=folder/(label+'.input');out=folder/(label+'.output');log=folder/(label+'.log')
    inp.write_bytes(np.array([capacity,spins,mix_first],dtype='i4').tobytes()+np.float64(charge).tobytes()+overlap.tobytes()+density.tobytes(order='F'))
    with log.open('w') as stream:subprocess.run([str(exe),str(inp),str(out)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    data=out.read_bytes();q,scale=np.frombuffer(data[:16],dtype='f8');mix=np.frombuffer(data[16:20],dtype='i4')[0]
    values=np.frombuffer(data[20:],dtype='f8');assert len(values)==capacity*(spins+min(spins,4))
    return float(q),float(scale),int(mix),values[:capacity*spins].reshape(capacity,spins,order='F'),values[capacity*spins:].reshape(capacity,min(spins,4),order='F')

def run(build,output):
    folder=output.resolve().parent;folder.mkdir(parents=True,exist_ok=True);exe=build_reference(build,folder);rng=np.random.default_rng(84118);checks=[];arrays={}
    for spins in (1,2,4,8):
        for capacity in (1,5,257,4097):
            for mismatch in (0.,.5e-4,2.e-4,.02,-.02):
                for mix in (0,1):
                    overlap=rng.uniform(.01,1.,size=capacity);density=rng.normal(0.,.1,size=(capacity,spins));density[:,:min(spins,2)]+=2.
                    q=sum(float(density[i,s]*overlap[i]) for s in range(min(spins,2)) for i in range(capacity));target=q+mismatch
                    label=f's{spins}-cap{capacity}-delta{mismatch}-mix{mix}'
                    actual_q,scale,effective,d,e=evaluate(exe,folder,label,overlap,density,target,mix)
                    np.testing.assert_allclose(actual_q,q,atol=1e-10,rtol=1e-12)
                    assert scale==target/actual_q
                    assert effective==int(bool(mix) and abs(actual_q-target)<=1e-4)
                    np.testing.assert_array_equal(d,density*scale);np.testing.assert_array_equal(e,0.)
                    checks.append(dict(name=label,capacity=capacity,spins=spins,mix_first=mix,target=target,charge=actual_q,scale=scale,effective_mix=effective,note=abs(actual_q/target-1)>1e-3,passed=True))
                    for name,v in dict(overlap=overlap,density=density,normalized=d,energy=e).items():arrays[label+'__'+name]=v
    npz=output.with_suffix('.npz');np.savez_compressed(npz,**arrays)
    paths=[Path(__file__),folder/'reference.f90',npz,build/'Src/libsiesta.a',Path('upstream/siesta/Src/normalize_dm.F90')]
    report=dict(passed=True,scope='Actual pristine normalize_dm(first=.true.), then state_init zero EDM; initial normalization is distinct from in-SCF tolerance.',gpu_validated=False,checks=checks,evidence_sha256={str(p.resolve()):sha(p) for p in paths})
    output.write_text(json.dumps(report,indent=2)+'\n');print('Pinned initial density checks:',len(checks),'passed')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=Path('build/upstream'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.build,a.output)
