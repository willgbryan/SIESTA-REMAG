#!/usr/bin/env python3
"""Check the shared CPU/device PBE algebra against pristine GridXC on the CPU.

This does not validate CUDA execution. A separate full-mesh oracle is required;
we also exercise its gradient/adjoint/stress contract with this shared algebra.
"""
import argparse
import ctypes as ct
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np

WRAPPER='''subroutine reference_pbe(nspin,d,g,o) bind(C)
 use iso_c_binding
 use gridxc_gga, only: ggaxc
 implicit none
 integer(c_int),value :: nspin
 real(c_double),intent(in) :: d(nspin),g(3,nspin)
 real(c_double),intent(out) :: o(34)
 real(c_double) :: x,c,dx(nspin),dc(nspin),gx(3,nspin),gc(3,nspin)
 o=0._c_double
 call ggaxc('PBE',0,nspin,d,g,x,c,dx,dc,gx,gc)
 o(1)=x;o(2)=c;o(3:2+nspin)=dx;o(7:6+nspin)=dc
 o(11:10+3*nspin)=reshape(gx,(/3*nspin/))
 o(23:22+3*nspin)=reshape(gc,(/3*nspin/))
end subroutine
'''
CPP='''#include "pbe_math.hpp"
extern "C" void candidate_pbe(int spins,const double* d,const double* g,double* o) {
 double grad[4][3]{};
 for(int s=0;s<spins;s++) for(int j=0;j<3;j++) grad[s][j]=g[3*s+j];
 auto p=remag::pbe_point(spins,d,grad);
 for(int i=0;i<34;i++) o[i]=0.;
 o[0]=p.x;o[1]=p.c;
 for(int s=0;s<spins;s++) {
   o[2+s]=p.dx[s];o[6+s]=p.dc[s];
   for(int j=0;j<3;j++) {o[10+3*s+j]=p.gx[s][j];o[22+3*s+j]=p.gc[s][j];}
 }
}
'''

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(build,oracle,output):
    root=Path(__file__).resolve().parents[1];build=build.resolve()
    report=dict(gpu_validation=None,script_sha256=sha(Path(__file__)),
                pbe_math_sha256=sha(root/'native/pbe_math.hpp'),
                reference_library_sha256=sha(build/'libgridxc/libgridxc.a'),
                oracle_json_sha256=sha(oracle),pointwise=[],mesh=[])
    reference_info=json.loads(oracle.read_text())
    assert reference_info['passed'] and reference_info['old_noncollinear_scheme']
    array_path=oracle.parent/reference_info['arrays_file']
    assert sha(array_path)==reference_info['arrays_sha256']
    report['oracle_arrays_sha256']=sha(array_path)
    with tempfile.TemporaryDirectory(prefix='remag-pbe-math-') as tmp:
        tmp=Path(tmp);(tmp/'reference.f90').write_text(WRAPPER);(tmp/'candidate.cpp').write_text(CPP)
        compiler=shlex.split((build/'Src/CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())[0]
        subprocess.run([compiler,'-O2','-shared','-fPIC','-I'+str(build/'libgridxc/include'),
                        str(tmp/'reference.f90'),str(build/'libgridxc/libgridxc.a'),'-o',str(tmp/'reference.so')],check=True)
        subprocess.run(['c++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-shared','-fPIC',
                        '-I'+str(root/'native'),str(tmp/'candidate.cpp'),'-o',str(tmp/'candidate.so')],check=True)
        reference=ct.CDLL(str(tmp/'reference.so')).reference_pbe
        candidate=ct.CDLL(str(tmp/'candidate.so')).candidate_pbe
        for func in (reference,candidate): func.argtypes=[ct.c_int]+[ct.c_void_p]*3;func.restype=None
        def evaluate(func,d,g):
            d=np.ascontiguousarray(d);g=np.ascontiguousarray(g);o=np.zeros(34)
            func(d.size,d.ctypes.data,g.ctypes.data,o.ctypes.data)
            assert np.isfinite(o).all()
            return o
        rng=np.random.default_rng(9182)
        for spins in (1,2,4):
            maximum=0.;relative=0.;count=0
            for sample in range(300):
                total=10**rng.uniform(-14,2);polarization=rng.uniform(0.,.999)
                if spins==1: d=np.array([total])
                elif spins==2: d=total*np.array([1+polarization,1-polarization])/2
                else:
                    theta=rng.uniform(0,np.pi);phi=rng.uniform(-np.pi,np.pi);moment=polarization*total
                    d=np.array([(total+moment*np.cos(theta))/2,(total-moment*np.cos(theta))/2,
                                moment*np.sin(theta)*np.cos(phi)/2,-moment*np.sin(theta)*np.sin(phi)/2])
                g=rng.normal(size=(spins,3))*total*rng.uniform(0,4)
                if sample%10==0: g.fill(0.)
                if sample%10==1 and spins==2: d[1]=0.;g[1]=0.
                if sample%10==2 and spins==2: d[1]=-.1*d[0] # pinned collinear clipping
                expected=evaluate(reference,d,g);actual=evaluate(candidate,d,g)
                np.testing.assert_allclose(actual,expected,atol=1.e-10,rtol=1.e-10)
                maximum=max(maximum,float(np.max(np.abs(actual-expected))))
                relative=max(relative,float(np.max(np.abs(actual-expected)/(1+np.abs(expected)))))
                count+=1
            report['pointwise'].append(dict(spins=spins,fixtures=count,max_absolute_error=maximum,
                                            max_scaled_error=relative,passed=True))
        arrays=np.load(array_path)
        weights=(-1/60,3/20,-3/4,0.,3/4,-3/20,1/60)
        for record in reference_info['checks']:
            label=record['fixture'];d=arrays[label+'_density'];cell=arrays[label+'_cell']
            shape=d.shape[:3];spins=d.shape[-1];volume=abs(np.linalg.det(cell));n=np.prod(shape)
            reciprocal=np.linalg.inv(cell).T@np.diag(shape)
            grad=np.zeros((*d.shape,3))
            for axis in range(3):
                for offset,weight in zip(range(-3,4),weights):
                    for j in range(3):
                        grad[...,j]+=reciprocal[j,axis]*weight*np.roll(d,-offset,axis=axis)
            fields=np.zeros((*shape,34));total=d[...,:min(2,spins)].sum(axis=-1)
            for index in np.ndindex(shape):
                if total[index]>=1.e-15: fields[index]=evaluate(candidate,d[index],grad[index])
            gx=fields[...,10:10+3*spins].reshape((*shape,spins,3));gc=fields[...,22:22+3*spins].reshape((*shape,spins,3))
            vx=fields[...,2:2+spins].copy();vc=fields[...,6:6+spins].copy()
            for axis in range(3):
                for offset,weight in zip(range(-3,4),weights):
                    vx+=np.roll(np.sum(gx*reciprocal[:,axis],axis=-1)*weight,offset,axis=axis)
                    vc+=np.roll(np.sum(gc*reciprocal[:,axis],axis=-1)*weight,offset,axis=axis)
            energies=np.array([np.sum(total*fields[...,0]),np.sum(total*fields[...,1]),0.,0.])*2*volume/n
            energies[2]=energies[0]-2*volume/n*np.sum(d*vx)
            energies[3]=energies[1]-2*volume/n*np.sum(d*vc)
            potential=2*(vx+vc)
            stress=np.eye(3)*energies[:2].sum()/volume
            for row in range(3):
                for col in range(3): stress[row,col]-=2/n*np.sum(grad[...,row]*(gx+gc)[...,col])
            errors={}
            for name,actual in (('energies',energies),('potential',potential),('stress',stress)):
                expected=arrays[label+'_'+name]
                np.testing.assert_allclose(actual,expected,atol=1.e-9,rtol=1.e-10)
                errors[name]=float(np.max(np.abs(actual-expected)))
            report['mesh'].append(dict(fixture=label,max_absolute_errors=errors,passed=True))
    report['passed']=True;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"PBE CPU algebra: 900 point fixtures and {len(report['mesh'])} full-mesh fixtures passed; CUDA unvalidated")

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=Path('build/upstream'))
    parser.add_argument('--oracle',type=Path,default=Path('results/local-pbe-reference-20261005/reference.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.build,args.oracle,args.output)
