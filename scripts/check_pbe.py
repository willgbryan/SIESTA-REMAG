#!/usr/bin/env python3
"""Retain a full-mesh PBE oracle from pristine GridXC before its GPU migration.

This is CPU reference preparation, not a GPU validation or an SCF speed claim.
The reference includes the pinned legacy NC spin transformation, finite-
difference gradients/adjoint potential, four reductions and all nine stresses.
"""
import argparse
import ctypes as ct
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np

WRAPPER='''subroutine remag_reference_pbe(n1,n2,n3,nspin,cell,density,energies,stress,potential) bind(C)
 use iso_c_binding
 use gridxc, only: gridxc_init,gridxc_setXC,gridxc_cellXC
 implicit none
 integer(c_int),value :: n1,n2,n3,nspin
 real(c_double),intent(in) :: cell(3,3),density(n1,n2,n3,nspin)
 real(c_double),intent(out) :: energies(4),stress(3,3),potential(n1,n2,n3,nspin)
 logical,save :: initialized=.false.
 if (.not.initialized) then
   call gridxc_init()
   call gridxc_setXC(1,(/'GGA'/),(/'PBE'/),(/1._c_double/),(/1._c_double/))
   initialized=.true.
 endif
 call gridxc_cellXC(0,cell,(/n1,n2,n3/),0,n1-1,0,n2-1,0,n3-1,nspin,density, &
      energies(1),energies(2),energies(3),energies(4),stress,potential,keep_input_distribution=.true.)
end subroutine
'''

def fixtures():
    for label,shape,nspin in (
        ('constant_scalar',(6,8,10),1),('constant_collinear',(6,8,10),2),
        ('constant_spinor',(6,8,10),4),('smooth_scalar',(8,10,12),1),
        ('smooth_collinear',(8,10,12),2),('noncollinear_texture',(7,9,5),4),
        ('small_periodic_spinor',(2,3,4),4),('fully_polarized',(8,6,10),2),
        ('density_floor',(4,6,8),4)):
        cell=np.asfortranarray([[8.,1.1,-.4],[0.,9.,.6],[0.,0.,10.]])
        xyz=np.indices(shape)/np.array(shape)[:,None,None,None]
        rho=.12+.02*np.cos(2*np.pi*xyz[0])+.015*np.sin(2*np.pi*xyz[1])+.01*np.cos(4*np.pi*xyz[2])
        density=np.zeros((*shape,nspin),order='F')
        if nspin==1: density[...,0]=rho
        elif nspin==2:
            pol=.4+.1*np.sin(2*np.pi*xyz[2])
            density[...,0]=rho*(1+pol)/2;density[...,1]=rho*(1-pol)/2
        else:
            theta=.8+.3*np.cos(2*np.pi*xyz[0]);phi=.4+.2*np.sin(2*np.pi*xyz[1]);moment=.45*rho
            density[...,0]=(rho+moment*np.cos(theta))/2;density[...,1]=(rho-moment*np.cos(theta))/2
            density[...,2]=moment*np.sin(theta)*np.cos(phi)/2
            density[...,3]=-moment*np.sin(theta)*np.sin(phi)/2
        if label.startswith('constant'):
            density.fill(0.);density[...,:min(nspin,2)]=.12/min(nspin,2)
        if label=='fully_polarized': density[...,0]=rho;density[...,1]=0.
        if label=='density_floor': density.fill(1.e-25)
        yield label,cell,density

def run(upstream,build,output):
    upstream=upstream.resolve();build=build.resolve();source=upstream/'External/libgridxc/src'
    ggaxc=(source/'ggaxc.F90').read_text()
    if not re.search(r'logical\s*,\s*parameter\s*::\s*old_scheme\s*=\s*\.true\.',ggaxc,re.I):
        raise ValueError('Reference no longer selects the pinned legacy NC transformation')
    names=('cellxc.F90','ggaxc.F90','ldaxc.F90','xcmod.F90','cellsubs.F90')
    report=dict(reference='Unmodified pristine full GridXC cellXC, built-in PBE, irel=0, FP64, serial',
                old_noncollinear_scheme=True,stencil_radius=3,energy_unit='Rydberg',
                source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names},
                fixture_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reference_library_sha256=hashlib.sha256((build/'libgridxc/libgridxc.a').read_bytes()).hexdigest(),
                gpu_validation=None,checks=[])
    arrays={}
    with tempfile.TemporaryDirectory(prefix='remag-pbe-reference-') as temporary:
        temporary=Path(temporary);wrapper=temporary/'reference.f90';wrapper.write_text(WRAPPER)
        library=temporary/'reference.so'
        compiler=shlex.split((build/'Src/CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())[0]
        subprocess.run([compiler,'-O2','-shared','-fPIC','-I'+str(build/'libgridxc/include'),str(wrapper),
                        str(build/'libgridxc/libgridxc.a'),'-o',str(library)],cwd=temporary,check=True)
        reference=ct.CDLL(str(library)).remag_reference_pbe
        reference.argtypes=[ct.c_int]*4+[ct.c_void_p]*5;reference.restype=None
        def evaluate(cell,density):
            cell=np.asfortranarray(cell);density=np.asfortranarray(density)
            energies=np.zeros(4);stress=np.zeros((3,3),order='F');potential=np.zeros_like(density,order='F')
            reference(*density.shape,cell.ctypes.data,density.ctypes.data,energies.ctypes.data,stress.ctypes.data,potential.ctypes.data)
            assert all(np.isfinite(a).all() for a in (energies,stress,potential))
            return energies,stress,potential
        constant=None
        rng=np.random.default_rng(829)
        for label,cell,density in fixtures():
            energies,stress,potential=evaluate(cell,density)
            volume=float(abs(np.linalg.det(cell)));dvol=volume/np.prod(density.shape[:3])
            # Literal upstream double-counting convention sums all stored spin
            # components once; do not silently replace it by a different trace.
            identity=float(energies[:2].sum()-dvol*np.sum(density*potential)-energies[2:].sum())
            np.testing.assert_allclose(identity,0.,atol=1.e-9)
            check=dict(fixture=label,shape=density.shape,energies=energies.tolist(),stress=stress.tolist(),
                       potential_range=[float(potential.min()),float(potential.max())],
                       double_counting_identity_error=abs(identity))
            if label.startswith('constant'):
                if constant is None: constant=energies.copy()
                np.testing.assert_allclose(energies,constant,atol=1.e-11,rtol=1.e-12)
                for spin in range(density.shape[-1]):
                    np.testing.assert_allclose(potential[...,spin],potential[0,0,0,spin],atol=1.e-12)
                np.testing.assert_allclose(stress,np.eye(3)*energies[:2].sum()/volume,atol=1.e-12)
            if label.startswith('smooth'):
                # Independent variational checks for differentiable collinear
                # fixtures. Do not impose a new NC convention on upstream.
                delta=np.asfortranarray(rng.normal(scale=.01,size=density.shape));eps=1.e-5
                plus=evaluate(cell,density+eps*delta)[0][:2].sum()
                minus=evaluate(cell,density-eps*delta)[0][:2].sum()
                derivative=(plus-minus)/(2*eps);predicted=dvol*np.sum(potential*delta)
                np.testing.assert_allclose(derivative,predicted,atol=1.e-7,rtol=1.e-6)
                check['potential_variation_error']=float(abs(derivative-predicted))
                stress_errors=[]
                for row,col in ((0,0),(0,1),(2,2)):
                    deform=np.eye(3);deform[row,col]+=eps
                    plus=evaluate(deform@cell,density)[0][:2].sum()
                    deform[row,col]-=2*eps
                    minus=evaluate(deform@cell,density)[0][:2].sum()
                    measured=(plus-minus)/(2*eps*volume)
                    np.testing.assert_allclose(measured,stress[row,col],atol=1.e-9,rtol=1.e-6)
                    stress_errors.append(float(abs(measured-stress[row,col])))
                check['stress_variation_max_error']=max(stress_errors)
            if label=='density_floor':
                for array in (energies,stress,potential): np.testing.assert_array_equal(array,0.)
            for name,array in (('cell',cell),('density',density),('energies',energies),('stress',stress),('potential',potential)):
                arrays[f'{label}_{name}']=array
            check['passed']=True;report['checks'].append(check)
    output.parent.mkdir(parents=True,exist_ok=True)
    artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**arrays)
    report['arrays_file']=artifact.name;report['arrays_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
    report['passed']=all(c['passed'] for c in report['checks'])
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"Upstream full-mesh PBE fixtures: {len(report['checks'])}; CPU oracle checks passed={report['passed']}; GPU validation pending")
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=Path('upstream/siesta'))
    parser.add_argument('--build',type=Path,default=Path('build/upstream'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.upstream,args.build,args.output)
