#!/usr/bin/env python3
"""Retain full upstream convergence decisions before moving control to device.

The wrapper calls the unchanged scfconvergence_test with printing disabled.
Five supported criteria, the outer minimum-iteration guard and converger reset
are covered. This does not implement a GPU loop or qualify loop termination.
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
 use siesta_options
 use m_scfconvergence_test, only: scfconvergence_test
 use m_convergence, only: converger_t,reset,set_tolerance,is_converged
 use m_energies, only: init_Energies,Eharrs,FreeE
 use parallel, only: IOnode
 use m_ts_global_vars, only: TSrun
 use dftu_specs, only: switch_dftu
 implicit none
 type(converger_t) :: ch,cf
 integer :: fi,fo,mask,minimum,cycles,reset_before,k,iteration
 real(dp) :: tolerances(5),values(5)
 logical :: converged
 character(2048) :: input,output
 call get_command_argument(1,input);call get_command_argument(2,output)
 open(newunit=fi,file=trim(input),access='stream',form='unformatted',status='old')
 read(fi) mask,minimum,cycles,reset_before,tolerances
 call init_Energies()
 IOnode=.false.;TSrun=.false.;switch_dftu=.false.;harrisfun=.false.
 converge_DM=btest(mask,0);converge_H=btest(mask,1);converge_EDM=btest(mask,2)
 converge_Eharr=btest(mask,3);converge_FreeE=btest(mask,4)
 dDtol=tolerances(1);dHtol=tolerances(2);tolerance_EDM=tolerances(3)
 call reset(ch);call reset(cf)
 call set_tolerance(ch,tolerances(4));call set_tolerance(cf,tolerances(5))
 open(newunit=fo,file=trim(output),access='stream',form='unformatted',status='replace')
 iteration=0
 do k=1,cycles
   if(k==reset_before) then
     call reset(ch);call reset(cf);iteration=0
   endif
   iteration=iteration+1
   read(fi) values
   Eharrs=values(4);FreeE=values(5)
   call scfconvergence_test(iteration==1,iteration,values(1),values(2),values(3),0._dp,ch,cf,converged)
   write(fo) merge(1,0,converged),merge(1,0,converged.and.iteration>=minimum), &
             merge(1,0,is_converged(ch)),merge(1,0,is_converged(cf)),iteration
 enddo
 close(fi);close(fo)
end program
'''


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run(upstream,build,output):
    build=build.resolve();src=build/'Src';output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(reference='Unchanged scfconvergence_test and m_convergence; outer minimum-iteration guard',
        gpu_validation=None,checks=[],fixture_script_sha256=sha(Path(__file__)),
        reference_library_sha256=sha(src/'libsiesta.a'),
        source_sha256={name:sha(upstream/'Src'/name) for name in ('scfconvergence_test.F','m_convergence.f90','siesta_forces.F90')})
    retained={}
    with tempfile.TemporaryDirectory(prefix='remag-convergence-') as directory:
        tmp=Path(directory);code=tmp/'reference.f90';code.write_text(WRAPPER);exe=tmp/'reference'
        link=shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())
        libraries=link[next(i for i,s in enumerate(link) if Path(s).name=='libsiesta.a'):]
        flags=(src/'CMakeFiles/SIESTA.libsiesta.dir/flags.make').read_text()
        includes=next(line.split('=',1)[1].strip() for line in flags.splitlines() if line.startswith('Fortran_INCLUDES ='))
        subprocess.run([link[0],'-O2',*shlex.split(includes),'-I'+str(src),str(code),'-o',str(exe),*libraries],cwd=src,check=True)
        tol=np.array([.125,.25,.5,.125,.25])
        base=np.array([[2.,2.,2.,0.,0.],[1.,1.,1.,.125,.25],[.5,.5,.5,.125,.25],
                       [2.,2.,2.,4.,8.],[0.,0.,0.,-8.,-16.],[1.,1.,1.,-8.,-16.],
                       [0.,0.,0.,0.,0.],[2.,2.,2.,0.,0.]])
        values=base.copy();values[:,:3]*=tol[:3]
        values[5,:3]=np.nextafter(tol[:3],0.)
        fixtures=[(f'criteria_{mask:02d}',mask,0,0,values) for mask in range(1,32)]
        fixtures += [('minimum_iterations',31,7,0,values),('reset_clears_energy_latch',31,0,5,values),
                     ('strict_threshold',7,0,0,np.column_stack((np.array([np.nextafter(tol[:3],0.),tol[:3],
                         np.nextafter(tol[:3],np.inf)]),np.zeros((3,2)))))]
        for label,mask,minimum,reset,sequence in fixtures:
            inp=tmp/'input';out=tmp/'output'
            inp.write_bytes(np.array([mask,minimum,len(sequence),reset],np.int32).tobytes()+tol.tobytes()+sequence.tobytes())
            subprocess.run([str(exe),str(inp),str(out)],cwd=tmp,check=True,capture_output=True)
            actual=np.fromfile(out,np.int32).reshape((-1,5))
            latch=np.zeros(2,bool);previous=None;iteration=0;expected=[]
            for k,v in enumerate(sequence,1):
                if k==reset: latch[:]=False;previous=None;iteration=0
                iteration+=1
                if previous is not None: latch|=np.abs(previous-v[3:])<tol[3:]
                previous=v[3:]
                tests=np.concatenate((v[:3]<tol[:3],latch))
                converged=all(tests[i] for i in range(5) if mask&(1<<i))
                expected.append([converged,converged and iteration>=minimum,*latch,iteration])
            np.testing.assert_array_equal(actual,np.array(expected,np.int32))
            retained[label+'_values']=sequence;retained[label+'_decisions']=actual
            report['checks'].append(dict(fixture=label,criteria_mask=mask,minimum_iterations=minimum,
                reset_before=reset,steps=len(sequence),input_sha256=sha(inp),passed=True))
        retained['tolerances']=tol
    arrays=output.with_suffix('.npz');np.savez_compressed(arrays,**retained)
    wrapper=output.with_name('scf-convergence-reference.f90');wrapper.write_text(WRAPPER)
    report.update(passed=True,arrays_file=arrays.name,arrays_sha256=sha(arrays),wrapper_file=wrapper.name,
        wrapper_sha256=sha(wrapper),notes='Energy convergers latch true until reset; matrix criteria use strict less-than each step.')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(f"SCF convergence upstream sequences: {len(report['checks'])}; passed=True; GPU validation=None")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,default=Path('upstream/siesta'))
    p.add_argument('--build',type=Path,default=Path('build/upstream'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.upstream,args.build,args.output)
