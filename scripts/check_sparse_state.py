#!/usr/bin/env python3
"""Couple the production sparse owner to upstream seed/scalar/Pulay oracles.

Explicitly staged mock density and grid producers exercise state ownership and
energy composition. Real producer physics still requires full SIESTA regression.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np
import check_hamiltonian_seeds as seeds_ref
import check_scf_scalars as scalars_ref
import check_mixing as mixing_ref


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def mixing_executable(build,tmp):
    src=build.resolve()/'Src';wrapper=tmp/'reference.f90';wrapper.write_text(mixing_ref.WRAPPER)
    link=shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())
    libraries=link[next(i for i,s in enumerate(link) if Path(s).name=='libsiesta.a'):]
    flags=(src/'CMakeFiles/SIESTA.libsiesta.dir/flags.make').read_text()
    includes=next(line.split('=',1)[1].strip() for line in flags.splitlines() if line.startswith('Fortran_INCLUDES ='))
    exe=tmp/'reference'
    subprocess.run([link[0],'-O2',*shlex.split(includes),'-I'+str(src),str(wrapper),'-o',str(exe),*libraries],cwd=src,check=True)
    return exe


def mix(exe,tmp,n,spins,x,f):
    inp=tmp/'input';out=tmp/'output'
    inp.write_bytes(np.array([n,spins,3,len(x),0],np.int32).tobytes()+np.array([.05,.13]).tobytes()+
                   b''.join(a.tobytes(order='F')+b.tobytes(order='F') for a,b in zip(x,f)))
    subprocess.run([str(exe),str(inp),str(out)],cwd=tmp,check=True,capture_output=True)
    return np.fromfile(out).reshape((len(x),n*spins+2))[-1,:-2].reshape((n,spins),order='F')


def run(upstream,build,output,library=None):
    output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(reference='Unchanged upstream seed/SOC blocks, SCF scalars and stable Pulay',
        scope='Coupled production sparse owner with staged mock density/grid producers; not full SCF',
        gpu_validation=None,checks=[],fixture_script_sha256=sha(Path(__file__)),
        reference_library_sha256=sha(build/'Src/libsiesta.a'),
        first_mix_semantics='Upstream mixer.F always updates history, then reconstructs Xin+F when first-step mixing is disabled',
        mixer_source_sha256=sha(upstream/'Src/mixer.F'))
    retained={};candidate=None
    if library:
        candidate=ct.CDLL(str(library.resolve())).remag_check_sparse_state
        candidate.argtypes=[ct.c_int]*7+[ct.c_void_p]*10+[ct.POINTER(ct.c_int)]
        candidate.restype=ct.c_int
        report['native_library_sha256']=sha(library)
        report['production_library_sha256']=sha(library.with_name('libremag_gpu.so'))
    def execute(n,spins,soc,initial_energy,geometry,constants,initial,sequence,producers,failure=0):
        values=np.full((len(sequence),16),123.);borrowed=np.full((len(sequence),n*spins),123.)
        final=np.full_like(initial,123.);rejected=ct.c_int()
        inputs=[num,ptr,geometry,constants,initial,sequence,producers,values,borrowed,final]
        assert all(a.flags.c_contiguous for a in inputs)
        report['active_fixture']=dict(n=n,spins=spins,soc=soc,initial_energy=initial_energy,mix_first=mix_first,failure=failure)
        code=candidate(n,spins,soc,len(sequence),initial_energy,mix_first,failure,
            *[a.ctypes.data for a in inputs],ct.byref(rejected))
        report['active_fixture'].update(native_return=code,rejected_calls=rejected.value)
        if code:
            for name,a in (('scalars',values),('borrowed',borrowed),('final',final)):
                retained['failed_native_'+name]=a
        assert code==0,(code,n,spins,soc,failure)
        assert rejected.value>0
        return values,borrowed,final,rejected.value
    rng=np.random.default_rng(61007)
    try:
        with tempfile.TemporaryDirectory(prefix='remag-sparse-owner-') as directory:
            tmp=Path(directory);st=tmp/'seeds';sc=tmp/'scalars';mx=tmp/'mixing'
            for folder in (st,sc,mx): folder.mkdir()
            se,wrapper,block_hash=seeds_ref.build_reference(upstream,build,st)
            ce=scalars_ref.build_reference(build,sc);me=mixing_executable(build,mx)
            report['extracted_blocks_sha256']=block_hash
            for name,content in (('sparse-owner-seeds.F',wrapper),('sparse-owner-scalars.f90',scalars_ref.WRAPPER),
                                 ('sparse-owner-mixing.f90',mixing_ref.WRAPPER)):
                output.with_name(name).write_text(content)
            report['wrappers_sha256']={p.name:sha(p) for p in output.parent.glob('sparse-owner-*')}
            for n in (19,257):
                for spins,soc in ((1,0),(2,0),(4,0),(8,1),(8,2)):
                    for mix_first in (1,0):
                        label=f'spin{spins}_soc{soc}_n{n}_mix_first{mix_first}';cycles=8;eh=min(spins,4)
                        num=np.array([2,3,1,4],np.int32);ptr=np.array([0,4,9,n-7],np.int32)
                        overlap=rng.normal(size=n);kin=rng.normal(size=n);vkb=rng.normal(size=n)
                        on=rng.normal(size=(n,6))*.01
                        off=(rng.normal(size=(n,4))+1j*rng.normal(size=(n,4)))*.01
                        geometry=np.concatenate((overlap,kin,vkb,on.ravel(order='F'),off.ravel(order='F').view(np.float64)))
                        constants=np.array([.7,2.1,.12,.01])
                        initial_arrays=[rng.normal(size=(n,spins))*.1 for _ in range(4)]+[
                            rng.normal(size=(n,eh))*.1 for _ in range(2)]
                        initial=np.concatenate([a.ravel(order='F') for a in initial_arrays])
                        initial_energy=int(n==257)
                        h=initial_arrays[0].copy();d=initial_arrays[2].copy()
                        e=initial_arrays[4].copy() if initial_energy else np.zeros((n,eh))
                        producers=rng.normal(size=(cycles,10))*.1;producers[:,9]=np.abs(producers[:,9])
                        sequence=[];expected=[];borrowed=[];old_h=[];residuals=[]
                        for k in range(cycles):
                            nd=rng.normal(size=d.shape)*.1;ne=rng.normal(size=e.shape)*.2
                            grid=rng.normal(size=h.shape)*.02
                            if spins==8: grid[:,4:6]=0.;grid[:,6:8]=grid[:,2:4]
                            sequence.append(np.concatenate([a.ravel(order='F') for a in (nd,ne,grid)]))
                            borrowed.append(h.ravel(order='F'))
                            hold,seed,energies,nh,_=seeds_ref.evaluate(se,st,soc,h,nd,kin,vkb,on,off,grid)
                            p=producers[k]
                            seeds=np.array([constants[0],*energies,constants[1],p[4],p[3],p[1],p[5]+p[6],p[9],constants[3],constants[2]])
                            values,_=scalars_ref.evaluate(ce,sc,num,ptr,seeds,nh,hold,nd,d,ne,e,overlap)
                            expected.append(np.concatenate((values,energies)))
                            old_h.append(hold);residuals.append(nh-hold)
                            if k+1<cycles:
                                h=mix(me,mx,n,spins,old_h,residuals)
                                if k==0 and not mix_first: h=hold+(nh-hold)
                            else: h=nh
                            dold=d;eold=e;d=nd;e=ne
                        final=np.concatenate([a.ravel(order='F') for a in (h,hold,d,dold,e,eold)])
                        sequence=np.array(sequence);expected=np.array(expected);borrowed=np.array(borrowed)
                        record=dict(fixture=label,iterations=cycles,initial_energy=initial_energy,mix_first=mix_first)
                        for name,a in (('geometry',geometry),('num',num),('ptr',ptr),('constants',constants),('initial',initial),
                            ('sequence',sequence),('producer_values',producers),('expected_scalars',expected),
                            ('expected_borrowed',borrowed),('expected_final',final)):
                            retained[label+'_'+name]=a
                        if candidate:
                            actual,seen,exported,rejected=execute(n,spins,soc,initial_energy,geometry,constants,initial,sequence,producers)
                            for name,a in (('actual_scalars',actual),('actual_borrowed',seen),('actual_final',exported)):
                                retained[label+'_'+name]=a
                            np.savez_compressed(output.with_suffix('.npz'),**retained)
                            np.testing.assert_allclose(actual,expected,atol=1.e-9,rtol=1.e-10)
                            np.testing.assert_allclose(seen,borrowed,atol=1.e-9,rtol=1.e-10)
                            np.testing.assert_allclose(exported,final,atol=1.e-9,rtol=1.e-10)
                            record.update(rejected_calls=rejected,max_scalar_error=float(np.max(np.abs(actual-expected))),
                                max_borrowed_error=float(np.max(np.abs(seen-borrowed))),max_final_error=float(np.max(np.abs(exported-final))))
                        record['passed']=True;report['checks'].append(record)
                        if candidate and not initial_energy:
                            actual,seen,exported,rejected=execute(n,spins,soc,initial_energy,geometry,constants,initial,sequence,producers,-2)
                            wanted=final.copy();wanted[-n*eh:]=initial[-n*eh:]
                            for name,a in (('actual_scalars',actual),('actual_borrowed',seen),('actual_final',exported),('expected_final',wanted)):
                                retained[label+'_absent_eold_'+name]=a
                            np.testing.assert_allclose(actual,expected,atol=1.e-9,rtol=1.e-10)
                            np.testing.assert_allclose(seen,borrowed,atol=1.e-9,rtol=1.e-10)
                            np.testing.assert_allclose(exported,wanted,atol=1.e-9,rtol=1.e-10)
                            report['checks'].append(dict(fixture=label+'_absent_host_eold',rejected_calls=rejected,passed=True))
            if candidate:
                actual,seen,exported,rejected=execute(n,spins,soc,initial_energy,geometry,constants,initial,sequence,producers,-1)
                np.testing.assert_allclose(actual,expected,atol=1.e-9,rtol=1.e-10)
                np.testing.assert_allclose(seen,borrowed,atol=1.e-9,rtol=1.e-10)
                np.testing.assert_allclose(exported,final,atol=1.e-9,rtol=1.e-10)
                report['checks'].append(dict(fixture='same_geometry_new_cycle_resets_pulay',rejected_calls=rejected,passed=True))
                for failure,label in enumerate(('host_mutation','geometry_mutation','grid_error','fermi_error','seed_nan'),1):
                    geometry=geometry.copy()
                    if failure==5: geometry[n]=np.nan
                    _,_,_,rejected=execute(n,spins,soc,initial_energy,geometry,constants,initial,sequence,producers,failure)
                    report['checks'].append(dict(fixture=label,rejected_calls=rejected,passed=True))
                # A failed owner must permit an explicit new-geometry lifetime.
                geometry[n]=kin[0]
                actual,seen,exported,rejected=execute(n,spins,soc,initial_energy,geometry,constants,initial,sequence,producers)
                np.testing.assert_allclose(actual,expected,atol=1.e-9,rtol=1.e-10)
                np.testing.assert_allclose(exported,final,atol=1.e-9,rtol=1.e-10)
                report['checks'].append(dict(fixture='recovery_after_failed_lifetimes',rejected_calls=rejected,passed=True))
                report['gpu_validation']=True
        report.update(passed=True)
        report.pop('active_fixture',None)
    except Exception as error:
        report.update(passed=False,error=f'{type(error).__name__}: {error}')
        raise
    finally:
        artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**retained)
        report.update(arrays_file=artifact.name,arrays_sha256=sha(artifact))
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"Sparse owner scenarios: {len(report['checks'])}; passed=True; GPU validation={report['gpu_validation']}")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,default=Path('upstream/siesta'))
    p.add_argument('--build',type=Path,default=Path('build/upstream'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path)
    args=p.parse_args();run(args.upstream,args.build,args.output,args.library)
