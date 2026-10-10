#!/usr/bin/env python3
"""Retain the literal pinned Hamiltonian seed/SOC arithmetic before porting it.

This compiles unchanged blocks extracted from setup_hamiltonian.F. It supplies
the local grid contribution explicitly; it is not a full setup_H or SCF oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def blocks(source):
    text=source.read_text()
    save=text[text.index('!     Save present H matrix'):text.index('!$OMP single')]
    start=text.index('      ! Initialize diagonal Hamiltonian')
    initialize=text[start:text.index('! ..................',start)]
    start=text.index('      Ekin = 0.0_dp')
    energies=text[start:text.index('!$OMP end parallel',start)]
    start=text.index('      if ( spin%SO_offsite ) then',text.index('      call dhscf('))
    finish=text[start:text.index('! Energy term coming from the chemical potential',start)]
    return save,initialize,energies,finish


def build_reference(upstream,build,tmp):
    upstream=upstream.resolve();build=build.resolve();src=build/'Src'
    original=upstream/'Src/setup_hamiltonian.F';save,initialize,energies,finish=blocks(original)
    wrapper='''      program reference
      use precision, only: dp
      use m_spin, only: spin
      implicit none
      integer :: maxnh,nh,soc,io,ispin,fi,fo
      real(dp),allocatable :: H(:,:),Hold(:,:),Dscf(:,:),H_kin(:)
      real(dp),allocatable :: H_vkb(:),H_so_on(:,:),grid(:,:)
      complex(dp),allocatable :: H_so_off(:,:)
      complex(dp) :: Dc
      real(dp) :: Ekin,Enl,Eso
      character(2048) :: input,output
      call get_command_argument(1,input)
      call get_command_argument(2,output)
      open(newunit=fi,file=trim(input),access='stream',
     &     form='unformatted',status='old')
      read(fi) maxnh,nh,soc
      spin%H=nh
      spin%spinor=min(nh,2)
      spin%SO_onsite=soc==1
      spin%SO_offsite=soc==2
      allocate(H(maxnh,nh),Hold(maxnh,nh),Dscf(maxnh,nh))
      allocate(H_kin(maxnh),H_vkb(maxnh),H_so_on(maxnh,6))
      allocate(H_so_off(maxnh,4),grid(maxnh,nh))
      read(fi) H,Dscf,H_kin,H_vkb,H_so_on,H_so_off,grid
      close(fi)
'''+save+initialize+energies+'''
      open(newunit=fo,file=trim(output),access='stream',
     &     form='unformatted',status='replace')
      write(fo) Hold,H,Ekin,Enl,Eso
      H=H+grid
'''+finish+'''
      write(fo) H
      close(fo)
      end program
'''
    code=tmp/'reference.F';code.write_text(wrapper);exe=tmp/'reference'
    link=shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())
    libraries=link[next(i for i,s in enumerate(link) if Path(s).name=='libsiesta.a'):]
    flags=(src/'CMakeFiles/SIESTA.libsiesta.dir/flags.make').read_text()
    includes=next(line.split('=',1)[1].strip() for line in flags.splitlines() if line.startswith('Fortran_INCLUDES ='))
    subprocess.run([link[0],'-O2','-ffixed-line-length-none',*shlex.split(includes),'-I'+str(src),
                    str(code),'-o',str(exe),*libraries],cwd=src,check=True)
    return exe,wrapper,hashlib.sha256((save+initialize+energies+finish).encode()).hexdigest()


def evaluate(exe,tmp,soc,h,d,kinetic,nonlocal_h,onsite,offsite,grid):
    n,nh=h.shape;inp=tmp/'input';out=tmp/'output'
    inp.write_bytes(np.array([n,nh,soc],dtype=np.int32).tobytes()+
        b''.join(a.tobytes(order='F') for a in (h,d,kinetic,nonlocal_h,onsite,offsite,grid)))
    subprocess.run([str(exe),str(inp),str(out)],cwd=tmp,check=True,capture_output=True)
    raw=np.fromfile(out,dtype=np.float64);assert len(raw)==3*n*nh+3 and np.isfinite(raw).all()
    old=raw[:n*nh].reshape((n,nh),order='F')
    seed=raw[n*nh:2*n*nh].reshape((n,nh),order='F')
    energy=raw[2*n*nh:2*n*nh+3]
    final=raw[2*n*nh+3:].reshape((n,nh),order='F')
    return old,seed,energy,final,sha(inp)


def run(upstream,build,output):
    original=upstream.resolve()/'Src/setup_hamiltonian.F';src=build.resolve()/'Src'
    report=dict(reference='Literal unchanged setup_hamiltonian.F seed, energy and post-grid SOC blocks',
        full_setup_hamiltonian_validation=False,gpu_validation=None,fixture_script_sha256=sha(Path(__file__)),
        source_sha256=sha(original),reference_library_sha256=sha(src/'libsiesta.a'),
        checks=[])
    arrays={};rng=np.random.default_rng(730516)
    with tempfile.TemporaryDirectory(prefix='remag-hamiltonian-seeds-') as directory:
        tmp=Path(directory);exe,wrapper,block_hash=build_reference(upstream,build,tmp)
        report['extracted_blocks_sha256']=block_hash
        for nh,soc in ((1,0),(2,0),(4,0),(8,1),(8,2)):
            for n in (19,128,513):
                label=f'spin{nh}_soc{soc}_capacity{n}'
                h=rng.normal(size=(n,nh));d=rng.normal(size=(n,nh))*.1
                kinetic=rng.normal(size=n);nonlocal_h=rng.normal(size=n)
                onsite=rng.normal(size=(n,6))*.01
                offsite=(rng.normal(size=(n,4))+1j*rng.normal(size=(n,4)))*.01
                grid=rng.normal(size=(n,nh))*.02
                if nh==8: grid[:,4:6]=0.;grid[:,6:8]=grid[:,2:4]
                old,seed,energy,final,input_hash=evaluate(exe,tmp,soc,h,d,kinetic,nonlocal_h,onsite,offsite,grid)
                expected=np.zeros((n,nh));expected[:,:min(nh,2)]=(kinetic+nonlocal_h)[:,None]
                if soc==1: expected[:,2:]=onsite
                expected_final=expected+grid
                expected_energy=np.array([np.sum(kinetic[:,None]*d[:,:min(nh,2)]),
                                          np.sum(nonlocal_h[:,None]*d[:,:min(nh,2)]),0.])
                if soc==1:
                    expected_energy[2]=np.sum(onsite[:,0]*d[:,6]+onsite[:,1]*d[:,7]+
                        onsite[:,4]*d[:,2]+onsite[:,5]*d[:,3]-onsite[:,2]*d[:,4]-onsite[:,3]*d[:,5])
                elif soc==2:
                    spin_density=np.column_stack((d[:,0]+1j*d[:,4],d[:,1]+1j*d[:,5],
                                                   d[:,2]+1j*d[:,3],d[:,6]-1j*d[:,7]))
                    expected_energy[2]=np.real(np.sum(offsite[:,[0,1,3,2]]*spin_density))
                    expected_final+=np.column_stack((offsite[:,0].real,offsite[:,1].real,offsite[:,2].real,
                        offsite[:,2].imag,offsite[:,0].imag,offsite[:,1].imag,offsite[:,3].real,-offsite[:,3].imag))
                np.testing.assert_array_equal(old,h);np.testing.assert_array_equal(seed,expected)
                np.testing.assert_array_equal(final,expected_final)
                np.testing.assert_allclose(energy,expected_energy,atol=1.e-11,rtol=1.e-12)
                for name,a in (('h',h),('density',d),('kinetic',kinetic),('nonlocal',nonlocal_h),('onsite',onsite),
                               ('offsite',offsite),('grid',grid),('seed',seed),('energies',energy),('final',final)):
                    arrays[label+'_'+name]=a
                report['checks'].append(dict(fixture=label,capacity=n,spin_components=nh,soc=soc,input_sha256=input_hash,
                    max_energy_error=float(np.max(np.abs(energy-expected_energy))),matrices_exact=True,passed=True))
    output.parent.mkdir(parents=True,exist_ok=True)
    artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**arrays)
    wrapper_path=output.with_name('hamiltonian-seeds-reference.F');wrapper_path.write_text(wrapper)
    report.update(arrays_file=artifact.name,arrays_sha256=sha(artifact),wrapper_file=wrapper_path.name,
                  wrapper_sha256=sha(wrapper_path),passed=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"Hamiltonian seed/SOC reference scenarios: {len(report['checks'])}; passed=True; GPU validation=None")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=Path('upstream/siesta'))
    parser.add_argument('--build',type=Path,default=Path('build/upstream'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.upstream,args.build,args.output)
