#!/usr/bin/env python3
"""Compile patched Fortran against real pinned modules, without a solver run.

The C/CUDA library is an unresolved interface dependency for this compile-only
check. This does not build a runnable REMAG binary or verify GPU behavior.
Dependencies must already exist at their pinned revisions; no network fetches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time

from patch_siesta import patch

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'upstream/siesta')
    parser.add_argument('--dependencies',type=Path,help='Directory containing the four pinned dependency repositories')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compiler',default=shutil.which('gfortran'))
    parser.add_argument('--blas',type=Path)
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    if not args.compiler or not 1<=args.jobs<=16:parser.error('A Fortran compiler and 1..16 build jobs are required')
    if out.exists():parser.error('Output must be a new directory; failed checks are retained')
    lock=json.loads((ROOT/'upstream.lock.json').read_text());repos=[(source,Path('.'),lock['commit'])]
    dependencies=(args.dependencies or source/'External').resolve()
    for relative in lock['submodules']:
        pin=git(source,'ls-tree','HEAD',relative).split()[2]
        repos.append((dependencies/Path(relative).name,Path(relative),pin))
    for path,_,pin in repos:
        if git(path,'rev-parse','HEAD')!=pin or git(path,'status','--porcelain'):
            raise SystemExit('Missing, dirty or mismatched pinned source: '+str(path))
    out.mkdir(parents=True);target=out/'source';target.mkdir();started=time.monotonic()
    evidence=dict(format='REMAG-FORTRAN-COMPILE/1',source_commit=lock['commit'],
        dependencies=[dict(path=str(rel),commit=pin) for _,rel,pin in repos[1:]],
        implementation_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/patch_siesta.py',*sorted((ROOT/'native').glob('*.F90.in'))]},
        native_execution=False,cuda_compiled=False,cuda_linked=False,gpu_allocation=False,
        scope='All patched Fortran library sources and main program object against real pinned modules; unresolved C/CUDA interface dependency')
    try:
        with tempfile.TemporaryDirectory(dir=out,prefix='archives-') as temporary:
            for index,(path,rel,_) in enumerate(repos):
                archive=Path(temporary)/f'{index}.tar'
                subprocess.run(['git','-C',str(path),'archive','--format=tar','-o',str(archive),'HEAD'],check=True)
                destination=target/rel;destination.mkdir(parents=True,exist_ok=True)
                with tarfile.open(archive) as bundle:bundle.extractall(destination,filter='data')
        patch(target)
        cmake=target/'CMakeLists.txt';text=cmake.read_text();anchor='add_subdirectory(${REMAG_NATIVE_DIR} remag-native)'
        if text.count(anchor)!=1:raise ValueError('Expected one native link integration point')
        cmake.write_text(text.replace(anchor,'add_library(remag_gpu INTERFACE) # compile-only, unresolved C/CUDA symbols'))
        flags=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DFETCHCONTENT_FULLY_DISCONNECTED=ON','-DSIESTA_FIND_METHOD=source',
            '-DREMAG_NATIVE_DIR='+str(ROOT/'native'),'-DCMAKE_Fortran_COMPILER='+args.compiler]
        flags+=['-DSIESTA_WITH_'+n+'=OFF' for n in ('MPI','ELSI','DFTD3','FLOOK','NETCDF','OPENMP','FFTW','HOST_OPTIMIZATION')]
        if args.blas:flags+=['-DBLAS_LIBRARY='+str(args.blas.resolve()),'-DLAPACK_LIBRARY='+str(args.blas.resolve())]
        commands=[('configure',['cmake','-S',str(target),'-B',str(out/'build'),*flags]),
                  ('library',['cmake','--build',str(out/'build'),'--target','SIESTA.libsiesta','--parallel',str(args.jobs)])]
        evidence['commands']=[]
        def run(name,command,cwd=None):
            evidence['commands'].append(dict(name=name,argv=command,cwd=str(cwd or Path.cwd())))
            with (out/(name+'.log')).open('w') as log:
                subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        for name,command in commands:run(name,command)
        units=json.loads((out/'build/compile_commands.json').read_text())
        mains=[u for u in units if Path(u['file']).resolve()==target/'Src/siesta.F']
        if len(mains)!=1:raise ValueError('Expected one native main compilation command')
        command=mains[0].get('arguments') or shlex.split(mains[0]['command'])
        if '-c' not in command or '-o' not in command:raise ValueError('Main check must be compilation only')
        run('main',command,Path(mains[0]['directory']))
        library=out/'build/Src/libsiesta.a'
        main_object=Path(mains[0]['directory'])/command[command.index('-o')+1]
        members=subprocess.check_output(['ar','-t',str(library)],text=True).splitlines()
        evidence.update(status='passed',configured_source_translation_units=len([u for u in units if '/Src/' in u['file']]),
            compiled_library_members=len(members),
            library_sha256=sha(library),library_bytes=library.stat().st_size,main_object_sha256=sha(main_object),
            compiler=subprocess.check_output([args.compiler,'--version'],text=True).splitlines()[0])
    except Exception as exc:
        evidence.update(status='failed',error=type(exc).__name__,message=str(exc));raise
    finally:
        evidence['elapsed_seconds']=time.monotonic()-started
        evidence['logs']={p.name:sha(p) for p in sorted(out.glob('*.log'))}
        (out/'result.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({k:evidence[k] for k in ('status','elapsed_seconds','native_execution','cuda_linked')}))


if __name__=='__main__':main()
