#!/usr/bin/env python3
"""Retain exact upstream off-site SOC projector operators and their derivatives.

This CPU oracle extracts calc_Vj_offsiteSO literally from the pinned nlefsm.f.
Sparse density reconstruction, force accumulation and material ownership are
separate gates; no GPU integration is claimed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import numpy as np

FIELDS=('epsilon','left','right','left_gradient','right_gradient')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def block(upstream):
    text=(upstream/'Src/nlefsm.f').read_text();a=text.index('      subroutine calc_Vj_offsiteSO(')
    b=text.index('      end subroutine calc_Vj_offsiteSO',a)+len('      end subroutine calc_Vj_offsiteSO')
    return text[a:b]+'\n'
def binary(l,split,values):
    return np.array([l,int(split),values['left'].shape[0]],np.int32).tobytes()+b''.join(values[k].tobytes() for k in FIELDS)
def reference(upstream,build,folder):
    source=block(upstream)
    wrapper='''      module pristine_soc
      use iso_fortran_env, only: dp=>real64
      implicit none
      contains
'''+source+'''      end module
      program reference
      use pristine_soc
      implicit none
      integer l,split,n,fi,fo,q
      real(dp) epsilon(2),ion
      real(dp),allocatable :: si(:,:,:),sj(:,:,:),gi(:,:,:,:),
     & gj(:,:,:,:)
      complex(dp) v(2,2),g(3,2,2)
      character(2048) input,output
      call get_command_argument(1,input)
      call get_command_argument(2,output)
      open(newunit=fi,file=trim(input),access='stream',
     & form='unformatted',status='old')
      read(fi) l,split,n
      allocate(si(-l:l,2,n),sj(-l:l,2,n),
     & gi(3,-l:l,2,n),gj(3,-l:l,2,n))
      read(fi) epsilon,si,sj,gi,gj
      close(fi)
      open(newunit=fo,file=trim(output),access='stream',
     & form='unformatted',status='replace')
      do q=1,n
        call calc_Vj_offsiteSO(l,epsilon,si(:,:,q),sj(:,:,q),
     &   gi(:,:,:,q),gj(:,:,:,q),ion,v,g,split==1)
        write(fo) ion,real(v,dp),aimag(v),real(g,dp),aimag(g)
      enddo
      close(fo)
      end program
      subroutine die(message)
      character(*) message
      write(*,*) message
      stop 1
      end subroutine
'''
    path=folder/'reference.F';path.write_text(wrapper);exe=folder/'reference'
    compiler=shlex.split((build/'Src/CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())[0]
    subprocess.run([compiler,'-O2','-ffixed-line-length-none',str(path),'-o',str(exe)],check=True,cwd=folder)
    return exe,wrapper

def run(upstream,build,output):
    upstream,build,output=upstream.resolve(),build.resolve(),output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(613719);arrays={};checks=[]
    signs=((1.3,.8),(-.4,-.9),(.7,-1.1),(-.6,.9),(0.,.7),(0.,0.))
    with tempfile.TemporaryDirectory(prefix='remag-soc-projector-') as directory:
        folder=Path(directory);exe,wrapper=reference(upstream,build,folder)
        fixtures=[(l,s,split,17) for l in range(1,6) for s in range(len(signs)) for split in (0,1)]
        fixtures.extend([(2,0,split,257) for split in (0,1)])
        fixtures.extend([(1,0,split,131073) for split in (0,1)])
        for l,s,split,n in fixtures:
            name=f'l{l}_epsilon{s}_split{split}_queries{n}';width=2*l+1
            data=dict(epsilon=np.array(signs[s]),left=rng.normal(size=(n,2,width))*.1,right=rng.normal(size=(n,2,width))*.1,
                      left_gradient=rng.normal(size=(n,2,width,3))*.2,right_gradient=rng.normal(size=(n,2,width,3))*.2)
            # Include self pairs, zero amplitudes and gradients independently.
            data['right'][0]=data['left'][0];data['left'][1]=0.;data['right'][2]=0.;data['left_gradient'][3]=0.
            inp=folder/'input.bin';out=folder/'output.bin';raw=binary(l,split,data);inp.write_bytes(raw)
            subprocess.run([str(exe),str(inp),str(out)],check=True,capture_output=True)
            expected=np.fromfile(out).reshape(n,33);assert np.isfinite(expected).all()
            arrays[name+'_config']=np.array([l,split,n],np.int32);arrays[name+'_expected']=expected
            for k,v in data.items():arrays[name+'_'+k]=v
            # The routine intentionally computes the derivative on the left
            # only. An unrelated right gradient must not alter either output.
            changed={**data,'right_gradient':data['right_gradient']+7.}
            inp.write_bytes(binary(l,split,changed));subprocess.run([str(exe),str(inp),str(out)],check=True,capture_output=True)
            assert np.array_equal(np.fromfile(out).reshape(n,33),expected)
            checks.append(dict(fixture=name,l=l,split=bool(split),queries=n,epsilon=list(signs[s]),input_sha256=hashlib.sha256(raw).hexdigest(),right_gradient_unused=True,passed=True))
    archive=output.with_suffix('.npz');np.savez_compressed(archive,**arrays);source=output.with_suffix('.F');source.write_text(wrapper)
    report=dict(scope=__doc__,passed=True,gpu_validation=None,checks=checks,arrays_file=archive.name,arrays_sha256=sha(archive),wrapper_file=source.name,wrapper_sha256=sha(source),
                literal_block_sha256=hashlib.sha256(block(upstream).encode()).hexdigest(),source_sha256=sha(upstream/'Src/nlefsm.f'),script_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(report,indent=2)+'\n');print('Pristine SOC projector oracle:',len(checks),'fixtures;',sum(c['queries'] for c in checks),'queries')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','build','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.upstream,a.build,a.output)
