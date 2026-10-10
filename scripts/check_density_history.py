#!/usr/bin/env python3
"""Independent coordinate/history qualification; preserves raw inputs and outputs."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from scipy.linalg import lstsq

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = None
REPLAY = None
COMMAND = np.dtype([('generation','u8'),('distribution','u8'),('epoch','u8'),('capacity','i4'),('nsc','i4',(3,)),('mode','i4')], align=True)
RESULT = np.dtype([('status','i4'),('count','i4'),('rank','i4'),('fallback','i4'),('generation','u8'),('epoch','u8'),('last_push','u8')], align=True)
CPP = '''#include "density_history_math.hpp"
extern "C" void coefficients(int d,int n,int m,const double* x,const double* y,double* c,int* r){
 remag::HistoryMathWorkspace w{};auto s=remag::density_history_coefficients(d,n,m,x,y,c,w);
 r[0]=s.status;r[1]=s.rank;r[2]=s.fallback;
}
extern "C" int svd(int n,const double* a,const double* b,double* x,int* rank){
 remag::HistoryMathWorkspace w{};for(int i=0;i<n*n;i++)w.matrix[i]=a[i];for(int i=0;i<n;i++)w.rhs[i]=b[i];
 if(!remag::history_detail::svd(n,w,*rank))return 1;for(int i=0;i<n;i++)x[i]=w.solution[i];return 0;
}
'''

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def reference(folder):
    report=json.loads((folder/'coefficient-modes.json').read_text())
    assert report['passed']
    for name in ('coefficients.npz','coefficient-modes.npz'):
        expected=[v for k,v in report['evidence_sha256'].items() if Path(k).name==name]
        assert len(expected)==1 and sha(folder/name)==expected[0]
    return np.load(folder/'coefficients.npz'),np.load(folder/'coefficient-modes.npz')

def math_checks(folder,output):
    fixtures,oracle=reference(folder);checks=[];saved={}
    with tempfile.TemporaryDirectory(prefix='remag-history-math-') as name:
        tmp=Path(name);(tmp/'check.cpp').write_text(CPP)
        subprocess.run(['c++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-shared','-fPIC',
                        '-I'+str(ROOT/'native'),str(tmp/'check.cpp'),'-o',str(tmp/'check.so')],check=True)
        lib=ct.CDLL(str(tmp/'check.so'));f=lib.coefficients;f.argtypes=[ct.c_int]*3+[ct.c_void_p]*4
        def solve(x,y,mode,n=None):
            x=np.ascontiguousarray(x,dtype='f8');y=np.ascontiguousarray(y,dtype='f8');c=np.full(32,123.);info=np.full(3,-1,dtype='i4')
            f(y.size,x.shape[0] if n is None else n,mode,x.ctypes.data,y.ctypes.data,c.ctypes.data,info.ctypes.data)
            return c,info
        for label in ('single','random3','random8','identical','affine','small'):
            x=np.ascontiguousarray(fixtures[label+'_history'].transpose(2,1,0));y=np.ascontiguousarray(fixtures[label+'_x0'].T)
            for mode in range(4):
                name=f'coeff-mode{mode}-{label}';actual,info=solve(x,y,mode);expected=oracle[name+'_actual']
                assert info[0]==0,(name,info)
                np.testing.assert_allclose(actual[:len(expected)],expected,atol=1e-10,rtol=0)
                np.testing.assert_array_equal(actual[len(expected):],123.)
                saved[name+'_actual']=actual[:len(expected)];saved[name+'_reference']=expected;saved[name+'_info']=info
                checks.append(dict(name=name,passed=True,max_absolute_error=float(max(abs(actual[:len(expected)]-expected)))))
        for mode in range(4):
            for name,x,y,n,m in [('nan-history',np.array([[np.nan,0.,1.],[1.,0.,0.]]),np.zeros(3),2,mode),
                    ('nan-current',np.ones((2,3)),np.full(3,np.nan),2,mode),
                    ('overflow',np.full((2,3),1e300),np.full(3,-1e300),2,mode),
                    ('zero-depth',np.ones((1,3)),np.zeros(3),0,mode),
                    ('excess-depth',np.ones((33,3)),np.zeros(3),33,mode),
                    ('bad-mode',np.ones((2,3)),np.zeros(3),2,4)]:
                actual,info=solve(x,y,m,n)
                # Original mean-centered identical coordinates legitimately
                # select the most recent entry; no overflow is generated.
                if name=='overflow' and mode==0:assert info[0]==0
                else:assert info[0]!=0,(name,mode,info);np.testing.assert_array_equal(actual,123.)
                checks.append(dict(name=f'{name}-mode{mode}',status=int(info[0]),passed=True))
        svd=lib.svd;svd.argtypes=[ct.c_int]+[ct.c_void_p]*4;svd.restype=ct.c_int
        rng=np.random.default_rng(98451)
        for n in (2,5,16,33):
            q=np.linalg.qr(rng.normal(size=(n,n)))[0]
            for scale in (1e-120,1.,1e120):
                # Place eigenvalues on both sides of the upstream cutoff,
                # avoiding a rounding-ambiguous exact equality.
                e=np.geomspace(1.,1e-8,n);e[1:]*=.7;e[::2]*=-1
                a=np.ascontiguousarray(((q*e)@q.T)*scale);a=np.ascontiguousarray((a+a.T)*.5)
                b=rng.normal(size=n);x=np.zeros(n);rank=np.zeros(1,dtype='i4')
                assert svd(n,a.ctypes.data,b.ctypes.data,x.ctypes.data,rank.ctypes.data)==0
                expected=lstsq(a/scale,b,cond=1e-6,lapack_driver="gelss")[0]
                np.testing.assert_allclose(x*scale,expected,atol=2e-6,rtol=2e-9)
                assert rank[0]==sum(abs(e)>1e-6)
                name=f'svd-{n}-{scale}';saved[name+'_matrix']=a;saved[name+'_rhs']=b;saved[name+'_actual']=x;saved[name+'_expected_scaled']=expected
                checks.append(dict(name=name,rank=int(rank[0]),passed=True,max_scaled_error=float(max(abs(x*scale-expected)))))
    return checks,saved

class Sequence:
    def __init__(self,rows,orbitals,atoms,spins,capacity,depth,initial_capacity=None):
        self.config=np.array([rows,orbitals,atoms,spins,capacity,depth],dtype='i4');self.steps=[]
        self.initial_capacity=capacity if initial_capacity is None else initial_capacity
    def add(self,op,epoch=0,nsc=(1,1,1),pattern=None,values=None,xyz=None,mode=2,status=0,
            generation=1,distribution=1,converged=1,capacity=None,label=''):
        rows,orb,atoms,spins,bound,depth=map(int,self.config);capacity=bound if capacity is None else capacity
        num=np.zeros(rows,dtype='i4');ptr=np.zeros(rows,dtype='i4');col=np.full(bound,-1,dtype='i4');k=0
        if pattern is not None:
            for row,columns in enumerate(pattern):ptr[row]=k;num[row]=len(columns);col[k:k+len(columns)]=columns;k+=len(columns)
        d=np.full(bound*spins,np.nan)
        if values is not None:d[:np.size(values)]=np.asarray(values).ravel(order='F')
        x=np.zeros(3*atoms) if xyz is None else np.asarray(xyz).reshape(-1)
        p=np.zeros(1,dtype=COMMAND);p[0]=(generation,distribution,epoch,capacity,nsc,mode)
        self.steps.append(dict(op=op,command=p[0],converged=converged,num=num,ptr=ptr,col=col,d=d,xyz=x,status=status,label=label))
        return self
    def run(self,lib,captured,name,reference_coeff=None):
        rows,orb,atoms,spins,bound,depth=map(int,self.config);steps=len(self.steps);slot=bound*spins;dims=3*atoms
        arrays={key:np.ascontiguousarray([s[key] for s in self.steps],dtype=dtype) for key,dtype in
                [('op','i4'),('command',COMMAND),('converged','i4'),('num','i4'),('ptr','i4'),('col','i4'),('d','f8'),('xyz','f8')]}
        arrays.update(output=np.zeros((steps,slot)),coeff=np.zeros((steps,depth)),history=np.zeros((steps,depth,spins,bound)),
            positions=np.zeros((steps,depth,dims)),topology=np.zeros((steps,2*rows+bound),dtype='i4'),
            result=np.zeros(steps,dtype=RESULT),snapshot=np.zeros(steps,dtype=RESULT),allocations=np.zeros(steps,dtype='i4'),report=np.zeros(3,dtype=np.uintp))
        args=[self.config.ctypes.data,steps,captured,self.initial_capacity]+[arrays[k].ctypes.data for k in
            ('op','command','converged','num','ptr','col','d','xyz','output','coeff','history','positions','topology','result','snapshot','allocations','report')]
        if REPLAY is None:
            code=lib(*args)
        else:
            raw=np.load(REPLAY/(name+".npz"))
            np.testing.assert_array_equal(raw['config'],self.config)
            assert int(raw['initial_capacity'])==self.initial_capacity
            for key in ('op','command','converged','num','ptr','col','d','xyz'):
                np.testing.assert_array_equal(raw[key],arrays[key],err_msg=f'{name} input {key}')
            for key in ('output','coeff','history','positions','topology','result','snapshot','allocations','report'):
                assert raw[key].shape==arrays[key].shape and raw[key].dtype==arrays[key].dtype
                arrays[key]=raw[key]
            code=0
        if EVIDENCE is not None:
            EVIDENCE.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(EVIDENCE/(name+".npz"),config=self.config,initial_capacity=self.initial_capacity,**arrays)
        assert code==0,(name,code)
        assert arrays['report'][1]==0
        if captured:assert arrays['report'][0]>0
        history=[];positions=[];old_pattern=[[] for _ in range(rows)];old_nsc=(1,1,1);gen=epoch=last=0;old_num=np.zeros(rows,dtype='i4');old_ptr=old_num.copy();old_cols=np.zeros(bound,dtype='i4')
        max_error=0.;rejected=0;allocated=self.initial_capacity
        for i,s in enumerate(self.steps):
            result=arrays['result'][i];p=s['command'];op=s['op'];status=int(result['status']);assert status==s['status'],(name,i,s['label'],status,s['status'])
            if status:rejected+=1
            else:
                if op==0:history=[];positions=[];gen=int(p['generation']);epoch=last=0;old_pattern=[[] for _ in range(rows)];old_num.fill(0);old_ptr.fill(0);old_cols.fill(0)
                elif op==1:
                    target=[list(map(int,s['col'][start:start+count])) for start,count in zip(s['ptr'],s['num'])]
                    # Physical image-coordinate keys; no dependence on the GPU's translated-column arithmetic.
                    def key(column,nsc):
                        unit=(column-1)%orb;image=(column-1)//orb;coord=[]
                        for n in nsc:
                            c=image%int(n);image//=int(n);coord.append(c if c<=int(n)//2 else c-int(n))
                        return unit,*coord
                    mapped=[]
                    for h in history:
                        new=np.zeros((spins,bound))
                        for row,cols in enumerate(old_pattern):
                            lookup={key(column,old_nsc):h[:,int(old_ptr[row])+j].copy() for j,column in enumerate(cols)}
                            for j,column in enumerate(target[row]):new[:,int(s['ptr'][row])+j]=lookup.get(key(column,p['nsc']),np.zeros(spins))
                        mapped.append(new)
                    history=mapped;old_pattern=target;old_nsc=tuple(p['nsc']);epoch=int(p['epoch']);old_num=s['num'].copy();old_ptr=s['ptr'].copy();old_cols.fill(0)
                    for start,count in zip(old_ptr,old_num):old_cols[start:start+count]=s['col'][start:start+count]
                elif op==2:
                    last=epoch
                    if depth:
                        new=np.zeros((spins,bound));d=s['d'][:int(p['capacity'])*spins].reshape(spins,int(p['capacity']))
                        for start,count in zip(old_ptr,old_num):new[:,start:start+count]=d[:,start:start+count]
                        history=(history+[new])[-depth:];positions=(positions+[s['xyz'].copy()])[-depth:]
                elif op==3:
                    c=arrays['coeff'][i,:len(history)]
                    if reference_coeff is not None:
                        np.testing.assert_allclose(c,reference_coeff,atol=1e-10,rtol=0)
                        max_error=max(max_error,float(max(abs(c-reference_coeff))))
                    elif len(history)==1:np.testing.assert_array_equal(c,[1.])
                    else:
                        x=np.array(positions)-s['xyz'];m=len(history);matrix=np.block([[x@x.T,np.ones((m,1))],[np.ones((1,m)),np.zeros((1,1))]])
                        rhs=np.zeros(m+1);rhs[-1]=1;expected=np.linalg.lstsq(matrix,rhs,rcond=1e-6)[0][:-1]
                        np.testing.assert_allclose(c,expected,atol=1e-9,rtol=1e-10)
                    expected=sum(v*h for v,h in zip(c,history))[:,:int(p['capacity'])].ravel()
                    actual=arrays['output'][i,:int(p['capacity'])*spins]
                    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-12);np.testing.assert_array_equal(arrays['coeff'][i,len(history):],0.)
                elif op==4:
                    allocated=max(allocated,int(p['capacity']))
            assert int(arrays['allocations'][i])==allocated,(name,i,'allocation bound changed incorrectly')
            if op!=3 or status:assert np.isnan(arrays['output'][i]).all() and np.isnan(arrays['coeff'][i]).all(),(name,i,'modified failed output')
            expected_history=np.zeros((depth,spins,bound));expected_positions=np.zeros((depth,dims))
            if history:expected_history[:len(history)]=history;expected_positions[:len(history)]=positions
            np.testing.assert_array_equal(arrays['history'][i],expected_history,err_msg=f'{name} step{i} history')
            np.testing.assert_array_equal(arrays['positions'][i],expected_positions)
            np.testing.assert_array_equal(arrays['topology'][i],np.concatenate((old_num,old_ptr,old_cols)))
            for field,want in [('generation',gen),('epoch',epoch),('last_push',last),('count',len(history))]:assert arrays['snapshot'][i][field]==want,(name,i,field)
        return dict(name=name,steps=steps,rejected=rejected,captured=bool(captured),capture_mode=captured,
            initial_capacity=self.initial_capacity,final_capacity=allocated,growth_requests=sum(s['op']==4 for s in self.steps),
            graph_nodes=int(arrays['report'][0]),forbidden_nodes=int(arrays['report'][1]),workspace_bytes=int(arrays['report'][2]),max_coefficient_error=max_error,passed=True),arrays

def gpu_checks(folder,library):
    fixtures,oracle=reference(folder);f=None;checks=[];saved={}
    if REPLAY is None:
        lib=ct.CDLL(str(library));f=lib.remag_check_density_history
        f.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int]+[ct.c_void_p]*17;f.restype=ct.c_int
    def run(seq,name,ref=None,capture_modes=(0,1)):
        for captured in capture_modes:
            label=name+{0:'-direct',1:'-graph',2:'-segmented-graph'}[captured];check,arrays=seq.run(f,captured,label,ref)
            checks.append(check);saved.update({label+'__'+k:v for k,v in arrays.items()});print(label,'passed',flush=True)
    # All modes against actual pinned routine coefficients, including its
    # non-unit-sum experimental behavior. Three-entry and larger histories.
    for label in ('single','random3','random8','identical','affine','small'):
        x=fixtures[label+'_history'].transpose(2,1,0).reshape(fixtures[label+'_history'].shape[2],-1);y=fixtures[label+'_x0'].T.ravel();n=len(x)
        for mode in range(4):
            seq=Sequence(2,2,y.size//3,4,5,n);seq.add(0)
            for j,pos in enumerate(x):
                seq.add(1,j+1,pattern=[[1,2],[1,2]],capacity=4)
                seq.add(2,j+1,values=np.arange(16).reshape(4,4)+j+.25,xyz=pos,capacity=4)
            seq.add(1,n+1,pattern=[[1,2],[1,2]],capacity=4);seq.add(3,n+1,xyz=y,mode=mode,capacity=4)
            run(seq,f'coefficient-{label}-{mode}',oracle[f'coeff-mode{mode}-{label}_actual'])
    # Multiple saved entries, five pushes into three slots, followed by two
    # destructive topology transitions. Changed cells use independent ownership
    # semantics, not the known-broken shared-pattern upstream stack.
    for spins in (1,2,4,8):
        for name,old,new in [('expand',(1,1,1),(3,3,1)),('contract',(5,3,1),(3,1,1)),('mixed',(3,1,3),(3,3,1)),('unchanged',(3,3,3),(3,3,3))]:
            degree=2*int(np.prod(old));target=2*int(np.prod(new));bound=max(2*degree+1,2*target+1)
            seq=Sequence(2,2,2,spins,bound,3);seq.add(0)
            pattern=[list(range(1,degree+1)),list(range(1,degree+1))]
            for epoch in range(1,6):
                seq.add(1,epoch,old,pattern)
                seq.add(2,epoch,old,values=np.arange(bound*spins).reshape(bound,spins,order='F')+epoch+.25,xyz=np.arange(6)+epoch*.05)
            seq.add(1,6,new,[list(range(1,target+1)),list(range(1,target+1))]);seq.add(3,6,new,xyz=np.arange(6)+.4)
            seq.add(1,7,old,pattern);seq.add(3,7,old,xyz=np.arange(6)+.5)
            run(seq,f'cell-{name}-s{spins}')
        for depth in (0,1,3):
            seq=Sequence(3,3,2,spins,8,depth);seq.add(0)
            initial=[[1,1,3],[],[2]];d=np.arange(8*spins).reshape(8,spins,order='F')+.25
            seq.add(1,1,pattern=initial);seq.add(3,1,status=6)
            seq.add(2,1,values=d,xyz=np.arange(6),converged=0,status=5)
            seq.add(2,1,values=d,xyz=np.full(6,np.nan),status=2)
            bad=d.copy();bad[0,0]=np.nan;seq.add(2,1,values=bad,status=2)
            seq.add(2,1,values=d,xyz=np.arange(6));seq.add(2,1,values=d,status=5)
            seq.add(1,1,pattern=initial,status=4)
            seq.add(1,2,pattern=[[1,4],[],[]],status=1)
            seq.add(1,2,nsc=(2,1,1),pattern=initial,status=1)
            seq.add(1,2,pattern=initial,generation=2,status=4)
            seq.add(1,2,pattern=initial,distribution=2,status=4)
            seq.add(1,2,pattern=[[1,2],[2],[2]],capacity=5)
            seq.add(3,2,xyz=np.arange(6),capacity=5,status=6 if not depth else 0)
            seq.add(3,2,xyz=np.full(6,np.nan),capacity=5,status=6 if not depth else 2)
            seq.add(1,3,pattern=initial)
            seq.add(3,3,xyz=np.arange(6),status=6 if not depth else 0)
            seq.add(0,generation=1,status=4);seq.add(0,generation=2,distribution=2)
            seq.add(1,1,pattern=initial,generation=1,status=4)
            seq.add(1,1,pattern=initial,generation=2,distribution=2)
            seq.add(2,1,values=d,xyz=np.arange(6),generation=2,distribution=2)
            run(seq,f'ownership-depth{depth}-s{spins}')
        # Growth must preserve every entry, live topology, epoch and last-push
        # guard. The independent snapshots use a fixed padded output stride.
        for depth in (0,1,3):
            seq=Sequence(3,3,2,spins,48,depth,initial_capacity=4)
            seq.add(4,capacity=6);seq.add(0)
            initial=[[1,1,3],[],[2]]
            for epoch in range(1,6):
                seq.add(1,epoch,pattern=initial,capacity=6)
                values=np.arange(6*spins).reshape(6,spins,order='F')+epoch+.25
                seq.add(2,epoch,values=values,xyz=np.arange(6)+epoch*.05,capacity=6)
                if epoch==2:
                    seq.add(4,capacity=10)
                    seq.add(2,epoch,values=values,capacity=6,status=5)
            seq.add(4,capacity=12);seq.add(4,capacity=6)
            for invalid in (0,-1,2**31//8):seq.add(4,capacity=invalid,status=1)
            expanded=[[1,4,2],[3,7],[2,8]]
            seq.add(1,6,nsc=(3,1,1),pattern=expanded,capacity=16,status=1)
            seq.add(4,capacity=32)
            seq.add(1,6,nsc=(3,1,1),pattern=expanded,capacity=16)
            seq.add(3,6,nsc=(3,1,1),xyz=np.arange(6)+.4,capacity=16,status=6 if not depth else 0)
            seq.add(2,6,nsc=(3,1,1),values=np.arange(16*spins).reshape(16,spins,order='F')+.75,xyz=np.arange(6)+.4,capacity=16)
            seq.add(1,7,pattern=initial,capacity=7)
            seq.add(4,capacity=48)
            seq.add(3,7,xyz=np.arange(6)+.5,capacity=7,status=6 if not depth else 0)
            seq.add(0,generation=2,distribution=2)
            seq.add(1,1,nsc=(3,1,1),pattern=expanded,capacity=40,generation=2,distribution=2)
            seq.add(2,1,nsc=(3,1,1),values=np.arange(40*spins).reshape(40,spins,order='F')+.125,capacity=40,generation=2,distribution=2)
            seq.add(4,capacity=16)
            run(seq,f'growth-depth{depth}-s{spins}',capture_modes=(0,2))
        seq=Sequence(3,3,2,spins,12,3,initial_capacity=6);seq.add(0)
        seq.add(1,1,pattern=[[1,1,3],[],[2]],capacity=6)
        seq.add(2,1,values=np.arange(6*spins).reshape(6,spins,order='F'),capacity=6)
        seq.add(4,capacity=12,status=900) # cudaErrorStreamCaptureUnsupported
        seq.add(1,2,pattern=[[1,2],[2],[2]],capacity=12,status=1)
        seq.add(1,2,pattern=[[1,2],[2],[2]],capacity=6)
        seq.add(3,2,capacity=6)
        run(seq,f'growth-reject-capture-s{spins}',capture_modes=(1,))
    return checks,saved

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,default=ROOT/'results/preflight-density-history-v84')
    p.add_argument('--library',type=Path);p.add_argument('--replay',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.library and args.replay:p.error('Choose GPU execution or raw evidence replay')
    REPLAY=args.replay
    EVIDENCE=args.output.parent/'raw-sequences' if args.library else None
    checks,arrays=gpu_checks(args.reference,args.library) if args.library or args.replay else math_checks(args.reference,args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True);npz=args.output.with_suffix('.npz');np.savez_compressed(npz,**arrays)
    sources=[Path(__file__),*ROOT.joinpath('native').glob('*density_history*'),args.reference/'coefficient-modes.json',args.reference/'coefficient-modes.npz',args.reference/'coefficients.npz',npz]
    if args.library:sources.append(args.library)
    if args.replay:sources.extend(sorted(args.replay.glob('*.npz')))
    report=dict(passed=True,scope='Isolated density-history component, not SIESTA integration or full material/transfer qualification.',gpu_validated=bool(args.library),raw_gpu_evidence_rechecked=bool(args.replay),checks=checks,evidence_sha256={str(path):sha(path) for path in sources})
    args.output.write_text(json.dumps(report,indent=2)+'\n');print('Density history checks:',len(checks),'passed')
