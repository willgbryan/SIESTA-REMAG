#!/usr/bin/env python3
"""GPU initial normalization against retained pristine executable results."""
import argparse,ctypes as ct,hashlib,json
from pathlib import Path
import numpy as np
class Problem(ct.Structure):
    _fields_=[('capacity',ct.c_int),('spins',ct.c_int),('spinor',ct.c_int),('mix_first',ct.c_int),('target_charge',ct.c_double)]
class Result(ct.Structure):
    _fields_=[('charge',ct.c_double),('scale',ct.c_double)]+[(n,ct.c_int) for n in ('status','mix_first','note','reserved')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(reference,library,output):
    ref=json.loads(reference.read_text());assert ref['passed'];npz=reference.with_suffix('.npz')
    expected=[h for p,h in ref['evidence_sha256'].items() if Path(p).name==npz.name];assert len(expected)==1 and sha(npz)==expected[0]
    raw=np.load(npz);lib=ct.CDLL(str(library));call=lib.remag_check_initial_density
    call.argtypes=[ct.c_int]*5+[ct.POINTER(Problem),ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.POINTER(Result),ct.c_void_p]
    call.restype=ct.c_int;checks=[];saved={};output.parent.mkdir(parents=True,exist_ok=True)
    def execute(name,p,overlap,density,captured,inplace,dependency,status,want=None):
        capacity,spins=density.shape;overlap=np.ascontiguousarray(overlap);density=np.asfortranarray(density)
        out=np.empty_like(density,order='F');energy=np.empty((capacity,min(spins,4)),order='F');r=Result();graph=np.zeros(2,dtype='i4')
        code=call(capacity,spins,captured,inplace,dependency,ct.byref(p),overlap.ctypes.data,density.ctypes.data,out.ctypes.data,energy.ctypes.data,ct.byref(r),graph.ctypes.data)
        label=name+f'-graph{captured}-inplace{inplace}';saved[label+'__density']=out;saved[label+'__energy']=energy;saved[label+'__status']=np.array([r.status,r.mix_first,r.note],dtype='i4');saved[label+'__charge']=np.array([r.charge,r.scale])
        assert code==0 and r.status==status,(label,code,r.status,status)
        assert graph[1]==0 and graph[0]==(3 if captured else 0)
        if status:
            if inplace:np.testing.assert_array_equal(out,density)
            else:assert np.isnan(out).all()
            assert np.isnan(energy).all()
            error=None
        else:
            np.testing.assert_allclose([r.charge,r.scale],[want['charge'],want['scale']],atol=1e-10,rtol=1e-12)
            assert r.mix_first==want['effective_mix'] and bool(r.note)==want['note'],label
            np.testing.assert_allclose(out,raw[name+'__normalized'],atol=1e-10,rtol=1e-12);np.testing.assert_array_equal(energy,raw[name+'__energy'])
            error=float(np.max(abs(out-raw[name+'__normalized'])))
        checks.append(dict(name=label,status=r.status,graph_nodes=int(graph[0]),forbidden_nodes=int(graph[1]),max_density_error=error,passed=True))
    for c in ref['checks']:
        name=c['name'];density=raw[name+'__density'];overlap=raw[name+'__overlap'];p=Problem(c['capacity'],c['spins'],min(c['spins'],2),c['mix_first'],c['target'])
        for captured in (0,1):
            for inplace in (0,1):execute(name,p,overlap,density,captured,inplace,0,0,c)
    for spins in (1,2,4,8):
        capacity=5;density=np.ones((capacity,spins));overlap=np.ones(capacity)
        for name in ('dependency','bad-capacity','bad-spinor','bad-mix','nan-target','zero-target','zero-charge','nan-offdiagonal','overflow'):
            p=Problem(capacity,spins,min(spins,2),1,10.);d=density.copy();s=overlap.copy();dependency=0;status=1
            if name=='dependency':dependency=7;status=4
            elif name=='bad-capacity':p.capacity+=1
            elif name=='bad-spinor':p.spinor=0
            elif name=='bad-mix':p.mix_first=2
            elif name=='nan-target':p.target_charge=np.nan
            elif name=='zero-target':p.target_charge=0.
            elif name=='zero-charge':d[:]=0.;status=2
            elif name=='nan-offdiagonal':d[-1,-1]=np.nan;status=2
            elif name=='overflow':d[:]=1e300;s[:]=1e300;status=2
            for captured in (0,1):
                for inplace in (0,1):execute(f'fault-s{spins}-{name}',p,s,d,captured,inplace,dependency,status)
    data=output.with_suffix('.npz');np.savez_compressed(data,**saved)
    report=dict(passed=True,scope='Device initial normalization and first-mix decision only; no material qualification',checks=checks,
                evidence_sha256={str(p):sha(p) for p in (Path(__file__),reference,npz,library,data)})
    output.write_text(json.dumps(report,indent=2)+'\n');print('Initial density GPU checks:',len(checks),'passed')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.reference,a.library,a.output)
