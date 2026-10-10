#!/usr/bin/env python3
"""Validate shared CPU/device Pulay coefficient algebra on the CPU only."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

CPP='''#include "pulay_math.hpp"
extern "C" void coefficients(int n,const double* a,const double* r,double condition,double* c,int* info) {
 auto result=remag::pulay_coefficients(n,a,r,condition,c);
 info[0]=result.status;info[1]=result.method;info[2]=result.rank;
}
extern "C" int pseudoinverse(int n,const double* a,double condition,double* b,int* rank) {
 return remag::pulay_detail::pseudoinverse(n,a,condition,b,*rank)?0:1;
}
'''


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run(oracle,output):
    root=Path(__file__).resolve().parents[1]
    reference=json.loads(oracle.read_text());assert reference['passed']
    array_path=oracle.parent/reference['arrays_file'];assert sha(array_path)==reference['arrays_sha256']
    arrays=np.load(array_path)
    report=dict(gpu_validation=None,script_sha256=sha(Path(__file__)),
                pulay_math_sha256=sha(root/'native/pulay_math.hpp'),oracle_sha256=sha(oracle),
                oracle_arrays_sha256=sha(array_path),sequences=[],pseudoinverse=[],invalid=[])
    with tempfile.TemporaryDirectory(prefix='remag-mixing-math-') as tmp:
        tmp=Path(tmp);source=tmp/'check.cpp';source.write_text(CPP);library=tmp/'check.so'
        subprocess.run(['c++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror',
                        '-shared','-fPIC','-I'+str(root/'native'),str(source),'-o',str(library)],check=True)
        lib=ct.CDLL(str(library));candidate=lib.coefficients
        candidate.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_double,ct.c_void_p,ct.c_void_p]
        candidate.restype=None
        def solve(gram,projection,condition=1.e-8,n=None):
            gram=np.ascontiguousarray(gram);projection=np.ascontiguousarray(projection)
            c=np.full(16,123.);info=np.full(3,-1,dtype=np.int32)
            candidate(len(projection) if n is None else n,gram.ctypes.data,projection.ctypes.data,condition,c.ctypes.data,info.ctypes.data)
            return c,info
        for record in reference['checks']:
            label=record['fixture'];dx=[];df=[];last_f=last_out=None;actual=[];methods=[];ranks=[]
            x=arrays[label+'_input'];f=arrays[label+'_residual'];history=record['history'];weight=record['weight']
            for cycle,(xin,residual) in enumerate(zip(x,f),1):
                if cycle==record['reset_before']: dx=[];df=[];last_f=last_out=None
                if last_f is not None:
                    dx.append(last_f-last_out+xin);df.append(residual-last_f)
                    dx=dx[-(history-1):];df=df[-(history-1):]
                value=xin+(weight if df else record['linear_weight'])*residual
                if df:
                    differences=np.array(df);gram=differences@differences.T
                    c,info=solve(gram,differences@residual)
                    assert info[0]==0,(label,cycle,info)
                    methods.append(int(info[1]));ranks.append(int(info[2]))
                    for coeff,d,r in zip(c,dx,df): value+=coeff*(d+weight*r)
                else: methods.append(0);ranks.append(0)
                actual.append(value);last_f=residual.copy();last_out=xin+residual
            expected=arrays[label+'_mixed'];actual=np.array(actual)
            np.testing.assert_allclose(actual,expected,atol=1.e-9,rtol=1.e-10)
            if label=='singular_history_svd': assert methods.count(3)==record['upstream_svd_fallback_count']
            report['sequences'].append(dict(fixture=label,cycles=len(x),methods=methods,ranks=ranks,
                                           max_absolute_error=float(np.max(np.abs(actual-expected))),passed=True))
        inverse=lib.pseudoinverse
        inverse.argtypes=[ct.c_int,ct.c_void_p,ct.c_double,ct.c_void_p,ct.POINTER(ct.c_int)];inverse.restype=ct.c_int
        rng=np.random.default_rng(1832)
        for n in (2,3,5,7,11,15):
            q=np.linalg.qr(rng.normal(size=(n,n)))[0]
            for rank in (1,n):
                eigenvalues=np.zeros(n);eigenvalues[:rank]=np.geomspace(1.,.001,rank)
                unscaled=(q*eigenvalues)@q.T;unscaled=(unscaled+unscaled.T)*.5
                expected=np.linalg.pinv(unscaled,rcond=1.e-8)
                for scale in (1.e-25,1.,1.e25):
                    a=np.ascontiguousarray(unscaled*scale);b=np.zeros_like(a);computed_rank=ct.c_int(-1)
                    assert inverse(n,a.ctypes.data,1.e-8,b.ctypes.data,ct.byref(computed_rank))==0
                    assert computed_rank.value==rank
                    np.testing.assert_allclose(b*scale,expected,atol=1.e-9,rtol=1.e-10)
                    report['pseudoinverse'].append(dict(size=n,rank=rank,scale=scale,
                        max_scaled_absolute_error=float(np.max(np.abs(b*scale-expected))),passed=True))
        for label,gram,rhs,condition,n,status in (
            ('zero_single_history',np.zeros((1,1)),np.ones(1),1.e-8,None,2),
            ('nan_gram',np.array([[np.nan]]),np.ones(1),1.e-8,None,2),
            ('nan_projection',np.eye(2),np.array([np.nan,1.]),1.e-8,None,2),
            ('asymmetric',np.array([[1.,.1],[.2,1.]]),np.ones(2),1.e-8,None,1),
            ('negative_cutoff',np.eye(2),np.ones(2),-1.,None,1),
            ('nan_cutoff',np.eye(2),np.ones(2),np.nan,None,1),
            ('excess_history',np.eye(16),np.ones(16),1.e-8,16,1),
            ('negative_history',np.eye(1),np.ones(1),1.e-8,-1,1)):
            c,info=solve(gram,rhs,condition,n)
            assert info[0]==status,(label,info)
            np.testing.assert_array_equal(c,123.)
            report['invalid'].append(dict(fixture=label,status=status,passed=True))
    report['passed']=True;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"Pulay algebra: {len(report['sequences'])} sequences, {len(report['pseudoinverse'])} pseudoinverses, {len(report['invalid'])} invalid checks; passed=True; GPU validation=None")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle',type=Path,default=Path('results/local-mixing-reference-20261006/reference.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.oracle,args.output)
