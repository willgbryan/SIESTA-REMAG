"""Compare retained material H0 matrices and audit pre-SCF integral lifetimes."""
import hashlib
import math
from pathlib import Path
import numpy as np


def audit(report,folder,epochs,geometry,residency,*,production=False,cycle_records=None,poisoned=False,compact_queries=False):
    errors=[];checks=[];rows=report.get('cycles',[]);owners=geometry.get('cycles',[])
    if report.get('schema')!=1 or not rows or [r.get('epoch') for r in rows]!=epochs or [r.get('epoch') for r in owners]!=epochs:
        return dict(passed=False,errors=['H0 shadow does not cover every electronic/force geometry'],checks=[])
    if production and (cycle_records is None or [r.get('epoch') for r in cycle_records]!=epochs):
        return dict(passed=False,errors=['Resident H0 SCF lease coverage is missing'],checks=[])
    if poisoned and not production:return dict(passed=False,errors=["Host poisoning requires production H0"],checks=[])
    if compact_queries and not production:return dict(passed=False,errors=['Compact queries require production H0'],checks=[])
    totals=dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0)
    for index,(r,owner) in enumerate(zip(rows,owners),1):
        issues=[]
        dimensions=[r.get(k) for k in ('rows','columns','capacity','consumer','split','queries','local_neighbors')]
        if any(type(v) is not int for v in dimensions):
            errors.append('Invalid H0 dimensions');continue
        nr,no,cap,consumer,split,queries,local=dimensions;soc=consumer==3
        if not (0<nr<=no and 0<cap and consumer in (2,3) and split in (0,1) and queries>=0 and 0<=local<=owner.get('neighbors',-1)):
            errors.append('H0 dimensions outside supported geometry');continue
        if any(r.get(k)!=owner.get(k) for k in ('rows','columns','capacity','consumer','queries','epoch')):
            issues.append('H0 and force geometry differ')
        if type(r.get('preparation_epoch')) is not int or not 0<=r['preparation_epoch']<r['epoch']:
            issues.append('Invalid H0 preparation epoch')
        values=cap*(8+64*soc);staged=4*(nr+1+local)+(0 if production else values)+4
        query_nodes=0
        if r.get('compact_queries',False) is not compact_queries or owner.get('compact_queries',False) is not compact_queries:
            issues.append('Compact query mode differs from requested execution')
        if compact_queries:
            centers=owner.get('centers')
            if type(centers) is not int or centers<0:
                issues.append('Compact center coverage missing')
            else:query_nodes=(1+bool(owner['neighbors'])) if centers else 0
        if r.get('query_expansion_nodes',0)!=query_nodes:issues.append('Compact query graph coverage differs')
        expected=dict(graph_nodes=(3 if queries else 2)+query_nodes,status=0,setup_bytes=staged,
                      device_allocation_bytes=staged+(values if production else 0)+(4 if local==0 else 0)+8*max(1,4*queries),
                      export_bytes=4 if production else values+4,raw_bytes=0 if production else 3*values,derivative_upload_bytes=0,integral_download_bytes=0)
        if any(r.get(k)!=v for k,v in expected.items()):issues.append('H0 graph/storage/transfer inventory differs')
        if r.get('production',False) is not production:issues.append('H0 requested mode differs from native owner')
        if any(r.get(k) is not True for k in ('passed','bound','force_borrowed','complete')):
            issues.append('H0 comparison or retained integral lifetime incomplete')
        if r.get('poison_requested',False) is not poisoned or r.get('host_poisoned',False) is not poisoned:
            issues.append('H0 host poisoning control is incomplete or unexpected')
        if production:
            scalar_error=soc_error=digest=None;filename=None
            if (r.get('compared') is not False or r.get('raw_file')!='' or
                any(r.get(k) is not True for k in ('scf_borrowed','scf_released')) or
                any(r.get(k) is not None for k in ('scalar_max_error','soc_max_error'))):
                issues.append('Resident H0 comparison/export/SCF lifetime mismatch')
            if cycle_records[index-1].get('nonlocal_h0')!=dict(device=True,retained_bytes=values):
                issues.append('Resident H0 SCF matrix lease size differs')
            if list(Path(folder).glob('remag-nonlocal-h0-*.bin')):
                issues.append('Unexpected resident H0 matrix evidence export')
        else:
            if r.get('compared') is not True:issues.append('H0 shadow comparison is missing')
            filename=f'remag-nonlocal-h0-{index:04d}.bin';path=Path(folder)/filename
            scalar_error=soc_error=None;digest=None
            if r.get('raw_file')!=filename or not path.is_file() or path.stat().st_size!=3*values:
                issues.append('H0 raw matrix evidence missing or wrong size')
            else:
                data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
                h=np.frombuffer(data,dtype='<f8',count=3*cap).reshape(3,cap)
                finite=bool(np.isfinite(h).all())
                if not finite:issues.append('Nonfinite H0 matrix evidence')
                else:
                    error=np.abs(h[2]-h[1]);scalar_error=float(error.max())
                    if not np.all(error<=1e-12+1e-12*np.abs(h[1])):issues.append('GPU scalar H0 differs from upstream')
                soc_error=0.
                if soc:
                    z=np.frombuffer(data,dtype='<c16',count=12*cap,offset=24*cap).reshape(3,4*cap)
                    if not np.isfinite(z).all():issues.append('Nonfinite SOC H0 evidence');soc_error=None
                    else:
                        error=np.abs(z[2]-z[1]);soc_error=float(error.max())
                        if not np.all(error<=1e-12+1e-12*np.abs(z[1])):issues.append('GPU SOC H0 differs from upstream')
                for key,value in (('scalar_max_error',scalar_error),('soc_max_error',soc_error)):
                    reported=r.get(key)
                    if value is None or type(reported) not in (int,float) or not math.isfinite(reported) or not math.isclose(reported,value,rel_tol=1e-10,abs_tol=1e-18):
                        issues.append('H0 recorded error differs from raw matrices: '+key)
        totals['host_to_device_bytes']+=staged;totals['device_to_host_bytes']+=4 if production else values+4
        totals['copy_calls']+=(3 if production else 5+2*soc)+bool(local)
        checks.append(dict(epoch=r['epoch'],passed=not issues,errors=issues,scalar_max_error=scalar_error,
                           soc_max_error=soc_error,raw_file=filename,raw_sha256=digest))
        errors.extend(issues)
    actual=dict.fromkeys(totals,0)
    for interval in residency.get('intervals',[]):
        counter=interval.get('boundaries',{}).get('nonlocal_h0' if production else 'nonlocal_h0_shadow',{})
        if counter and interval.get('phase')!='initialization':errors.append('H0 shadow transfer outside initialization')
        for key in actual:actual[key]+=counter.get(key,0)
    if actual!=totals:errors.append('H0 transfers do not match raw phase accounting')
    return dict(passed=not errors,errors=errors,checks=checks,traffic=totals,
                production_h0=production,integrals_retained_through_forces=True,strict_residency_verified=False)
