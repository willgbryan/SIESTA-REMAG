"""Validate full material two-center derivative shadows and explicit traffic."""
import math

def record_errors(report,epochs,soc,pair_shadow=False,nonlocal_shadow=False,pair_geometry=False):
    errors=[];tables=report.get('tables',[]);rows=report.get('cycles',[])
    if report.get('complete') is not True or report.get('atol')!=1e-10 or report.get('rtol')!=1e-10:errors.append('Incomplete two-center shadow or changed tolerances')
    if len(tables)!=2 or [t.get('kind') for t in tables]!=[1,2]:errors.append('Missing or unordered S/T table owners')
    for t in tables:
        shape=t.get('shape',[])
        if len(shape)!=11 or any(type(v) is not int or v<0 for v in shape):
            errors.append('Invalid two-center table shape');continue
        nf,nr,nt,na,nb,b1,b2,lo1,hi1,lo2,hi2=shape
        if not (0<nf<=1000000 and 0<=nr<=1000000 and 0<=nt<=100000000 and na>0 and nb>0 and na*nb<=100000000 and
                1<=lo1<=hi1<=nf and 1<=lo2<=hi2<=nf):errors.append('Two-center dimensions or registry bounds invalid')
        staged=4*(nf+1+na*nb+1+2*nt)+8*(nf+2050*nr+nt)
        allocated=staged+(16 if nt==0 else 0)+(8 if nr==0 else 0)
        if staged>4*1024**3 or t.get('setup_bytes')!=staged or t.get('device_allocation_bytes')!=allocated:errors.append('Table shape-derived traffic differs')
        if t.get('uses')!=sum(r.get('table')==t['kind'] and r.get('complete') is True for r in rows):errors.append('Table lifetime consumer count differs')
        if any(not isinstance(t.get(k),(int,float)) or not math.isfinite(t[k]) for k in ('delta','pi')) or t.get('delta',0)<=0 or not 3.14<t.get('pi',0)<3.15:
            errors.append('Invalid two-center constants')
    expected=[(e,c) for e in epochs for c in (1,3 if soc else 2,4)]
    if not expected or [(r.get('epoch'),r.get('consumer')) for r in rows]!=expected:errors.append('Two-center shadow does not cover all geometry/force consumers')
    for r in rows:
        n=r.get('queries');executed=isinstance(n,int) and n>0
        pair=pair_shadow and r.get('consumer') in (1,4)
        nonlocal_force=nonlocal_shadow and r.get('consumer') in (2,3)
        if type(n) is not int or n<0:errors.append('Invalid two-center query count');continue
        if r.get('table')!=(2 if r.get('consumer')==1 else 1) or r.get('complete') is not True or r.get('passed') is not True or r.get('status')!=0:
            errors.append('Two-center consumer failed')
        wanted=dict(graph_nodes=(2 if executed else 0)+int(pair),setup_bytes=64*n+60 if executed or pair else 0,
                    device_allocation_bytes=96*n+60+(28 if n==0 else 0) if executed or pair else 0,
                    export_bytes=60 if executed or pair else 0,host_reference_bytes=32*n,compared=4*n,mismatches=0,nonfinite=0)
        if nonlocal_force:
            wanted.update(graph_nodes=(2 if n else 0)+2,setup_bytes=32*n+60,device_allocation_bytes=64*n+60+(16 if n==0 else 0),export_bytes=60)
        if pair and pair_geometry:
            wanted.update(setup_bytes=32*n+60,device_allocation_bytes=64*n+60+(16 if n==0 else 0))
        if any(r.get(k)!=v for k,v in wanted.items()):errors.append('Two-center full comparison or shape-derived traffic differs')
        for key in ('max_absolute_error','max_scaled_error'):
            values=r.get(key,[])
            if not isinstance(values,list) or len(values)!=2 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):
                errors.append('Invalid two-center comparison maxima')
            elif key=='max_scaled_error' and max(values)>1:errors.append('Two-center derivative tolerance exceeded')
    return errors

def traffic(report):
    return dict(host_to_device_bytes=report.get('radial_setup',{}).get('host_to_device_bytes',0)+sum(t.get('setup_bytes',0) for t in report.get('tables',[]))+sum(r.get('setup_bytes',0) for r in report.get('cycles',[])),
                device_to_host_bytes=report.get('radial_setup',{}).get('device_to_host_bytes',0)+sum(r.get('export_bytes',0) for r in report.get('cycles',[])))
