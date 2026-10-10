"""Validate compact nonlocal shadows against full scalar/complex CPU outputs."""
import math

def record_errors(report,integrals,forces,soc):
    errors=[];rows=report.get('cycles',[]);consumer=3 if soc else 2
    expected=[(f['epoch'],consumer) for f in forces]
    if not expected or report.get('complete') is not True or [(r.get('epoch'),r.get('consumer')) for r in rows]!=expected:
        errors.append('Nonlocal shadow does not cover every geometry')
    for r in rows:
        f=next((f for f in forces if f['epoch']==r['epoch']),{});shape=f.get('grid_shape',[])
        source=next((s for s in integrals.get('cycles',[]) if (s['epoch'],s['consumer'])==(r['epoch'],consumer)),{})
        dims=[r.get(k) for k in ('atoms','rows','columns','capacity','components','queries','centers','neighbors','blocks')]
        if any(type(v) is not int or v<0 for v in dims):errors.append('Invalid nonlocal dimensions');continue
        na,nr,nc,cap,components,n,centers,neighbors,blocks=dims;size=3*na+18
        if len(shape)!=6 or [nc,nr,cap]!=[shape[0],shape[3],shape[4]] or na!=f.get('atoms') or components!=f.get('spins') or (soc and components!=8):errors.append('Nonlocal retained density shape differs')
        geometry=8*nr+4*cap+28*centers+16*neighbors+24*blocks+32*n
        allocated=geometry+8*size+8*cap+(28 if not centers else 0)+(16 if not neighbors else 0)+(24 if not blocks else 0)+(12 if not n else 0)
        wanted=dict(status=0,passed=True,complete=True,electronic_upload_bytes=0,graph_nodes=2,geometry_bytes=geometry,setup_bytes=8*size,device_allocation_bytes=allocated,export_bytes=8*size)
        if any(r.get(k)!=v for k,v in wanted.items()):errors.append('Nonlocal execution or shape-derived traffic differs')
        if n!=source.get('queries') or source.get('graph_nodes')!=(4 if n else 2):errors.append('Nonlocal integral execution differs')
        if not isinstance(r.get('volume'),(float,int)) or not math.isfinite(r['volume']) or r['volume']<=0:errors.append('Invalid nonlocal volume')
        arrays=[r.get(k,[]) for k in ('initial','reference','actual')]
        if any(len(a)!=size or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in a) for a in arrays):errors.append('Missing/nonfinite nonlocal outputs');continue
        initial,reference,actual=arrays
        if initial[:9]!=[0]*9 or (not soc and (reference[1:9]!=[0]*8 or actual[1:9]!=[0]*8)):errors.append('Nonlocal energy convention differs')
        for i,(a,b) in enumerate(zip(actual,reference)):
            tolerance=1e-10+1e-10*abs(b) if i<9 else 1e-10 if i<9+3*na else 1e-12
            if abs(a-b)>tolerance:errors.append('Nonlocal full contribution tolerance exceeded');break
    return errors

def traffic(report):
    rows=report.get('cycles',[])
    return dict(host_to_device_bytes=sum(r.get('geometry_bytes',0)+r.get('setup_bytes',0) for r in rows),device_to_host_bytes=sum(r.get('export_bytes',0) for r in rows))
