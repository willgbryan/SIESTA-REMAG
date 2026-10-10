"""Validate material kinetic/overlap shadows against complete CPU contributions."""
import math

def record_errors(report,integrals,forces,geometry=None):
    errors=[];rows=report.get('cycles',[])
    expected=[(f['epoch'],c) for f in forces for c in (1,4)]
    if not expected or report.get('complete') is not True or [(r.get('epoch'),r.get('consumer')) for r in rows]!=expected:
        errors.append('Pair shadow does not cover every geometry and consumer')
    for r in rows:
        e,c=r.get('epoch'),r.get('consumer');source=next((s for s in integrals.get('cycles',[]) if (s['epoch'],s['consumer'])==(e,c)),{})
        force=next((f for f in forces if f['epoch']==e),{})
        values=[r.get(k) for k in ('atoms','rows','columns','capacity','spinor','components','queries')]
        if any(type(v) is not int for v in values):errors.append('Invalid pair dimensions');continue
        na,nr,nc,cap,ns,components,n=values;size=3*na+10
        shape=force.get('grid_shape',[])
        if len(shape)!=6 or [nc,nr,cap]!=[shape[0],shape[3],shape[4]] or na!=force.get('atoms') or ns!=min(2,force.get('spins',0)) or components!=(force.get('spins') if c==1 else min(4,force.get('spins',0))):
            errors.append('Pair electronic shape differs from retained grid')
        if n<0 or n!=source.get('queries') or source.get('graph_nodes')!=(3 if n else 1):errors.append('Pair query/integral execution differs')
        staged=16+8*size if geometry is not None else 16*n+8*nr+4*cap+16+8*size
        wanted=dict(status=0,passed=True,complete=True,electronic_inputs_resident=True,electronic_upload_bytes=0,
                    contraction_graph_nodes=1,setup_bytes=staged,device_allocation_bytes=staged+(12 if n==0 and geometry is None else 0),export_bytes=8*size)
        if any(r.get(k)!=v for k,v in wanted.items()):errors.append('Pair execution or shape-derived traffic differs')
        if not isinstance(r.get('volume'),(float,int)) or not math.isfinite(r['volume']) or r['volume']<=0:errors.append('Invalid pair volume')
        arrays=[r.get(k,[]) for k in ('initial','reference','actual')]
        if any(len(a)!=size or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in a) for a in arrays):
            errors.append('Missing/nonfinite pair output arrays');continue
        initial,reference,actual=arrays
        if initial[0]!=0 or (c==4 and (reference[0]!=0 or actual[0]!=0)):errors.append('Pair energy convention differs')
        for i,(a,b) in enumerate(zip(actual,reference)):
            tolerance=1e-10+1e-10*abs(b) if i==0 else 1e-10 if i<=3*na else 1e-12
            if abs(a-b)>tolerance:errors.append('Pair contribution tolerance exceeded');break
    if geometry is not None:
        records=geometry.get('cycles',[])
        if report.get('geometry_staged_before_scf') is not True or geometry.get('complete') is not True or [r.get('epoch') for r in records]!=[f['epoch'] for f in forces]:
            errors.append('Missing pre-SCF pair geometry owner')
        for g in records:
            consumers=[r for r in rows if r.get('epoch')==g.get('epoch')]
            if [r.get('consumer') for r in consumers]!=[1,4] or any(any(g.get(k)!=r.get(k) for k in ('atoms','rows','columns','capacity','queries','volume')) for r in consumers):
                errors.append('Pair geometry changed between kinetic and overlap')
            shape=[g.get(k) for k in ('rows','capacity','queries')]
            if any(type(v) is not int or v<0 for v in shape):errors.append('Invalid pair geometry shape');continue
            nr,cap,n=shape;staged=8*nr+4*cap+48*n
            if g.get('setup_bytes')!=staged or g.get('device_allocation_bytes')!=staged+(24 if n==0 else 0) or g.get('consumer_mask')!=5 or g.get('complete') is not True:
                errors.append('Pair geometry lifecycle or transfer counts differ')
    return errors

def traffic(report,geometry=None):
    rows=report.get('cycles',[])
    return dict(host_to_device_bytes=sum(r.get('setup_bytes',0) for r in rows)+sum(r.get('setup_bytes',0) for r in (geometry or {}).get('cycles',[])),device_to_host_bytes=sum(r.get('export_bytes',0) for r in rows))
