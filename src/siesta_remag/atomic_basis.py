"""Validate geometry-owned core/neutral derivative shadow boundaries."""
import math


def record_errors(record,force,shadow=True):
    errors=[];s=record.get('shape',[])
    if len(s)!=13 or any(type(v) is not int for v in s) or any(s[i]<=0 for i in (0,2,3,4,5,6,7,8,9,10,11)) or s[1]<0 or s[12] not in (0,1):
        return ['Invalid atomic basis shape']
    ns,samples,pairs,nsp,atoms,offsets,units,points,nsm,mx,my,mz,core=s
    if nsp!=nsm**3 or points!=mx*my*mz or any(v%nsm for v in (mx,my,mz)):errors.append('Atomic basis mesh shape differs')
    n=pairs*nsp;staged=52*ns+16*samples+12*pairs+32*atoms+24*offsets+24*nsp;terms=record.get('terms',[])
    if record.get('compact_setup_bytes')!=staged or record.get('device_allocation_bytes')!=staged+(16 if samples==0 else 0)+(core+1)*(136*n+120 if shadow else 68*n):
        errors.append('Atomic basis shape-derived staging/allocation differs')
    if record.get('complete') is not True or [t.get('kind') for t in terms]!=([1,2] if core else [2]):errors.append('Incomplete atomic basis consumers')
    if record.get('shadow')!=shadow or record.get('host_derivatives_authoritative')!=shadow:errors.append('Atomic derivative mode differs')
    for t in terms:
        m=t.get('reference_selected');c=t.get('comparison')
        if any(t.get(k) is not True for k in ('begun','generated','consumed','passed')) or t.get('status')!=0:errors.append('Failed atomic derivative generation')
        if shadow:
            if type(m) is not int or m<0 or m>n or not isinstance(c,dict):errors.append('Missing atomic selection comparison');continue
            if t.get('shadow_setup_bytes')!=68*n+120 or t.get('device_allocation_bytes')!=136*n+120 or t.get('shadow_export_bytes')!=120:
                errors.append('Atomic shadow shape-derived traffic differs')
            if any(c.get(k)!=v for k,v in dict(points=n,selected=m,reference_selected=m,compared=4*n+3*m,index_compared=2*m,mismatches=0,nonfinite=0,mask_mismatches=0,index_mismatches=0).items()):
                errors.append('Atomic full-array comparison coverage/failure')
            for key in ('absolute','scaled'):
                v=c.get(key,[])
                if not isinstance(v,list) or len(v)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in v):
                    errors.append('Missing/nonfinite atomic derivative error');continue
                if v[2]!=0 or (key=='scaled' and max(v)>1):errors.append('Atomic derivative tolerance failed')
        elif ('reference_selected' not in t or m is not None or 'comparison' not in t or c is not None or
              t.get('shadow_setup_bytes')!=0 or t.get('shadow_export_bytes')!=0 or t.get('device_allocation_bytes')!=68*n):
            errors.append('Invalid production allocation or claimed internal comparison')
        matching=[r for r in force.get('atomic_terms',[]) if r.get('kind')==t.get('kind')]
        if len(matching)!=1 or matching[0].get('entries')!=(m if shadow else n) or matching[0].get('graph_nodes')!=(3 if shadow else 2):errors.append('Atomic derivative force consumer differs')
    grid=force.get('grid_shape',[])
    if record.get('epoch')!=force.get('epoch') or force.get('atomic_basis_shadow')!=shadow or force.get('atomic_basis_device')!=(not shadow) or force.get('final_partial_core')!=core or force.get('atoms')!=units or len(grid)!=6 or [grid[1]*grid[2],grid[2]]!=[points,nsp]:
        errors.append('Atomic owner does not match force geometry')
    return errors
