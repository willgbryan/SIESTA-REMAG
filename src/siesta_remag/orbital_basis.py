"""Validate geometry-owned orbital derivatives and their explicit boundaries."""
import math


def record_errors(record,force,shadow=True):
    errors=[];shape=record.get('shape',[])
    if len(shape)!=9 or any(type(v) is not int or v<0 for v in shape) or any(shape[i]==0 for i in (0,1,3,4,5,6,7,8)):
        return ['Invalid orbital basis shape']
    ns,nb,samples,entries,nsp,no,na,offsets,np=shape
    staged=8*ns+32*nb+16*samples+8*entries+8*no+28*na+24*offsets+24*nsp
    derivatives=56*entries*nsp;extra=72 if shadow else 0
    if (record.get('compact_setup_bytes')!=staged or record.get('derivative_bytes')!=derivatives or
        record.get('shadow_setup_bytes')!=extra or record.get('shadow_export_bytes')!=extra or
        record.get('device_allocation_bytes')!=staged+derivatives+extra+(16 if samples==0 else 0)):
        errors.append('Orbital basis shape-derived staging/allocation/export differs')
    if (record.get('consumed') is not True or record.get('passed') is not True or record.get('status')!=0 or
        record.get('generated') is not True or record.get('comparison_performed')!=shadow or
        record.get('host_derivatives_authoritative')!=shadow):errors.append('Incomplete or failed orbital derivative generation')
    if shadow:
        if (record.get('compared')!=7*entries*nsp or record.get('mismatches')!=0 or record.get('nonfinite')!=0 or
            record.get('atol')!=1e-10 or record.get('rtol')!=1e-10 or record.get('displacements_exact') is not True):
            errors.append('Incomplete or failed full orbital derivative comparison')
        for key in ('max_absolute_error','max_scaled_error'):
            values=record.get(key,[])
            if not isinstance(values,list) or len(values)!=3 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):
                errors.append('Missing/nonfinite orbital derivative error');continue
            if values[2]!=0 or (key=='max_scaled_error' and max(values[:2])>1.):errors.append('Orbital derivative tolerance failed')
    elif any(key not in record or record[key] is not None for key in
             ('compared','mismatches','nonfinite','max_absolute_error','max_scaled_error','atol','rtol','displacements_exact')):
        errors.append('Production orbital record claims an internal comparison')
    grid=force.get('grid_shape',[])
    if (record.get('epoch')!=force.get('epoch') or force.get('orbital_basis_shadow')!=shadow or
        force.get('orbital_basis_device')!=(not shadow) or force.get('graph_nodes')!=(3 if shadow else 2) or
        len(grid)!=6 or [grid[0],grid[1],grid[2]]!=[no,np,nsp] or force.get('entries')!=entries):
        errors.append('Orbital basis does not match its force consumer')
    return errors
