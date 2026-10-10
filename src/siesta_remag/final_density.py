"""Strict evidence checks for a requested final density output handoff."""
from pathlib import Path
import math
import numpy as np


def record_errors(r,folder,shadow):
    errors=[]
    try:
        mesh=r['mesh'];spins=r['spins'];nsm=r['nsm']
        if len(mesh)!=3 or any(type(n) is not int or n<1 or n%nsm for n in mesh) or nsm<1 or nsm>64 or spins not in (1,2,4):
            return ['Invalid retained density shape']
        count=math.prod(mesh)*spins;size=count*8
        if r.get('kind') not in (1,2):errors.append('Invalid requested density kind')
        if r['shadow'] is not shadow or type(r['requested']) is not bool or type(r['exported']) is not bool:
            errors.append('Density export mode not recorded correctly')
        if r['retained_device_bytes']!=(size if r['requested'] else 0) or r['host_to_device_bytes']!=0 or r['device_to_host_bytes']!=(size if r['exported'] else 0):
            errors.append('Density retention/export bytes differ from shape')
        if r['exported']:
            if not r['requested'] or r['passed'] is not True or r['comparison_performed'] is not shadow:
                errors.append('Requested density publication failed or has the wrong authority')
        elif r['passed'] is not None or r['comparison_performed'] is not False:
            errors.append('Unexported density must not claim a numerical comparison')
        if r['exported'] and shadow:
            name=r['array_file']
            if not isinstance(name,str) or Path(name).name!=name or not name.startswith('remag-final-density-shadow-'):
                return errors+['Invalid density shadow filename']
            path=Path(folder)/name
            if not path.is_file() or path.stat().st_size!=size*2:return errors+['Density shadow array missing or wrong size']
            cpu,gpu=np.fromfile(path,dtype=np.float64).reshape(2,count)
            difference=np.abs(gpu-cpu);maximum=float(difference.max())
            if not np.isfinite(cpu).all() or not np.isfinite(gpu).all() or np.any(difference>1.e-10+1.e-10*np.abs(cpu)):
                errors.append('Full density shadow differs from recomputation')
            if not math.isfinite(r['max_absolute_error']) or r['max_absolute_error']!=maximum:
                errors.append('Density shadow maximum differs from retained arrays')
        elif r['array_file'] is not None or r['max_absolute_error'] is not None:
            errors.append('Uncompared density must not claim a reference error')
    except (KeyError,ValueError,TypeError,ZeroDivisionError,OverflowError) as exc:
        errors.append('Malformed density handoff evidence: '+str(exc))
    return errors
