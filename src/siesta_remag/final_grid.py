"""Audit final force-grid publication from completed device-cycle outputs."""
import math
from pathlib import Path
import numpy as np


def record_errors(record, folder, shadow):
    errors=[]
    if (record.get('passed') is not True or record.get('authoritative') is not (not shadow) or
            record.get('comparison_performed') is not shadow or record.get('host_to_device_bytes')!=0 or
            record.get('device_to_host_bytes')!=0):
        errors.append('Invalid final force-grid mode/result/traffic')
    values=record.get('values',[]);reference=record.get('reference',[])
    n,nh=record.get('capacity'),record.get('spins')
    if type(n) is not int or n<=0 or nh not in (1,2,4,8) or type(record.get('hamiltonian_required')) is not bool:
        errors.append('Invalid final Hamiltonian shape/request')
    elif record.get('hamiltonian_cycle_export_bytes') != (8*n*nh if shadow or record['hamiltonian_required'] else 0):
        errors.append('Optional final Hamiltonian cycle export size differs')
    if len(values)!=27 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
        errors.append('Invalid final grid energy/stress publication')
    if shadow:
        if len(reference)!=27 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in reference):
            errors.append('Invalid final grid reference')
        elif len(values)==27:
            delta=np.abs(np.array(values)-reference)
            if not np.all(delta<=1e-9+1e-10*np.abs(reference)) or float(delta.max())!=record.get('max_grid_error'):
                errors.append('Final grid energy/stress differs from pinned recomputation')
        name=record.get('matrix_file');n=record.get('capacity');nh=record.get('spins')
        if (not isinstance(name,str) or Path(name).name!=name or not name.startswith('remag-final-grid-shadow-') or
                type(n) is not int or n<=0 or nh not in (1,2,4,8)):
            errors.append('Invalid final Hamiltonian fixture')
        elif not (Path(folder)/name).is_file():errors.append('Final Hamiltonian fixture missing')
        else:
            data=np.fromfile(Path(folder)/name,dtype=np.float64)
            if data.size!=2*n*nh or not np.isfinite(data).all():errors.append('Invalid final Hamiltonian array')
            else:
                reference_h,actual=np.split(data,2);delta=np.abs(actual-reference_h)
                if not np.all(delta<=1e-10+1e-10*np.abs(reference_h)) or float(delta.max())!=record.get('max_matrix_error'):
                    errors.append('Final Hamiltonian differs from pinned recomputation')
    elif (reference or record.get('matrix_file') is not None or record.get('max_grid_error') is not None or
          record.get('max_matrix_error') is not None):
        errors.append('Production final grid claims an internal reference comparison')
    return errors
