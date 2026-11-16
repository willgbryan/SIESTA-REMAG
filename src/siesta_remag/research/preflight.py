"""Offline mesh/capability checks before paid image preparation or submission.

The CPU builder probes a pure metadata symbol in the actual linked library and
binds that answer to executable/library hashes. Source limits alone are planning
evidence, never a replacement for a build-specific runtime manifest.
"""
import ctypes
import json
import math
from pathlib import Path
import re

from ..corpus import sha256
from ..discovery.common import load, digest


def validate_device_capabilities(value):
    keys={'format','max_kpoints','max_spatial_orbitals','max_spinor_dimension'}
    if (not isinstance(value,dict) or set(value)!=keys
            or value['format']!='REMAG-DEVICE-SCF-CAPABILITIES/1'
            or any(type(value[k]) is not int or not 1<=value[k]<=2**31-1 for k in keys-{'format'})
            or value['max_spinor_dimension']!=2*value['max_spatial_orbitals']):
        raise ValueError('Invalid compiled device-SCF capability metadata')
    return value


def probe_library(library):
    """Trusted operator/build operation. Reads metadata; never runs a solver."""
    lib=ctypes.CDLL(str(Path(library).resolve()))
    try:call=lib.remag_device_scf_capabilities
    except AttributeError as exc:raise ValueError('Library has no compiled device-SCF capability metadata') from exc
    call.argtypes=[];call.restype=ctypes.c_char_p
    raw=call()
    if not raw or len(raw)>4096:raise ValueError('Invalid compiled capability payload')
    return validate_device_capabilities(json.loads(raw))


def validate_runtime(runtime):
    if (not isinstance(runtime,dict) or set(runtime)-{'executable','executable_sha256','environment','threads','capabilities'}
            or not isinstance(runtime.get('executable'),str) or not runtime['executable'].startswith('/')
            or not isinstance(runtime.get('executable_sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',runtime['executable_sha256'])
            or type(runtime.get('threads',1)) is not int or not 1<=runtime.get('threads',1)<=8
            or not isinstance(runtime.get('environment',{}),dict)
            or any(not isinstance(k,str) or not isinstance(v,str) for k,v in runtime.get('environment',{}).items())):
        raise ValueError('Expected a pinned REMAG runtime manifest')
    enabled=runtime.get('environment',{}).get('REMAG_DEVICE_SCF')=='1'
    c=runtime.get('capabilities')
    if c is None:
        if enabled:raise ValueError('Device-SCF execution requires build-specific capabilities before spending')
        return None
    if (not isinstance(c,dict) or set(c)!={'format','executable_sha256','library','library_sha256','device_scf'}
            or c['format']!='REMAG-BOUND-CAPABILITIES/1'
            or c['executable_sha256']!=runtime['executable_sha256']
            or not isinstance(c['library'],str) or not c['library'].startswith('/')
            or not isinstance(c['library_sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',c['library_sha256'])):
        raise ValueError('Capabilities are not bound to this executable/library')
    validate_device_capabilities(c['device_scf'])
    if enabled and runtime.get('environment',{}).get('LD_LIBRARY_PATH','').split(':')[0]!=str(Path(c['library']).parent):
        raise ValueError('Runtime must select the pinned native library directory first')
    return c


def build_runtime(executable, library, environment, threads=1):
    executable,library=Path(executable).resolve(),Path(library).resolve()
    h=sha256(executable);lh=sha256(library)
    c=probe_library(library)
    if sha256(library)!=lh:raise ValueError('Library changed during capability probe')
    result=dict(executable=str(executable),executable_sha256=h,environment=environment,threads=threads,
        capabilities=dict(format='REMAG-BOUND-CAPABILITIES/1',executable_sha256=h,
            library=str(library),library_sha256=lh,device_scf=c))
    validate_runtime(result)
    return result


def verify_runtime_files(runtime):
    """Verify the uploaded manifest against actual files before native work."""
    c=validate_runtime(runtime)
    if sha256(Path(runtime['executable']))!=runtime['executable_sha256']:
        raise ValueError('Executable checksum differs from capability binding')
    if c and (sha256(Path(c['library']))!=c['library_sha256'] or probe_library(c['library'])!=c['device_scf']):
        raise ValueError('Native library bytes/capabilities differ from the authorized build')
    return dict(executable_sha256=runtime['executable_sha256'],capabilities=c,
                actual_files_verified=True)


def check_mesh_capacity(request, capabilities):
    """Full mesh is an upper bound; do not assume scalar time-reversal reduction."""
    validate_device_capabilities(capabilities)
    mesh=request.get('kmesh')
    if not isinstance(mesh,list) or len(mesh)!=3 or any(type(n) is not int or n<1 for n in mesh):
        raise ValueError('Expected three positive integer mesh dimensions')
    upper=math.prod(mesh);limit=capabilities['max_kpoints']
    if upper>limit:
        raise ValueError(f'Prepared mesh needs up to {upper} k-points; compiled device-SCF limit is {limit}; no spending')
    return dict(kmesh=mesh,native_kpoints_upper_bound=upper,compiled_max_kpoints=limit,
                reduction_assumed=False,mesh_capacity_passed=True)


def verify_prepared_mesh(bundle):
    """Verify the frozen input bytes and the mesh native scalar/SOC will use."""
    from .experiments import verify_experiment
    bundle=Path(bundle);request=verify_experiment(bundle)
    manifest=load(bundle/'scalar/manifest.json')
    mesh=request.get('kmesh')
    if not isinstance(mesh,list) or len(mesh)!=3 or any(type(n) is not int or n<1 for n in mesh):
        raise ValueError('Expected three positive integer mesh dimensions')
    if mesh!=manifest.get('kmesh'):raise ValueError('Requested and prepared scalar meshes differ')
    # These are generated, diagonal zero-shift input bundles. Fail closed on
    # indirection or alternate/duplicate mesh definitions rather than guessing.
    lines=[line.split('#',1)[0].strip().lower() for line in (bundle/'scalar/input.fdf').read_text().splitlines()]
    if any('<' in line or line.startswith('%include') for line in lines):
        raise ValueError('Indirect prepared input cannot establish the mesh before spending')
    canonical=[re.sub(r'[-_.]','',line) for line in lines]
    starts=[i for i,line in enumerate(canonical) if line=='%block kgridmonkhorstpack']
    ends=[i for i,line in enumerate(canonical) if line=='%endblock kgridmonkhorstpack']
    if len(starts)!=1 or ends!=[starts[0]+4]:raise ValueError('Expected one explicit prepared k-grid block')
    for i,line in enumerate(lines[starts[0]+1:ends[0]]):
        values=line.split()
        if len(values)!=4 or [int(x) for x in values[:3]]!=[mesh[i] if i==j else 0 for j in range(3)] or float(values[3])!=0:
            raise ValueError('Native input mesh differs from the registered request')
    return request


def preflight_query(bundle, runtime):
    from .experiments import validate_soc_request
    bundle=Path(bundle);request=verify_prepared_mesh(bundle);c=validate_runtime(runtime)
    validate_soc_request(request.get('soc_matrix',False),request.get('query'),runtime.get('environment',{}))
    result=dict(format='REMAG-QUERY-PREFLIGHT/1',request_sha256=sha256(bundle/'experiment.json'),
        runtime_sha256=digest(runtime),device_scf_enabled=runtime.get('environment',{}).get('REMAG_DEVICE_SCF')=='1',
        paid_actions=0,scope='Mesh capability and input/runtime identity only; not numerical acceptance, memory fit or latency.')
    if result['device_scf_enabled']:result.update(check_mesh_capacity(request,c['device_scf']))
    else:result.update(mesh_capacity_passed=None,reason='Host-controlled path; device-cycle limit does not apply')
    return result
