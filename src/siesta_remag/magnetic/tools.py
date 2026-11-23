"""Small authored-screen handoff shared by TWINS' actual tool interfaces.

No model calls, job queue, experiment registry, or alternative GPU launcher.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np
from .state import file_hash
from .authoring import export_authored_structure, _digest
from .compare import compare_magnetic_responses
from .force_theorem import directions_array


def _load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def authored_record(path):
    """Accept the existing REMAG handoff or GPAC's verified physical export."""
    value = _load(path)
    from gpac import ir
    program = ir.load_program(value['source']['program'])
    with tempfile.TemporaryDirectory() as temporary:
        record = export_authored_structure(program, Path(temporary)/'authored')
    if 'gpac_source_hash' in value:
        if (value['gpac_source_hash'] != record['gpac_source_hash'] or
                value['evaluated_geometry_hash'] != _digest(value['geometry']) or
                value['geometry'] != record['geometry'] or value['sites'] != record['sites']):
            raise ValueError('Authored geometry or exact source changed')
        # Preserve the actual previously authored edit/correspondence.
        return value
    from gpac.physical import verify_export
    export = verify_export(value)
    if export['source']['sha256'] != program.sha256:
        raise ValueError('GPAC source and physical export differ')
    geometry = export['geometry']
    if (not np.allclose(geometry['cell'], record['geometry']['cell_angstrom'], atol=2.e-8, rtol=0)
            or geometry['atomic_numbers'] != record['geometry']['atomic_numbers']
            or not np.allclose(np.asarray(geometry['fractional']) @ np.asarray(geometry['cell']),
                               record['geometry']['positions_angstrom'], atol=2.e-8, rtol=0)):
        raise ValueError('Magnetic handoff requires the common GPAC Cartesian frame and atom order')
    record['physical_site_mapping'] = dict(zip(geometry['site_ids'], record['geometry']['site_ids'], strict=True))
    return record


def prepare_screen(structure, output, *, meshes=None, directions=None,
                   exchange_mesh=None, exchange_radius_angstrom=2.8, angular_model=None):
    record = authored_record(structure)
    meshes = [[8,8,8]] if meshes is None else meshes
    if (not isinstance(meshes,list) or not 1 <= len(meshes) <= 4 or
            any(not isinstance(m,list) or len(m)!=3 or any(type(n) is not int or not 1<=n<=16 for n in m) for m in meshes)
            or len({tuple(m) for m in meshes}) != len(meshes)):
        raise ValueError('Choose one to four distinct meshes, each with three integers in 1..16')
    axes = directions_array([0,0,1], 'cartesian' if directions is None else directions).tolist()
    if len(axes)>12:raise ValueError('At most twelve screening directions per request')
    if exchange_mesh is not None and exchange_mesh not in meshes:raise ValueError('Exchange mesh must be one of the requested meshes')
    if not np.isfinite(exchange_radius_angstrom) or not 0<float(exchange_radius_angstrom)<=8:
        raise ValueError('Exchange radius must be in (0,8] Angstrom')
    if angular_model not in (None,'axial4','cubic6'):raise ValueError('Unknown angular model')
    request = dict(format='REMAG-MAGNETIC-REQUEST/1', structure=record, meshes=meshes,
        directions=axes, exchange_mesh=exchange_mesh,
        exchange_radius_angstrom=float(exchange_radius_angstrom), angular_model=angular_model)
    root=Path(output);root.mkdir(parents=True, exist_ok=False)
    _save(root/'request.json',request)
    return dict(request=str(root/'request.json'), source_sha256=record['gpac_source_hash'],
        request_sha256=file_hash(root/'request.json'), meshes=meshes,
        physical_site_mapping=record.get('physical_site_mapping'), calculation_performed=False)


def read_runtime(path):
    """Trusted operator configuration; request JSON cannot choose commands/resources."""
    runtime=_load(path)
    required={'format','repository','python','image','volume','template_volume','template_relative_path',
              'executable','settings','electronic_settings','timeout_seconds','scf_timeout_seconds'}
    if set(runtime)!=required or runtime['format']!='REMAG-MAGNETIC-RUNTIME/1':raise ValueError('Expected operator magnetic runtime')
    for key in ('timeout_seconds','scf_timeout_seconds'):
        if type(runtime[key]) is not int or not 1<=runtime[key]<=900:raise ValueError('Runtime timeouts must be 1..900 seconds')
    if runtime['scf_timeout_seconds']>runtime['timeout_seconds']:raise ValueError('SCF timeout exceeds allocation timeout')
    if not re.fullmatch(r'im-[A-Za-z0-9]+',runtime['image']):raise ValueError('Pinned Modal image required')
    for key in ('volume','template_volume'):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',runtime[key]):raise ValueError('Named Modal volume required')
    path=Path(runtime['template_relative_path'])
    if path.is_absolute() or '..' in path.parts:raise ValueError('Template must stay in the configured volume')
    if not isinstance(runtime['settings'],dict) or any(not isinstance(v,str) for v in runtime['settings'].values()):
        raise ValueError('Runtime environment must contain string values')
    repo=Path(runtime['repository']).resolve()
    if not (repo/'scripts/run_modal.py').is_file() or not (repo/'scripts/magnetic_modal.py').is_file():
        raise ValueError('Configured repository lacks the existing serialized launcher or magnetic entry point')
    if not Path(runtime['python']).is_file():raise ValueError('Configured Python executable is unavailable')
    return runtime


def run_prepared_screen(request_path, runtime_path, output):
    runtime=read_runtime(runtime_path)
    request_path=Path(request_path).resolve();request=_load(request_path)
    # Recheck geometry after preparation, before any cloud call.
    with tempfile.TemporaryDirectory() as temp:
        structure=Path(temp)/'structure.json';_save(structure,request['structure'])
        checked=prepare_screen(structure,Path(temp)/'checked',**{k:request[k] for k in
            ('meshes','directions','exchange_mesh','exchange_radius_angstrom','angular_model')})
        canonical=_load(checked['request'])
        if (np.shape(canonical['directions'])!=np.shape(request['directions']) or
                not np.allclose(canonical['directions'],request['directions'],atol=1.e-14,rtol=0)):
            raise ValueError('Prepared directions changed')
        canonical['directions']=request['directions']
        if canonical!=request:raise ValueError('Prepared request changed')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    _save(root/'submission.json',dict(request_sha256=file_hash(request_path),runtime_sha256=file_hash(runtime_path),
                                    request=str(request_path),runtime=str(Path(runtime_path).resolve())))
    env=dict(os.environ,REMAG_MAGNETIC_REQUEST=str(request_path),
             REMAG_MAGNETIC_RUNTIME=str(Path(runtime_path).resolve()),REMAG_MAGNETIC_OUTPUT=str(root))
    env['PYTHONPATH']=str(Path(runtime['repository'])/'src')
    command=[runtime['python'],'scripts/run_modal.py','--module-file','scripts/magnetic_modal.py']
    with (root/'launcher.log').open('w') as stream:
        try:
            process=subprocess.run(command,cwd=runtime['repository'],env=env,stdout=stream,stderr=subprocess.STDOUT,
                                   timeout=runtime['timeout_seconds']+600)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('Collection timed out; bounded remote work may still be active. Inspect owned-app.json and launcher.log before retrying') from error
    if process.returncode or not (root/'magnetic-run.json').exists():
        raise RuntimeError('Launch/collection did not complete; inspect launcher.log and owned-app.json before another launch')
    report = _load(root/'magnetic-run.json')
    if report.get('status') != 'complete':
        raise RuntimeError(f"Magnetic screen failed: {report.get('error', 'unknown native failure')}; evidence: {root/'magnetic-run.json'}")
    return inspect_screen(root/'magnetic-run.json')


def load_screen(report, mesh=None):
    report=Path(report);run=_load(report)
    if run.get('format')!='REMAG-MAGNETIC-RUN/1' or run.get('status')!='complete' or run.get('reference_converged') is not True:
        raise ValueError('A completed magnetic run with converged reference is required')
    selected=run['screens'][-1] if mesh is None else next((s for s in run['screens'] if s['mesh']==mesh),None)
    if selected is None:raise ValueError('Requested mesh is absent')
    path=(report.parent/selected['path']).resolve()
    if not path.is_relative_to(report.parent.resolve()) or file_hash(path)!=selected['sha256']:
        raise ValueError('Screen artifact changed or escapes its report directory')
    screen=_load(path)
    if screen['identity']!=run['identity']:raise ValueError('Screen identity differs from its reference')
    return run,screen,path


def inspect_screen(report, mesh=None):
    run,screen,path=load_screen(report,mesh)
    mft=screen['mft'];orbital=mft['electronic_descriptors'][0].get('orbital_response')
    exchange=screen.get('exchange')
    return dict(status='screening',identity=screen['identity'],mesh=screen['calculation_settings']['kmesh'],
        available_meshes=[s['mesh'] for s in run['screens']], reference_settings=run['reference_settings'],
        sites=screen.get('site_identity'), physical_site_mapping=screen.get('physical_site_mapping'),
        magnetization_muB_cell=screen['summary']['magnetization_muB_cell'],
        angular_response=[dict(direction=p['direction'],delta_free_ev=p['delta_free_ev']) for p in mft['points']],
        angular_model=mft.get('angular_model'), orbital_response=orbital,
        angle_sensitive_states=[dict(direction_index=d['direction_index'],states=[{k:v for k,v in s.items()
            if k!='orbital_weights'} for s in d['near_fermi_states'][:2]]) for d in mft['electronic_descriptors']],
        exchange_network=None if exchange is None else exchange['network'],
        performance=dict(mft=mft['performance'],exchange=None if exchange is None else exchange['performance']),
        limitations=screen['limitations']+['Dense frozen response inherits the original SCF density and magnetic field.',
            'Near-Fermi PAO mixing and band-index shifts do not prove a causal structural mechanism.'],
        screen=str(path),screen_sha256=file_hash(path),physical_accuracy_qualified=False)


def compare_screens(parent, child, *, mesh=None):
    a,sa,_=load_screen(parent,mesh);b,sb,_=load_screen(child,mesh)
    result=compare_magnetic_responses(sa,sb)
    if a['reference_settings']!=b['reference_settings']:
        result['numerically_comparable']=False
        result['comparability_limitations'].append('Underlying reference SCF settings differ')
    return result


def response_observable(screen, observable):
    if observable=='frozen_Fx_minus_Fz':
        return next(p['delta_free_ev'] for p in screen['mft']['points'] if np.allclose(p['direction'],[1,0,0])), 'eV/cell'
    if observable=='magnetization_muB_cell':return screen['summary'][observable], 'muB/cell'
    if isinstance(observable,dict) and set(observable)=={'co_weight'}:
        pair=observable['co_weight']
        for p in screen['mft']['electronic_descriptors'][0]['orbital_response']['intersite_band_mixing']:
            if p['groups']==pair:return p['near_fermi_co_weight'],'dimensionless'
    raise ValueError('Choose frozen_Fx_minus_Fz, magnetization_muB_cell, or an explicit co_weight group pair')
