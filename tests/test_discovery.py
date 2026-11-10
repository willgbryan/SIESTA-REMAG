"""Control-flow and scientific handoff tests; fixtures are not DFT results."""
import io
import json
from pathlib import Path
import shutil
import struct
import tarfile

import numpy as np
import pytest

ase = pytest.importorskip('ase')
from ase import Atoms
from ase.io import write
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import digest, load, save
from siesta_remag.discovery.config import Settings
from siesta_remag.discovery.inputs import axes, geometry, kmesh, prepare_case
from siesta_remag.discovery.pipeline import run_dft, verify_bundle
from siesta_remag.discovery.collection import pack, unpack
from siesta_remag.discovery.seed import collinear_to_soc, records
from siesta_remag.output import parse_output
from siesta_remag.runner import run_case


def scalar_dm(path):
    parts = [np.array([1, 2], dtype='<i4').tobytes(), np.array([1], dtype='<i4').tobytes(),
             np.array([1], dtype='<i4').tobytes(), np.array([1.5], dtype='<f8').tobytes(), np.array([.5], dtype='<f8').tobytes()]
    path.write_bytes(b''.join(struct.pack('<i', len(p))+p+struct.pack('<i', len(p)) for p in parts))


@pytest.fixture
def bundle(tmp_path):
    b = tmp_path/'bundle'
    (b/'orb').mkdir(parents=True)
    (b/'pseudos').mkdir()
    p = b/'pseudos/Fe.psml'
    p.write_text('<psml><pseudo-atom-spec atomic-label="Fe" relativity="dirac"><exchange-correlation><functional id="101"/><functional id="130"/></exchange-correlation></pseudo-atom-spec></psml>')
    atoms = Atoms('Fe2', positions=[[0,0,0],[1.5,1.5,1.5]], cell=np.eye(3)*3, pbc=True)
    write(b/'input.extxyz', atoms)
    write(b/'orb/relaxed.extxyz', atoms)
    save(b/'orb/result.json', {'accepted': True})
    request = dict(settings=Settings().to_dict(), pseudopotentials={'Fe.psml': sha256(p)}, input_sha256=sha256(b/'input.extxyz'))
    save(b/'request.json', request)
    files = ['request.json','input.extxyz','orb/result.json','orb/relaxed.extxyz','pseudos/Fe.psml']
    save(b/'prepared.json', dict(status='prepared', request_sha256=digest(request), kmesh=kmesh(atoms, Settings()),
                                 files={n:sha256(b/n) for n in files}, orb_seconds=1))
    return b


def backend(calls, fail=None, branch=False, axis_failure=False, mesh_mismatch=False):
    def run(case, executable, destination, *, timeout, threads, extra_env, restart):
        name = destination.name
        calls.append((name, timeout, restart))
        shutil.copytree(case, destination)
        manifest = load(destination/'manifest.json')
        direction = manifest['direction'] or [0,0,1]
        is_soc = name.startswith('soc')
        if is_soc:
            assert sha256(restart['path']) == restart['sha256']
            assert np.frombuffer(records(restart['path'])[0],dtype='<i4')[1] == 8
        moment = np.array(direction)*2
        if name == 'soc-x' and branch: moment *= -1
        if name == 'soc-x' and axis_failure: moment = np.array([0,0,2])
        obs = dict(total_energy_ev=-20+(0 if name in ('dft-relax','soc-z') else .0001), free_energy_ev=-20.01+(0 if name in ('dft-relax','soc-z') else .00008),
                   forces_ev_ang=[[0,0,0]]*2, stress_ev_ang3=np.zeros((3,3)).tolist(), atom_moments_muB=[moment.tolist()]*2,
                   total_moment_muB=(2*moment).tolist(), scf_iterations=3, spin_components=8 if is_soc else 2)
        text = 'SCF cycle converged\nJob completed\nInitMesh: MESH = 30 x 30 x 30\nsiesta: k-grid: Number of k-points = 45\n'
        text += 'redata: Spin configuration = '+('spin-orbit+offsite' if is_soc else 'collinear')+'\n'
        if is_soc: text += 'Attempting to read DM from file... Succeeded...\n'
        if name == 'dft-relax' and fail != 'optimizer': text += 'outcoor: Relaxed atomic coordinates\n'
        if name == fail: text += 'SCF_NOT_CONV\n'
        if name == 'soc-y' and mesh_mismatch: text = text.replace('30 x 30 x 30','32 x 30 x 30')
        (destination/'stdout.log').write_text(text)
        (destination/'stderr.log').write_text('')
        (destination/'fdf-1.log').write_text('DM.Tolerance 1e-4\nSCF.DM.Tolerance 1e-4\nSCF.H.Tolerance 1e-3 eV\nSCF.DM.Converge true\nSCF.H.Converge true\nSCF.MustConverge true\n')
        cell = manifest['geometry']['lattice_A']
        pos = manifest['geometry']['fractional_coordinates']
        (destination/'remag.STRUCT_OUT').write_text('\n'.join(' '.join(map(str,r)) for r in cell)+'\n2\n'+'\n'.join('1 26 '+' '.join(map(str,r)) for r in pos)+'\n')
        scalar_dm(destination/'remag.DM')
        result = dict(valid=name!=fail, returncode=0, timed_out=False, observables=obs, wall_seconds=.1, parse_errors=[])
        save(destination/'result.json', result)
        return result
    return run


def execute(bundle, tmp_path, calls, **kwargs):
    exe = tmp_path/'binary'
    exe.write_text('fixture only')
    return run_dft(bundle, tmp_path/'run', exe, executable_sha256=sha256(exe), runner=backend(calls, **kwargs))


def test_success_and_reuse(bundle, tmp_path):
    calls = []
    report = execute(bundle,tmp_path,calls)
    assert report['status'] == 'screened', report
    assert [c[0] for c in calls] == ['dft-relax','soc-z','soc-x','soc-y']
    assert report['screen']['signal'] == 'axial'
    assert report['K1'] is None and not report['screen']['precision_qualified']
    assert all(c[1] <= (1800 if c[0]=='dft-relax' else 600) for c in calls)
    assert execute(bundle,tmp_path,calls)['invocation_sha256'] == report['invocation_sha256']
    assert len(calls)==4
    (tmp_path/'run/dft-relax/remag.DM').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='evidence changed'): execute(bundle,tmp_path,calls)


@pytest.mark.parametrize('failure,expected,count', [('dft-relax','held_dft-relax',1),('optimizer','held_dft-relax',1),('soc-z','held_soc-z',2)])
def test_failure_stops_queue(bundle,tmp_path,failure,expected,count):
    calls=[]
    report=execute(bundle,tmp_path,calls,fail=failure)
    assert report['status']==expected and len(calls)==count
    assert report['screen'] is None
    if count==1: assert not (tmp_path/'run/dft-relaxed.extxyz').exists()


@pytest.mark.parametrize('kwargs,status', [({'branch':True},'held_magnetic_branch'),({'axis_failure':True},'held_soc-x'),({'mesh_mismatch':True},'held_error')])
def test_scientific_handoffs_stop(bundle,tmp_path,kwargs,status):
    report=execute(bundle,tmp_path,[],**kwargs)
    assert report['status']==status,report
    assert report['screen'] is None


def test_bundle_tamper_and_incomplete_manifest(bundle):
    receipt=load(bundle/'prepared.json')
    receipt['files'].pop('orb/relaxed.extxyz')
    save(bundle/'prepared.json',receipt)
    with pytest.raises(ValueError,match='Incomplete'): verify_bundle(bundle)


def test_interrupted_is_not_retry(bundle,tmp_path):
    execute(bundle,tmp_path,[])
    report=load(tmp_path/'run/report.json');report['status']='running';save(tmp_path/'run/report.json',report)
    with pytest.raises(RuntimeError,match='Interrupted'): execute(bundle,tmp_path,[])


def test_archive_and_corruption(bundle,tmp_path):
    execute(bundle,tmp_path,[])
    payload=pack(tmp_path/'run')
    assert unpack(payload,tmp_path/'collected')['status']=='screened'
    assert unpack(payload,tmp_path/'collected')['status']=='screened'
    (tmp_path/'collected/soc-z/stdout.log').write_text('changed')
    with pytest.raises(ValueError,match='checksum'): unpack(payload,tmp_path/'collected')


def test_unsafe_archive(tmp_path):
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w:gz') as archive:
        member=tarfile.TarInfo('../outside');member.size=1;archive.addfile(member,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='unsafe'): unpack(data.getvalue(),tmp_path/'bad')
    assert not (tmp_path/'outside').exists()


def test_seed_rotation_and_corruption(tmp_path):
    source=tmp_path/'source';scalar_dm(source)
    check=collinear_to_soc(source,tmp_path/'rotated',[1,0,0])
    assert check['charge_max_error']==0
    out=records(tmp_path/'rotated')
    values=[np.frombuffer(p,dtype='<f8')[0] for p in out[3:]]
    assert values==[1,1,.5,0,0,0,.5,0]
    with pytest.raises(ValueError): collinear_to_soc(source,tmp_path/'bad',[0,0,0])
    source.write_bytes(b'bad')
    with pytest.raises(ValueError): records(source)


def test_skew_axes_and_input_policy(bundle,tmp_path):
    a=Atoms('Fe',cell=[[3,0,.4],[0,3,0],[.3,0,3]],positions=[[0,0,0]],pbc=True)
    vectors=np.array(list(axes(a).values()))
    assert np.allclose(vectors@vectors.T,np.eye(3))
    p=prepare_case(a,bundle/'pseudos',tmp_path/'case',Settings(),relax=True)
    text=(p/'input.fdf').read_text()
    assert 'DM.Tolerance 0.0001' in text and 'SCF.H.Tolerance 0.001 eV' in text
    assert 'MD.VariableCell true' in text and 'MeshCutoff 300 Ry' in text
    assert 'DM.UseSaveDM false' in text
    with pytest.raises(ValueError): Settings.from_dict({'dm_tolerance':True})
    with pytest.raises(ValueError): Settings.from_dict({'typo':1})


def test_native_energy_precision():
    obs=parse_output('siesta: Total = -12.123456789\nsiesta: FreeEng = -12.123456799\n')
    assert obs['total_energy_ev']==-12.123456789 and obs['free_energy_ev']==-12.123456799


def test_runner_rejects_wrong_density(bundle,tmp_path):
    a=Atoms('Fe',cell=np.eye(3)*3,positions=[[0,0,0]],pbc=True)
    case=prepare_case(a,bundle/'pseudos',tmp_path/'case',Settings(),restart=True)
    exe=tmp_path/'exe';exe.write_text('not executed')
    seed=tmp_path/'seed';scalar_dm(seed)
    with pytest.raises(ValueError,match='checksum'):
        run_case(case,exe,tmp_path/'run',restart={'path':str(seed),'sha256':'wrong'})


def test_native_process_timeout_retains_failure(bundle,tmp_path):
    a=Atoms('Fe',cell=np.eye(3)*3,positions=[[0,0,0]],pbc=True)
    case=prepare_case(a,bundle/'pseudos',tmp_path/'case',Settings())
    exe=tmp_path/'sleeper'
    exe.write_text('#!/bin/sh\nexec sleep 10\n');exe.chmod(0o755)
    result=run_case(case,exe,tmp_path/'timed-out',timeout=.1)
    assert result['timed_out'] and not result['valid']
    assert result['wall_seconds'] < 5
    assert (tmp_path/'timed-out/execution-status.json').exists()


def test_fdf_spin_tokens_and_scalar_sign(bundle,tmp_path):
    a=Atoms('Fe2',cell=np.eye(3)*3,positions=[[0,0,0],[1.5,1.5,1.5]],pbc=True)
    a.set_initial_magnetic_moments([3.,-3.])
    case=prepare_case(a,bundle/'pseudos',tmp_path/'scalar',Settings())
    text=(case/'input.fdf').read_text().split('%block DM.InitSpin\n')[1].split('%endblock')[0]
    assert text.splitlines()==['1 3.000000000000','2 -3.000000000000']
    case=prepare_case(a,bundle/'pseudos',tmp_path/'soc',Settings(),direction=[1,0,0])
    text=(case/'input.fdf').read_text().split('%block DM.InitSpin\n')[1].split('%endblock')[0]
    assert text.splitlines()==['1 3.000000000000 90.000000000000 0.000000000000','2 3.000000000000 90.000000000000 180.000000000000']


@pytest.mark.parametrize('old,new', [('spin-orbit+offsite','spin-orbit'),('Attempting to read DM from file... Succeeded...','Density seed failed')])
def test_native_mode_and_seed_are_required(bundle,tmp_path,old,new):
    from siesta_remag.discovery.review import native_review
    execute(bundle,tmp_path,[])
    folder=tmp_path/'run/soc-z'
    log=folder/'stdout.log';log.write_text(log.read_text().replace(old,new))
    check=native_review(folder,load(folder/'result.json'),Settings(),axis=[0,0,1])
    assert not check['accepted']
