"""Native-format, angular-algebra and evidence-bound orbital-moment checks."""
from copy import deepcopy
from pathlib import Path
import shutil
import struct
import subprocess
import numpy as np
import pytest
from siesta_remag.corpus import sha256
from siesta_remag.density import read_dm
from siesta_remag.discovery.common import load, save
from siesta_remag.research.orbital_moments import (
    _angular, export_orbital_moments, load_orbital_moments, magnetic_orbital_metadata, moment_kernels, native_moments)

def write_dm(path, header, counts, columns, density, endian='<'):
    with path.open('wb') as stream:
        def record(value, kind):
            raw = np.asarray(value, dtype=endian+kind).tobytes()
            marker = struct.pack(endian+'i', len(raw))
            stream.write(marker+raw+marker)
        record(header, 'i4'); record(counts, 'i4')
        offsets = np.r_[0, np.cumsum(counts)]
        for start,end in zip(offsets[:-1], offsets[1:]): record(columns[start:end], 'i4')
        for spin in density:
            for start,end in zip(offsets[:-1], offsets[1:]): record(spin[start:end], 'f8')

def p_shell():
    # An occupied (p_y-i*p_x)/sqrt(2) orbital has L_z = +1 in the
    # pinned real-harmonic convention. Large translated-cell entries must
    # not contribute to this deliberately onsite observable.
    c = np.array([1.,0.,-1j])/np.sqrt(2)
    d = np.outer(c,c.conj())
    density = np.zeros((8,3,9))
    density[0,:,:3] = d.real
    density[4,:,:3] = d.imag
    density[4,:,3:] = 100.
    hsx = dict(header=np.array([1,3,8,1,3,1,1]), endian='<',
        labels=np.array([1,3]), basis=[b'Fe'.ljust(20)+struct.pack('<di',8.,3),
        np.array([[2,1,1]]*3,dtype='<i4').tobytes()],
        offsets=np.array([0,0,0,-1,0,0,1,0,0]), counts=np.array([9,9,9]),
        columns=np.tile(np.arange(1,10),3))
    dm = dict(header=np.array([3,8,3,1,1]),counts=hsx['counts'].copy(),
              columns=hsx['columns'].copy(), density=density.reshape(8,27))
    return hsx, dm

@pytest.mark.parametrize('endian', ['<','>'])
def test_native_dm_reader_and_circular_p_state(tmp_path,endian):
    hsx,dm = p_shell(); path = tmp_path/'remag.DM'
    write_dm(path, **dm, endian=endian)
    loaded = read_dm(path)
    orbitals,arrays = native_moments(hsx,loaded)
    assert [o['m'] for o in orbitals] == [-1,0,1]
    assert np.allclose(arrays['orbital_moments_mu_b'],[[0,0,.5],[0,0,0],[0,0,.5]])
    assert np.allclose(arrays['atom_moments_mu_b'],[[0,0,1.]])
    assert np.array_equal(loaded['density'],dm['density'])

@pytest.mark.parametrize('l', [0,1,2,3])
def test_shell_angular_algebra_and_spectrum(l):
    orbitals = [dict(atom_index=0,l=l,zeta=1,m=m) for m in range(-l,l+1)]
    operators = 1j*moment_kernels(orbitals)
    for axis in range(3):
        a,b = (axis+1)%3,(axis+2)%3
        assert np.allclose(operators[axis],operators[axis].conj().T)
        assert np.allclose(operators[a]@operators[b]-operators[b]@operators[a],1j*operators[axis])
        assert np.allclose(np.linalg.eigvalsh(operators[axis]),np.arange(-l,l+1))
    assert np.allclose(sum(a@a for a in operators), l*(l+1)*np.eye(2*l+1))

def test_atom_and_zeta_separation():
    orbs = [dict(atom_index=a,l=1,zeta=z,n=2,m=m) for a,z in [(0,1),(0,2),(1,1)] for m in range(-1,2)]
    kernels = moment_kernels(orbs)
    assert not kernels[:,:3,3:].any()
    assert not kernels[:,3:6,6:].any()
    assert kernels[:,:3,:3].any()

def test_partial_or_unsupported_shell_and_topology_are_rejected():
    hsx,dm = p_shell()
    bad = deepcopy(hsx); bad['basis'][1] = np.array([[2,1,1],[2,1,1],[2,0,2]],dtype='<i4').tobytes()
    with pytest.raises(ValueError,match='complete pinned'): magnetic_orbital_metadata(bad)
    bad = deepcopy(hsx); bad['header'][2] = 4
    with pytest.raises(ValueError,match='matching SOC'): native_moments(bad,dm)
    bad = deepcopy(dm); bad['columns'][0] = 2
    with pytest.raises(ValueError,match='topologies differ'): native_moments(hsx,bad)
    bad_h,bad_d = deepcopy(hsx),deepcopy(dm)
    bad_h['columns'][0] = bad_d['columns'][0] = 2
    with pytest.raises(ValueError,match='Repeated'): native_moments(bad_h,bad_d)

@pytest.mark.parametrize('mutation', ['tail','truncate','marker','column','nan','size'])
def test_malformed_density_rejected(tmp_path,mutation):
    _,dm = p_shell(); path = tmp_path/'bad.DM'
    if mutation == 'column': dm['columns'][0] = 10
    if mutation == 'nan': dm['density'][0,0] = np.nan
    if mutation == 'size': dm['header'][0] = 2147483647
    write_dm(path, **dm)
    raw = path.read_bytes()
    if mutation == 'tail': path.write_bytes(raw+b'x')
    if mutation == 'truncate': path.write_bytes(raw[:-5])
    if mutation == 'marker': path.write_bytes(raw[:24]+struct.pack('<i',16)+raw[28:])
    with pytest.raises(ValueError): read_dm(path)

def saved_fixture(root):
    root.mkdir(); stage = root/'soc-z'; stage.mkdir()
    hsx,dm = p_shell()
    write_dm(stage/'remag.DM', **dm)
    with (stage/'remag.HSX').open('wb') as f:
        def record(raw):
            marker=struct.pack('<i',len(raw)); f.write(marker+raw+marker)
        def array(value,kind): record(np.array(value,dtype='<'+kind).tobytes())
        array([2],'i4');array([1],'i4');array(hsx['header'],'i4')
        array([*np.eye(3).ravel(),0.,1.,.001],'f8')
        record(hsx['offsets'].astype('<i4').tobytes()+np.zeros(3,dtype='<f8').tobytes()+hsx['labels'].astype('<i4').tobytes())
        for value in hsx['basis']:record(value)
        record(bytes(60));array(hsx['counts'],'i4')
        for row in hsx['columns'].reshape(3,9):array(row,'i4')
        for _ in range(8*3):array(np.zeros(9),'f8')
        for i in range(3):array(np.eye(3,9)[i],'f8')
    # An earlier table is intentionally different; use only the final table.
    (stage/'stdout.log').write_text('moments: Magnetic moments from orbital angular momenta:\nSum 7 0 0 7\n'
        'moments: Magnetic moments from orbital angular momenta:\nSpecies: Fe\n'
        '1 1 2py .500 0.000 0.000 .500\n1 2 2pz .000 0.000 0.000 .000\n'
        '1 3 2px .500 0.000 0.000 .500\n1 Total 1.000 0.000 0.000 1.000\nSum 1.000 0.000 0.000 1.000\n')
    report = dict(format='REMAG-EXPERIMENT-RESULT/1',source_sha256='a'*64,request=dict(source_sha256='a'*64),
        executable_sha256='b'*64,stages=[dict(stage='soc-z',accepted=True,branch=dict(passed=True))],
        artifacts={'soc-z/'+p.name:sha256(p) for p in stage.iterdir()})
    save(root/'report.json',report)
    return root/'report.json',stage

def test_orbital_moment_cli_dispatch(tmp_path,capsys):
    from siesta_remag.research.__main__ import main
    report,native=saved_fixture(tmp_path/'saved')
    main(['orbital-moments',str(report),'--stage','soc-z','--native-directory',str(native),'--output',str(tmp_path/'out')])
    assert 'REMAG-ORBITAL-MOMENTS/1' in capsys.readouterr().out
    meta,_=load_orbital_moments(tmp_path/'out/orbital-moments.json')
    assert set(meta['implementation_sha256']) == {'orbital_moments.py','electronic.py','../density.py','../hsx.py'}

@pytest.mark.parametrize('mutation', ['hash','table','branch','stage','incomplete'])
def test_saved_evidence_failures_do_not_write_output(tmp_path,mutation):
    report,native=saved_fixture(tmp_path/'saved'); r=load(report)
    if mutation == 'hash': (native/'remag.DM').write_bytes((native/'remag.DM').read_bytes()+b'x')
    if mutation in ('table','incomplete'):
        path=native/'stdout.log'; text=path.read_text()
        text=text.replace('1 Total 1.000 0.000 0.000 1.000','1 Total 1.000 0.000 0.000 .900' if mutation=='table' else '')
        path.write_text(text);r['artifacts']['soc-z/stdout.log']=sha256(path)
    if mutation == 'branch':r['stages'][0]['branch']['passed']=False
    save(report,r)
    with pytest.raises(ValueError):
        export_orbital_moments(report,'../soc-z' if mutation=='stage' else 'soc-z',native,tmp_path/'out')
    assert not (tmp_path/'out').exists()

def test_port_matches_pinned_native_angular_routine(tmp_path):
    """Compile just angular arithmetic, never a SIESTA/native calculation."""
    source=Path(__file__).parents[1]/'upstream/siesta/Src/spinorbit.f'
    compiler=shutil.which('gfortran')
    if not source.exists() or compiler is None:pytest.skip('Pinned source and Fortran compiler are needed')
    text=source.read_text(); start=text.index('      subroutine int_so_ang(')
    end=text.index('      end subroutine int_so_ang',start)+len('      end subroutine int_so_ang')
    routine=text[start:end].replace('      implicit none','      implicit none\n      integer, parameter :: dp=kind(1.0d0)',1)
    (tmp_path/'angular.f').write_text(routine+'\n')
    (tmp_path/'check.f90').write_text('''program check
implicit none
integer :: l,i,j
real(8) :: v(3)
do l=1,3
 do i=-l,l
  do j=-l,l
   call int_so_ang(l,i,j,v)
   write(*,*) l,i,j,v
  end do
 end do
end do
end program
''')
    subprocess.run([compiler,str(tmp_path/'angular.f'),str(tmp_path/'check.f90'),'-o',str(tmp_path/'check')],check=True,capture_output=True,timeout=30)
    rows=subprocess.run([str(tmp_path/'check')],check=True,capture_output=True,text=True,timeout=5).stdout.splitlines()
    assert len(rows)==83
    for line in rows:
        fields=line.split();l,i,j=map(int,fields[:3])
        assert np.allclose(_angular(l,i,j),list(map(float,fields[3:])),atol=1e-14,rtol=0)
