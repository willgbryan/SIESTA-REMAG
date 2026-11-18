import shutil
import numpy as np
import pytest
from ase import Atoms
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save, digest
from siesta_remag.discovery.config import Settings
from siesta_remag.discovery.inputs import prepare_case
from siesta_remag.research.input_reproduction import verify_prepared_input, START, END


@pytest.fixture
def pair(tmp_path):
    p=tmp_path/'pseudos';p.mkdir()
    (p/'Ni.psml').write_text('<psml><pseudo-atom-spec atomic-label="Ni" relativity="dirac"><exchange-correlation><functional id="101"/><functional id="130"/></exchange-correlation></pseudo-atom-spec></psml>')
    atoms=Atoms('Ni2',scaled_positions=[[.003,.002,.001],[.5,.5,.5]],cell=[[3.5,.0001,0],[0,3.4,.0001],[.0001,0,3.6]],pbc=True)
    a=prepare_case(atoms,p,tmp_path/'expected',Settings(),mesh=[4,4,4],relax=True,diagnostics=True)
    b=tmp_path/'actual';shutil.copytree(a,b)
    return a,b


def perturb(folder,delta):
    m=load(folder/'manifest.json');m['geometry']['fractional_coordinates'][1][2]+=delta
    p=folder/'input.fdf';lines=p.read_text().splitlines(keepends=True);i=lines.index(START)+2
    lines[i]=' '.join(f'{v:.16f}' for v in m['geometry']['fractional_coordinates'][1])+' 1\n'
    p.write_text(''.join(lines));m['input_sha256']=sha256(p);m['geometry_sha256']=digest(m['geometry']);save(folder/'manifest.json',m)


def test_portable_coordinate_solve_retains_original_hashes(pair):
    a,b=pair;assert verify_prepared_input(a,b)['exact_input_bytes']
    perturb(b,np.spacing(.5));before=sha256(b/'input.fdf')
    proof=verify_prepared_input(a,b)
    assert not proof['exact_input_bytes'] and proof['max_fractional_reconstruction_delta']==np.spacing(.5)
    assert sha256(b/'input.fdf')==before


def test_reconstructed_soc_axis_requires_adjacent_components_and_identical_spin_input(pair):
    original, _ = pair
    atoms = Atoms('Ni', positions=[[0,0,0]], cell=[3.5]*3, pbc=True)
    a = prepare_case(atoms, original.parent/'pseudos', original.parent/'soc-x', Settings(),
                     direction=[0.9999999998760509, -1.263669054552652e-5, 9.392147327529871e-6])
    b = original.parent/'copy'
    shutil.copytree(a, b)
    m = load(b/'manifest.json')
    m['direction'][2] = float(np.nextafter(m['direction'][2], np.inf))
    save(b/'manifest.json', m)
    assert verify_prepared_input(a,b)['exact_input_bytes']
    m['direction'][2] = float(np.nextafter(m['direction'][2], np.inf))
    save(b/'manifest.json', m)
    with pytest.raises(ValueError):
        verify_prepared_input(a,b)

    m['direction'][2] = float(np.nextafter(m['direction'][2], -np.inf))
    fdf = b/'input.fdf'
    fdf.write_text(fdf.read_text().replace('%block DM.InitSpin\n1 ', '%block DM.InitSpin\n1 0 '))
    m['input_sha256'] = sha256(fdf)
    save(b/'manifest.json', m)
    with pytest.raises(ValueError):
        verify_prepared_input(a,b)


def test_fe2co_lattice_axis_reproduces_both_binary64_dot_reductions():
    from siesta_remag.research.input_reproduction import _reconstructed_axes
    axes = _reconstructed_axes([[2.814950438, .000000701, -.000059730],
                                [.000000918, 3.977232862, .000119616],
                                [-.000127210, -1.988436607, 5.973303953]], 'soc-x')
    local = (.9999999997749514, -1.2516734035804875e-5, 1.7129760785587528e-5)
    native = (.9999999997749514, -1.2516734035804873e-5, 1.712976078558751e-5)
    assert local in axes and native in axes
    assert (*native[:2], np.nextafter(native[2], 0)) not in axes


def test_fe10co3_polar_rounding_requires_one_consistent_reconstructed_block(pair, monkeypatch):
    import math
    original, _ = pair
    axis = [0.00036517154982476305, 0.9941769384487912, -0.10775936945958714]
    # The saved Linux acos and Darwin acos differ by one binary64 step.
    native = 1.6787653453881948
    monkeypatch.setattr(math, 'acos', lambda z: math.nextafter(native, 0.))
    atoms = Atoms('Ni2', scaled_positions=[[0,0,0],[.5,.5,.5]], cell=[3.5]*3, pbc=True)
    a = prepare_case(atoms, original.parent/'pseudos', original.parent/'soc-y', Settings(),
                     direction=axis, spins=[2.48, -1.68])
    b = original.parent/'linux'; shutil.copytree(a, b)
    f = b/'input.fdf'
    initial = f.read_text()
    def replace_angles(positive, negative):
        text = initial.replace('2.480000000000 96.186169083565', '2.480000000000 '+positive)
        text = text.replace('1.680000000000 83.813830916435', '1.680000000000 '+negative)
        assert text != initial
        f.write_text(text)
        m = load(b/'manifest.json'); m['input_sha256'] = sha256(f); save(b/'manifest.json', m)
    replace_angles('96.186169083566', '83.813830916434')
    assert verify_prepared_input(a, b)['soc_polar_rendering_reproduced']
    for positive, negative in [('96.186169083567','83.813830916434'),
                               ('96.186169083566','83.813830916435')]:
        replace_angles(positive, negative)
        with pytest.raises(ValueError):
            verify_prepared_input(a, b)


@pytest.mark.parametrize('change',['larger_coordinate','cell','species','settings','moment','option','coordinate_species','forged_hash','unrecorded_coordinate','newline'])
def test_roundoff_rule_cannot_admit_other_changes(pair,change):
    a,b=pair;m=load(b/'manifest.json');p=b/'input.fdf'
    if change=='larger_coordinate':perturb(b,1e-12)
    elif change=='unrecorded_coordinate':
        perturb(b,np.spacing(.5));p.write_text(p.read_text().replace('0.5000000000000001','0.5000000000000002'))
        m=load(b/'manifest.json');m['input_sha256']=sha256(p);save(b/'manifest.json',m)
    else:
        if change=='cell':m['geometry']['lattice_A'][0][0]+=1e-12
        if change=='species':m['geometry']['species'][0]='Co'
        if change=='settings':m['settings']['mesh_ry']=400
        if change=='moment':m['initial_moments_muB'][0]+=1e-12
        if change=='option':p.write_text(p.read_text().replace('SCF.MustConverge true','SCF.MustConverge false'))
        if change=='newline':p.write_bytes(p.read_bytes().replace(b'\n',b'\r\n'))
        if change=='coordinate_species':
            lines=p.read_text().splitlines(keepends=True);i=lines.index(START)+1;lines[i]=lines[i].rsplit(' ',1)[0]+' 2\n';p.write_text(''.join(lines))
        m['input_sha256']=sha256(p) if change!='forged_hash' else '0'*64
        m['geometry_sha256']=digest(m['geometry']);save(b/'manifest.json',m)
    with pytest.raises(ValueError):verify_prepared_input(a,b)
