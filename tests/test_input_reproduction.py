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
