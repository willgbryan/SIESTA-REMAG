from copy import deepcopy
import xml.etree.ElementTree as ET
import pytest
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save
from siesta_remag.research.basis import validate_basis, basis_lines, orbital_count, audit_native_basis
from siesta_remag.research.experiments import prepare_experiment, verify_experiment, run_experiment
from test_research import experiment

@pytest.fixture
def spec():
    return dict(format='REMAG-PAO-BASIS/1', species={'Fe': [
        dict(n=4, l=0, zeta=3, polarization=2, radii_bohr=[0, 0, 0]),
        dict(n=3, l=2, zeta=3, polarization=0, radii_bohr=[0, 0, 0])]})

def native_fixture(folder, spec, *, pol=2):
    root = ET.Element('ion'); ET.SubElement(root, 'symbol').text = 'Fe'
    paos = ET.SubElement(root, 'paos')
    shells = [(s['n'], s['l'], s['zeta']) for s in spec['species']['Fe']] + [(4, 1, pol)]
    for n, l, count in shells:
        for z in range(1, count+1):
            orbital = ET.SubElement(paos, 'orbital', n=str(n), l=str(l), z=str(z))
            ET.SubElement(ET.SubElement(orbital, 'radfunc'), 'cutoff').text = str(5-z*.1)
    ET.ElementTree(root).write(folder/'Fe.ion.xml')
    with (folder/'stdout.log').open('a') as stream:
        stream.write('\ninitatomlists: Number of atoms, orbitals, and projectors: 2 48 16\n')

def test_bounded_basis_and_native_identity(tmp_path, spec):
    assert orbital_count(spec, ['Fe', 'Fe']) == 48
    assert 'n=4 0 3 P 2' in basis_lines(spec, ['Fe'])
    canonical = validate_basis(spec, ['Fe'])
    spec['species']['Fe'][0]['radii_bohr'][0] = 9
    assert canonical['species']['Fe'][0]['radii_bohr'][0] == 0
    native_fixture(tmp_path, canonical)
    assert audit_native_basis(tmp_path, canonical, ['Fe', 'Fe'])['spatial_orbitals'] == 48
    native_fixture(tmp_path, canonical, pol=1)
    with pytest.raises(ValueError, match='polarization'): audit_native_basis(tmp_path, canonical, ['Fe', 'Fe'])

@pytest.mark.parametrize('field,value', [('n', True), ('l', 4), ('zeta', 5), ('polarization', 3),
    ('radii_bohr', [0, float('nan'), 0]), ('radii_bohr', [0, 0]), ('radii_bohr', [31, 1, 1]),
    ('radii_bohr', [1, 2, 0]), ('n', '4\nSCF.MustConverge false')])
def test_reject_invalid_basis_before_authoring(spec, field, value):
    spec['species']['Fe'][0][field] = value
    with pytest.raises(ValueError): validate_basis(spec, ['Fe'])

def test_species_shell_and_fdf_injection_boundaries(spec):
    with pytest.raises(ValueError): validate_basis(spec, ['Fe', 'Pt'])
    changed = deepcopy(spec); changed['species']['Fe'][0]['extra'] = 'arbitrary FDF'
    with pytest.raises(ValueError): validate_basis(changed, ['Fe'])
    changed = deepcopy(spec); changed['species']['Fe'].append(deepcopy(changed['species']['Fe'][0]))
    with pytest.raises(ValueError): validate_basis(changed, ['Fe'])
    changed = deepcopy(spec); changed['species']['Fe'][1]['l'] = 1
    with pytest.raises(ValueError): validate_basis(changed, ['Fe'])

def test_explicit_basis_survives_all_four_stages_and_is_bound(experiment, tmp_path, spec):
    from test_discovery import backend
    bundle = tmp_path/'explicit'
    prepare_experiment(load(experiment/'structure.json'), experiment/'pseudos', bundle, mesh=[3]*3, basis_spec=spec)
    assert verify_experiment(bundle)['basis_size'] == 'explicit'
    original = (bundle/'scalar/input.fdf').read_text()
    assert '%block PAO.Basis' in original and 'PAO.BasisSize' not in original
    calls=[]; fake=backend(calls)
    def runner(case, executable, destination, **kw):
        result=fake(case, executable, destination, **kw)
        native_fixture(destination, spec)
        return result
    exe=tmp_path/'binary';exe.write_text('fixture')
    r=run_experiment(bundle,tmp_path/'result',exe,executable_sha256=sha256(exe),runner=runner)
    assert r['status']=='measured',r
    assert len(r['basis_checks'])==4
    for stage in ['scalar','soc-z','soc-x','soc-y']:
        assert load(tmp_path/'result'/stage/'manifest.json')['basis_spec']==validate_basis(spec,['Fe'])
    request=load(bundle/'experiment.json');request['basis_spec']['species']['Fe'][0]['polarization']=1
    save(bundle/'experiment.json',request)
    with pytest.raises(ValueError,match='basis differ'):verify_experiment(bundle)
