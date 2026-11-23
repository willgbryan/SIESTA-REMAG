"""Small state/provenance checks and reuse of already executed FePt evidence."""
import json
from pathlib import Path
import numpy as np
import pytest
from siesta_remag.magnetic import MagneticState, load_magnetic_state, read_mft_evidence, summarize_magnetic_screen, compare_magnetic_responses
from siesta_remag.magnetic.operators import decompose, spin_scalar, rotation_between, rotated_direction, RY_EV
from siesta_remag.magnetic.force_theorem import angular_summary
from siesta_remag.magnetic.multiplet import spherical_tensor,coulomb_terms,solve_4f_multiplet,crystal_field

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'tests/fixtures/fept-frozen-potential.json'


def state():
    m = dict(schema='REMAG-MAGNETIC-STATE/1', converged=True, calculation_id='small-operator-check',
        energy_unit='Ry', spinor_order='orbital_up_down', occupation_model='fermi_dirac',
        magnetic_terms=dict(xc='rigid_spin_rotation', hubbard='absent', constraint='absent', external_field='absent'),
        soc_source='native_offsite', reference_has_soc=False, reference_direction=[0,0,1],
        evidence=[{'purpose':'operator/serialization fixture; not material evidence'}],
        kpoint_convention='fractional_reciprocal', mesh_complete=True, temperature_ry=.002,
        electrons=2., orbitals=[{'site_id':'A'}, {'site_id':'B'}])
    base = np.diag([0.,0.,1.,1.])[None].astype(complex)
    mag = np.diag([-.2,.2,-.1,.1])[None].astype(complex)
    soc = np.zeros_like(base);soc[0,0,3]=.01j;soc[0,3,0]=-.01j
    return MagneticState(m,base,mag,soc,np.eye(4)[None],np.zeros((1,3)),np.ones(1))


def test_state_roundtrip_and_corrupted_arrays(tmp_path):
    s=state();path=s.save(tmp_path/'state');loaded=load_magnetic_state(path)
    np.testing.assert_array_equal(s.h_mag, loaded.h_mag)
    with (path.parent/'matrices.npz').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='hash'):load_magnetic_state(path)


def test_native_scalar_trace_stays_in_base():
    s=state();scalar=spin_scalar(np.array([[[.7,.1j],[-.1j,.3]]]))
    h=s.h_base+s.h_mag+s.h_soc
    base,mag,soc=decompose(h,s.h_soc+scalar)
    np.testing.assert_allclose(base,s.h_base,atol=1.e-15)
    np.testing.assert_allclose(mag,s.h_mag,atol=1.e-15)
    np.testing.assert_allclose(soc,s.h_soc,atol=1.e-15)
    np.testing.assert_allclose(base+mag+soc,h,atol=1.e-15)


def test_state_rejects_partial_mesh_and_unsupported_hubbard():
    for key,value in [('mesh_complete',False),('magnetic_terms',dict(xc='rigid_spin_rotation',hubbard='included',constraint='absent',external_field='absent'))]:
        s=state();s.metadata[key]=value
        with pytest.raises(ValueError):MagneticState(s.metadata,s.h_base,s.h_mag,s.h_soc,s.overlap,s.kpoints,s.weights)


def test_antiparallel_rotation():
    a=np.array([1.,2.,3.]);a/=np.linalg.norm(a)
    np.testing.assert_allclose(rotated_direction(a,rotation_between(a,-a)),-a,atol=1.e-14)


def test_saved_fept_is_screening_not_qualification():
    result=read_mft_evidence(EVIDENCE,reference_direction=[0,0,1])
    assert result['status']=='screening'
    assert result['physical_accuracy_qualified'] is False
    assert result['points'][1]['delta_free_ev']==pytest.approx(-4.99946266195825e-5*RY_EV)
    assert result['angular_model'] is None
    assert result['classification']=='insufficient_angular_sampling'
    assert result['points'][1]['torque_ev_per_radian'] is None
    screen=summarize_magnetic_screen(dict(calculation_id='saved-fept',evidence=[]),mae=result)
    assert screen['identity']['gpac_source_hash'] is None
    assert screen['exchange'] is None
    assert compare_magnetic_responses(screen,screen)['numerically_comparable'] is False


def test_failed_native_output_not_exposed(tmp_path):
    report=json.loads(EVIDENCE.read_text());report['points'][1]['status']=2
    path=tmp_path/'failed.json';path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Failed'):read_mft_evidence(path,reference_direction=[0,0,1])


def test_two_directions_do_not_determine_k1():
    points=[dict(direction=[0,0,1],delta_free_ev=0.),dict(direction=[1,0,0],delta_free_ev=.001)]
    assert angular_summary(points,volume_angstrom3=100)['anisotropy_density_mj_m3']==pytest.approx(1.602176634)
    with pytest.raises(ValueError,match='independently'):angular_summary(points,model='axial4')


def test_native_lower_triangle_matches_solver_convention():
    from siesta_remag.magnetic.operators import native_lower_hermitian
    a=np.array([[1.,99.+2j],[.2-.3j,2.]])
    actual=native_lower_hermitian(a)
    assert actual[0,1]==.2+.3j
    np.testing.assert_array_equal(actual,actual.conj().T)


def test_atomic_coulomb_monopole_and_tensor_conjugation():
    np.testing.assert_allclose(spherical_tensor(0,0),np.eye(7),atol=1.e-14)
    for k in (2,4,6):
        for q in range(-k,k+1):
            np.testing.assert_allclose(spherical_tensor(k,-q),(-1.)**q*spherical_tensor(k,q).conj().T,atol=1.e-14)
    terms=coulomb_terms({0:1.,2:0.,4:0.,6:0.})
    assert len(terms)==91
    assert all(p==r and q==s and abs(value-1)<1.e-14 for p,q,r,s,value in terms)


def test_missing_multiplet_parameters_and_cf_convention():
    result=solve_4f_multiplet({'species':'Sm'})
    assert result['status']=='incomplete' and result['points'] is None
    with pytest.raises(ValueError,match='Stevens'):
        crystal_field(dict(crystal_field_convention='Stevens',crystal_field_ev=[]))
