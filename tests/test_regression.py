import copy
import struct
from pathlib import Path
import numpy as np
import pytest
from siesta_remag.compare import difference, compare, compare_mae
from siesta_remag.output import parse_output, read_eigenvalues, read_grid
from siesta_remag.profile import summarize
from siesta_remag.runner import run_case
from siesta_remag.corpus import prepare

def test_tree_accounts_exclusive_time_once():
    tree = {"global": {"_time": 10., "_calls": 1,
             "cdiag": {"_time": 6., "_calls": 2, "cdiag1": {"_time": 1., "_calls": 2}},
             "IterSCF_1": {"_time": 3., "_calls": 1}}}
    result = summarize(tree, 10.)
    assert sum(c["wall_seconds"] for c in result["components"].values()) == 10.
    assert result["components"]["diagonalization"]["wall_seconds"] == 6.
    assert result["scf_seconds"] == 3.

def test_invalid_observables_fail_closed():
    for a, b in ((None, None), ([], []), ([1], [1, 2]), ([np.nan], [np.nan]), ([np.inf], [np.inf])):
        assert not difference(a, b, 1.e-3)["passed"]

def test_nonconverged_output_with_final_energy_is_rejected():
    text = "SCF_NOT_CONV: SCF did not converge\nsiesta: Total = -1.0000\nJob completed\n"
    result = parse_output(text)
    assert result["total_energy_ev"] == -1.
    assert not result["converged"]

@pytest.mark.parametrize("spin,channels", [(1, 1), (2, 2), (4, 1), (8, 1)])
def test_eig_spinor_header_is_not_band_multiplicity(tmp_path, spin, channels):
    path = tmp_path / "test.EIG"
    path.write_text(f"-3.0\n2 {spin} 2\n1 " + " ".join(["-2", "1"]*channels) +
                    "\n2 " + " ".join(["-1", "2"]*channels))
    result = read_eigenvalues(path)
    assert np.shape(result["values_ev"]) == (2, 2*channels)

def test_current_force_and_spin_tables():
    output = parse_output('''SCF cycle converged after 2 iterations
siesta: Atomic forces (eV/Ang):
     1    0.100000   -0.200000   -0.000000
----------------------------------------
Mulliken Atomic Populations:
Atom #   charge [q] valence [e]           S      Sx [e]      Sy [e]      Sz [e]  Species
     1     0.319961    7.680039    2.823228   -0.000000   -0.000000    2.823228  Fe_fept_SOC
siesta: Eso(nil) = 0.0000
Job completed
''')
    assert output["forces_ev_ang"] == [[.1, -.2, 0.]]
    assert output["atom_moments_muB"] == [[0., 0., 2.823228]]
    assert output["soc_energy_ev"] is None
    assert output["soc_energy_representation"] == "included_in_nonlocal"

def test_nonmagnetic_population_table_requires_verified_spin_degeneracy():
    table = '''Mulliken Atomic Populations:
Atom #   charge [q] valence [e]  Species
     1     0.109436   18.890564  Au
     2    -0.109436    1.109436  H
'''
    declarations = '''redata: Spin configuration = none
redata: Number of spin components = 1
'''
    output = parse_output(declarations + table)
    assert output['atom_net_charges_e'] == [.109436, -.109436]
    assert output['atom_moments_muB'] == [[0.,0.,0.], [0.,0.,0.]]
    assert output['total_moment_muB'] == [0.,0.,0.]
    assert output['spin_degenerate'] is True
    for incomplete in ('', 'redata: Spin configuration = none\n',
                       'redata: Number of spin components = 1\n',
                       declarations.replace('= 1', '= 2'),
                       declarations.replace('= none', '= polarized')):
        output = parse_output(incomplete + table)
        assert output['atom_moments_muB'] is None
        assert output['total_moment_muB'] is None
        assert output['spin_degenerate'] is False

def test_grid_reader_preserves_spin_components_and_checks_records(tmp_path):
    path = tmp_path / "rho"
    def record(payload):
        size = struct.pack("<i", len(payload))
        return size+payload+size
    content = record(np.eye(3, dtype="<f8").tobytes()) + record(np.array([2,1,1,2],dtype="<i4").tobytes())
    content += record(np.array([1.,2.],dtype="<f4").tobytes()) + record(np.array([3.,4.],dtype="<f4").tobytes())
    path.write_bytes(content)
    cell, grid = read_grid(path)
    assert grid.shape == (2,1,1,2)
    assert grid[1,0,0,1] == 4.
    path.write_bytes(content[:-1])
    with pytest.raises(ValueError):
        read_grid(path)

def test_mae_gate_checks_energy_difference_and_final_direction(tmp_path):
    def record(payload):
        size = struct.pack("<i", len(payload))
        return size + payload + size
    (tmp_path / "remag.RHO").write_bytes(
        record(np.eye(3,dtype="<f8").tobytes()) +
        record(np.array([1,1,1,4],dtype="<i4").tobytes()) +
        b"".join(record(np.array([value],dtype="<f4").tobytes()) for value in [1,1,0,0]))
    def run(theta, energy):
        return dict(valid=True, wall_seconds=1., hardware={"host":"same"},
                    benchmark=dict(atoms=2,spin="spin-orbit",input_sha256=str(theta),pseudopotentials={},
                                   orientation_family_sha256="same-family",theta_deg=theta),
                    observables=dict(total_energy_ev=energy, fermi_energy_ev=0.,scf_iterations=10,
                        forces_ev_ang=[[0.,0.,0.]],stress_ev_ang3=np.zeros((3,3)).tolist(),
                        total_moment_muB=[2.,0.,0.] if theta==90 else [0.,0.,2.],
                        atom_moments_muB=[[0.,0.,2.]],atom_net_charges_e=[0.],
                        eigenvalues={"values_ev":[[0.,1.]]},soc_energy_representation="included_in_nonlocal"))
    rx,rz,cx,cz=run(90,-1000.),run(0,-1000.001),run(90,-1000.),run(0,-1000.001)
    dirs=[tmp_path]*4
    assert compare_mae(rx,rz,cx,cz,directories=dirs)["passed"]
    cx["runtime_settings"] = {"REMAG_TRACE_IO": "1"}
    traced = compare(rx,cx,reference_dir=tmp_path,candidate_dir=tmp_path)
    assert traced["passed"] and not traced["performance_comparable"]
    assert traced["end_to_end_speedup"] is None
    assert traced["performance_reasons"] == ["Filesystem tracing perturbs timing"]
    cx["runtime_settings"]["REMAG_TRACE_IO"] = "0"
    cx["runtime_settings"]["REMAG_TRACE_CUDA"] = "1"
    traced = compare(rx,cx,reference_dir=tmp_path,candidate_dir=tmp_path)
    assert traced["passed"] and traced["end_to_end_speedup"] is None
    assert traced["performance_reasons"] == ["CUDA activity tracing perturbs timing"]
    cx["runtime_settings"]["REMAG_TRACE_CUDA"] = "0"
    cx["runtime_settings"]["REMAG_NONLOCAL_H0_SHADOW"] = "1"
    shadow = compare(rx,cx,reference_dir=tmp_path,candidate_dir=tmp_path)
    assert shadow["passed"] and shadow["end_to_end_speedup"] is None
    assert shadow["performance_reasons"] == ["H0 matrix shadow retains upstream assembly and full comparison exports"]
    cx["runtime_settings"]["REMAG_NONLOCAL_H0_SHADOW"] = "0"
    cx["runtime_settings"]["REMAG_NONLOCAL_H0_POISON_HOST"] = "1"
    poisoned = compare(rx,cx,reference_dir=tmp_path,candidate_dir=tmp_path)
    assert poisoned["passed"] and poisoned["end_to_end_speedup"] is None
    assert poisoned["performance_reasons"] == ["Host H0 poisoning is diagnostic"]
    cx["runtime_settings"]["REMAG_NONLOCAL_H0_POISON_HOST"] = "0"
    cx["observables"]["total_energy_ev"] += 1.e-5
    assert compare(rx,cx,reference_dir=tmp_path,candidate_dir=tmp_path)["passed"]
    assert not compare_mae(rx,rz,cx,cz,directories=dirs)["passed"]
    cx=copy.deepcopy(rx)
    cx["observables"]["total_moment_muB"]=[0.,0.,2.]
    assert not compare_mae(rx,rz,cx,cz,directories=dirs)["passed"]
    cx=copy.deepcopy(rx)
    cx["valid"]=False
    assert compare(rx,cx,reference_dir=tmp_path,candidate_dir=tmp_path)["end_to_end_speedup"] is None

def test_stale_outputs_cannot_enter_new_run(tmp_path):
    case=tmp_path/"input"
    case.mkdir()
    (case/"manifest.json").write_text('{"pseudopotentials": {}}')
    (case/"input.fdf").write_text("SystemLabel remag\n")
    (case/"remag.DM").write_bytes(b"stale state")
    executable=tmp_path/"siesta"
    executable.touch()
    with pytest.raises(ValueError,match="unexpected files"):
        run_case(case,executable,tmp_path/"run")
