"""Guard the small upstream seam without recreating SIESTA's internal test framework."""
import importlib.util
from pathlib import Path
import shutil
import pytest

def test_patch_matches_pinned_source_and_rejects_second_application(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "upstream/siesta"
    if not source.exists():
        pytest.skip("Fetch pinned upstream to exercise integration patch")
    spec = importlib.util.spec_from_file_location("patch_siesta", root / "scripts/patch_siesta.py")
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    (tmp_path / "Src").mkdir()
    for filename in ("siesta.F", "diag.F90", "rhoofd.F90", "vmat.F90", "listsc.f", "diagg.F", "diagk.F",
                     "diag2g.F", "diag3g.F", "diag2k.F", "diag3k.F", "normalize_dm.F90",
                     "dhscf.F", "siesta_forces.F90", "compute_dm.F", "scfconvergence_test.F", "siesta_end.F",
                     "diagon.F", "fermid.F", "poison.F", "m_charge_add.F90", "m_hartree_add.F90",
                     "setup_hamiltonian.F", "mixer.F", "dfscf.f", "meshsubs.F",
                     "setup_H0.F", "matel.F90", "matel_table.F90", "kinefsm.F", "nlefsm.f", "overfsm.f90", "final_H_f_stress.F",
                     "siesta_analysis.F90", "state_analysis.F", "state_init.F", "mulliken.F", "partial_charges.F90", "m_new_dm.F90", "post_scf_work.F"):
        shutil.copyfile(source / "Src" / filename, tmp_path / "Src" / filename)
    shutil.copyfile(source / "CMakeLists.txt", tmp_path / "CMakeLists.txt")
    patch.patch(tmp_path)
    assert "remag_grid" in (tmp_path / "Src/vmat.F90").read_text()
    analysis=(tmp_path/'Src/siesta_analysis.F90').read_text()
    forces=(tmp_path/'Src/siesta_forces.F90').read_text()
    assert forces.index('remag_phase_enter(2,max(0,istep),0)') < forces.index('call remag_capture_soc(SCFconverged)') < forces.index('call post_scf_work(')
    assert analysis.index('ucell(1:3, 1:3) = ucell_last(1:3, 1:3)') < analysis.index('call remag_write_soc()')
    assert analysis.index('call remag_write_soc()') < analysis.index('if (fdf_get("Read-H-from-file", .false.))')
    for name in ('remag_soc_components','remag_soc_export'):
        assert (tmp_path/'Src'/f'{name}.F90').read_bytes()==(root/'native'/f'{name}.F90.in').read_bytes()
        assert f'Src/{name}.F90' in (tmp_path/'CMakeLists.txt').read_text()
    with pytest.raises(ValueError, match="already patched"):
        patch.patch(tmp_path)
