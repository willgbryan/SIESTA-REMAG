"""Check the actual patched Fortran formats across a sub-micro-eV difference."""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from siesta_remag.output import parse_output


def test_printed_directional_energies_round_trip(tmp_path):
    root = Path(__file__).resolve().parents[1]
    compiler = shutil.which("gfortran")
    source = root / "upstream/siesta/Src/write_subs.F"
    if not compiler or not source.exists():
        pytest.skip("Pinned SIESTA and gfortran required for energy formatting test")
    spec = importlib.util.spec_from_file_location("patch_siesta", root / "scripts/patch_siesta.py")
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    (tmp_path / "Src").mkdir()
    shutil.copyfile(source, tmp_path / "Src/write_subs.F")
    patch.patch_property_precision(tmp_path)
    text = (tmp_path / "Src/write_subs.F").read_text()
    # Execute the formats taken from the patched energy decomposition and final
    # summary, including Fortran format reversion for multiple energy rows.
    decomposition = re.search(r"write\(6,'([^']+)'\)\s*\n.*Program''s energy", text)[1]
    final = re.search(r"Final energy \(eV\):'\s*\n\s*write\(6,'([^']+)'", text)[1]
    program = f"""program check
  use iso_fortran_env, only: real64
  implicit none
  real(real64) :: e, f
  read(*,*) e, f
  write(6,'{decomposition}') 'siesta: energies', &
    'siesta: Etot =', e, 'siesta: FreeEng =', f
  write(6,'{final}') 'siesta: ', 'Total =', e
end program
"""
    (tmp_path / "check.f90").write_text(program)
    subprocess.run([compiler, "check.f90", "-o", "check"], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    values = [(-100000.12345678901, -100000.23456789012),
              (-100000.12345668902, -100000.23456779012)]
    parsed = []
    for energy, free_energy in values:
        result = subprocess.run([str(tmp_path / "check")], input=f"{energy!r} {free_energy!r}\n",
                                check=True, capture_output=True, text=True)
        observables = parse_output(result.stdout)
        assert observables["total_energy_ev"] == energy
        assert observables["free_energy_ev"] == free_energy
        parsed.append(observables)
    for key, column in (("total_energy_ev", 0), ("free_energy_ev", 1)):
        assert parsed[1][key] - parsed[0][key] == values[1][column] - values[0][column]
        assert 0 < parsed[1][key] - parsed[0][key] < 1e-6
    with pytest.raises(ValueError, match="already patched"):
        patch.patch_property_precision(tmp_path)


def test_missing_free_energy_is_unknown():
    assert parse_output("siesta: Total = -12.0\n")["free_energy_ev"] is None
