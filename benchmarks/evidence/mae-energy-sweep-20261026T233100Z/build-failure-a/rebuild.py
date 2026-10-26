"""Apply only the reviewed sweep changes to the last qualified staged image."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

stage = Path('/opt/energy-sweep-stage')
plan = json.loads((stage/'experiment.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
native = Path('/opt/hopper-native')
exe = Path('/opt/remag-candidate/Src/siesta')
lib = Path('/opt/hopper-build/libremag_gpu.so')
assert sha(exe) == plan['base_build']['candidate_executable_sha256']
assert sha(lib) == plan['base_build']['candidate_library_sha256']
assert {str(p): sha(p) for p in native.iterdir() if p.is_file()} == plan['base_build']['native_sources']
subprocess.run(['patch', '--batch', '--fuzz=0', '-p2', '-d', str(native), '-i', str(stage/'native.patch')], check=True)
assert {str(p): sha(p) for p in native.iterdir() if p.is_file()} == plan['candidate_native_sources']
cache = Path('/opt/remag-candidate/CMakeCache.txt').read_text().splitlines()
linked = Path(next(line.split('=', 1)[1] for line in cache if line.startswith('REMAG_NATIVE_DIR:')))
if linked.resolve() != native.resolve():
    # The last diagnostic rebuilt the runtime library only. Synchronize its
    # verified source set too, so the SIESTA rebuild cannot use older headers.
    # These are image sources, never wholesale copies of the dirty workspace.
    for path in plan['candidate_native_sources']:
        shutil.copy2(path, linked/Path(path).name)

source = Path('/opt/remag-candidate-siesta-src/Src')
original = {}
def replace(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, (path, old)
    original.setdefault(path.name, sha(path))
    path.write_text(text.replace(old, new))

replace(source/'setup_H0.F', '        integer(c_int) function remag_reuse_initial_mesh() bind(C)',
        '        integer(c_int) function remag_reuse_orbital_geometry() bind(C)\n'
        '          import c_int\n        end function\n'
        '        integer(c_int) function remag_reuse_initial_mesh() bind(C)')
replace(source/'setup_H0.F', '        call remag_dm_check(remag_reuse_initial_mesh())',
        '        call remag_dm_check(remag_reuse_orbital_geometry())\n'
        '        call remag_dm_check(remag_reuse_initial_mesh())')
# This admission change is deliberately staged only. Fixed geometry, independent
# atomic guesses and all other restrictions are still enforced by native code.
old = """          recompute_H_after_scf.or.fdf_isblock('REMAG.SpinRotations').or. &
          fdf_get('REMAG.ReuseOrientationGeometry',.false.).or. &"""
new = """          recompute_H_after_scf.or. &
          (fdf_isblock('REMAG.SpinRotations').neqv. &
           fdf_get('REMAG.ReuseOrientationGeometry',.false.)).or. &"""
replace(source/'remag_cycle.F90', old, new)
replace(source/'remag_cycle.F90',
        'requires one fixed-geometry SCF; no NetCDF, H recomputation or orientation loop',
        'requires fixed geometry and cached orientations; no NetCDF or H recomputation')
subprocess.run(['cmake', '-S', str(native), '-B', '/opt/hopper-build',
                '-DREMAG_BUILD_FROZEN_POTENTIAL=ON', '-DCMAKE_CUDA_ARCHITECTURES=90'], check=True)
subprocess.run(['cmake', '--build', '/opt/hopper-build', '--target', 'remag_gpu', '--parallel', '8'], check=True)
subprocess.run(['cmake', '--build', '/opt/remag-candidate', '--target', 'SIESTA.siesta', '--parallel', '8'], check=True)
receipt = dict(candidate_executable_sha256=sha(exe), candidate_library_sha256=sha(lib),
    native_sources={str(p): sha(p) for p in native.iterdir() if p.is_file()},
    original_fortran=original, candidate_fortran={name: sha(source/name) for name in original},
    staged_scf_admission_ceiling=32768, production_scf_admission_ceiling=128,
    frozen_build=True, frozen_runtime_enabled=False, cuda_architectures=[90], capacity_qualified=False)
(stage/'build.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({k: v for k, v in receipt.items() if k != 'native_sources'}, indent=2))
