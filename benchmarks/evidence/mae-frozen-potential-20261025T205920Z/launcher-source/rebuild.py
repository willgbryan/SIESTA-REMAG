import hashlib,json,shutil,subprocess
from pathlib import Path

stage=Path('/opt/frozen-stage')
plan=json.loads((stage/'experiment.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
native=Path('/opt/hopper-native');exe=Path('/opt/remag-candidate/Src/siesta');lib=Path('/opt/hopper-build/libremag_gpu.so')
assert sha(exe)==plan['base_build']['candidate_executable_sha256']
assert sha(lib)==plan['base_build']['candidate_library_sha256']
assert {str(p):sha(p) for p in native.iterdir() if p.is_file()}==plan['base_build']['native_sources']
cache=Path('/opt/remag-candidate/CMakeCache.txt').read_text().splitlines()
linked=Path(next(line.split('=',1)[1] for line in cache if line.startswith('REMAG_NATIVE_DIR:')))
for name,digest in plan['candidate_sources'].items():
    p=stage/'native'/name
    assert sha(p)==digest
    for target in dict.fromkeys([native,linked]):shutil.copy2(p,target/name)
# Carry forward the already-qualified shared-export initialization correction.
source=Path('/opt/remag-candidate-siesta-src')
p=source/'Src/final_H_f_stress.F';text=p.read_text()
old='      if(remag_energy_only_enabled()==1)then\n        call remag_cycle_energy_finish()\n      else\n      not_using_auxcell = (no_s == no_u)\n'
new='      not_using_auxcell = (no_s == no_u)\n      if(remag_energy_only_enabled()==1)then\n        call remag_cycle_energy_finish()\n      else\n'
assert text.count(old)==1;p.write_text(text.replace(old,new))
for src,build,target in [(native,Path('/opt/hopper-build'),'remag_gpu'),(source,Path('/opt/remag-candidate'),'SIESTA.siesta')]:
    subprocess.run(['cmake','-S',str(src),'-B',str(build),'-DREMAG_BUILD_FROZEN_POTENTIAL=ON','-DCMAKE_CUDA_ARCHITECTURES=90'],check=True)
    subprocess.run(['cmake','--build',str(build),'--target',target,'--parallel','8'],check=True)
receipt=dict(candidate_executable_sha256=sha(exe),candidate_library_sha256=sha(lib),
 native_sources={str(p):sha(p) for p in native.iterdir() if p.is_file()},
 final_force_source_sha256=sha(source/'Src/final_H_f_stress.F'),
 staged_scf_admission_ceiling=32768,production_scf_admission_ceiling=128,
 experimental_build=True,cuda_architectures=[90],capacity_qualified=False)
(stage/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='native_sources'},indent=2))
