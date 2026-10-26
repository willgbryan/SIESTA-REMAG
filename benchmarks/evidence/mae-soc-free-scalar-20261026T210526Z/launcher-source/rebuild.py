import hashlib,json,shutil,subprocess
from pathlib import Path
stage=Path('/opt/frozen-stage');plan=json.loads((stage/'experiment.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
native=Path('/opt/hopper-native');exe=Path('/opt/remag-candidate/Src/siesta');lib=Path('/opt/hopper-build/libremag_gpu.so')
assert sha(exe)==plan['base_build']['candidate_executable_sha256']
assert sha(lib)==plan['base_build']['candidate_library_sha256']
assert {str(p):sha(p) for p in native.iterdir() if p.is_file()}==plan['base_build']['native_sources']
for name,digest in plan['candidate_sources'].items():
    source=stage/'native'/name;assert sha(source)==digest;shutil.copy2(source,native/name)
subprocess.run(['cmake','-S',str(native),'-B','/opt/hopper-build','-DREMAG_BUILD_FROZEN_POTENTIAL=ON','-DCMAKE_CUDA_ARCHITECTURES=90'],check=True)
subprocess.run(['cmake','--build','/opt/hopper-build','--target','remag_gpu','--parallel','8'],check=True)
receipt=dict(candidate_executable_sha256=sha(exe),candidate_library_sha256=sha(lib),native_sources={str(p):sha(p) for p in native.iterdir() if p.is_file()},staged_scf_admission_ceiling=32768,production_scf_admission_ceiling=128,experimental_build=True,cuda_architectures=[90],capacity_qualified=False)
(stage/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='native_sources'},indent=2))
