"""One original-input setup diagnostic with a mandatory pre-SCF exit interposer."""
import hashlib
import os
from pathlib import Path
import sys
import uuid
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src') if modal.is_local() else '/opt/native-probe/python')
from siesta_remag.discovery.common import load, save, digest
from siesta_remag.research.budget import BudgetLedger
from siesta_remag.research.native_setup_probe import validate_policy, barrier_reached, summarize

if modal.is_local():
    folder = Path(os.environ['REMAG_NATIVE_SETUP_PROBE']).resolve()
    policy = validate_policy(load(folder/'input/policy.json'))
    for name, sha in policy['files'].items():
        if hashlib.sha256((folder/'input'/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Changed frozen input: '+name)
    for path, key in [(ROOT/'src/siesta_remag/research/native_setup_probe.py', 'helper_sha256'),
                      (Path(__file__), 'module_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != policy[key]:
            raise ValueError('Changed diagnostic code')
    budget = BudgetLedger(folder/'budget.json')
    if budget.remaining_cents() < policy['reservation_cents']:
        raise ValueError('Insufficient diagnostic reservation')
    image = (modal.Image.from_id(policy['image'])
        .add_local_dir(folder/'input', '/opt/native-probe/input', copy=True)
        .add_local_dir(ROOT/'src', '/opt/native-probe/python', copy=True, ignore=['__pycache__']))
else:
    policy = validate_policy(load('/opt/native-probe/input/policy.json')); image = None
app = modal.App('siesta-remag-single-gpu')
volume = modal.Volume.from_name(policy['volume'], create_if_missing=False)


@app.function(image=image, gpu='A100-80GB:1', cpu=4, memory=16384,
    timeout=180, max_containers=1, min_containers=0, buffer_containers=0,
    scaledown_window=2, retries=0, volumes={'/results': volume})
def probe(run_id):
    import resource, shutil, subprocess, time
    from siesta_remag.discovery.collection import pack
    import siesta_remag.research.native_setup_probe as helper
    root = Path('/results')/run_id; root.mkdir(exist_ok=False)
    inp = Path('/opt/native-probe/input'); runtime = load(inp/'runtime.json')
    report = {'status': 'running', 'invocation_sha256': digest(policy),
        'scope': 'Original Ni initialization only, mandatory exit before SCF cycle',
        'native_runtime_repaired': False, 'scientific_acceptance': False}
    def flush(): save(root/'report.json', report); volume.commit()
    def command(args, name, *, env=None, cwd=None, stdin=None, timeout=20):
        with (root/(name+'-stdout.log')).open('w') as out, (root/(name+'-stderr.log')).open('w') as err:
            p = subprocess.run(args, stdout=out, stderr=err, env=env, cwd=cwd, stdin=stdin, timeout=timeout)
        return p.returncode
    flush()
    try:
        if hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest() != policy['helper_sha256']:
            raise ValueError('Uploaded helper changed')
        for name, sha in policy['files'].items():
            if hashlib.sha256((inp/name).read_bytes()).hexdigest() != sha: raise ValueError('Uploaded input changed')
        exe = Path(runtime['executable']); lib = Path(runtime['capabilities']['library'])
        for path, sha in [(exe, runtime['executable_sha256']), (lib, runtime['capabilities']['library_sha256'])]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha: raise ValueError('Retained runtime changed')
        env = {k: v for k, v in os.environ.items() if not k.startswith('REMAG_') and k not in ('LD_PRELOAD', 'LD_DEBUG', 'LD_DEBUG_OUTPUT')}
        env.update(runtime['environment'])
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        shim = root/'diagnostic-interpose.so'
        args = ['g++', '-std=c++17', '-shared', '-fPIC', '-I/usr/local/cuda/include',
            str(inp/'probe_cycle_barrier.cpp'), str(inp/'probe_graph_interpose.cpp'),
            '-L/usr/local/cuda/lib64', '-lcudart', '-ldl', '-o', str(shim)]
        report['compile_command'] = args
        if command(args, 'compile', env=env): raise RuntimeError('Diagnostic shim compilation failed')
        report['shim_sha256'] = hashlib.sha256(shim.read_bytes()).hexdigest()
        # A real Linux dynamic-link test must stop before invoking the dummy body.
        test = root/'preload-test'; test.mkdir()
        (test/'dummy.cpp').write_text('extern "C" int remag_cycle_execute(int,int*,int*){return 0;}')
        (test/'caller.cpp').write_text('extern "C" int remag_cycle_execute(int,int*,int*);int main(){return remag_cycle_execute(7,nullptr,nullptr);}')
        if command(['g++', '-shared', '-fPIC', str(test/'dummy.cpp'), '-o', str(test/'libdummy.so')], 'compile-dummy', env=env):
            raise RuntimeError('Dummy compilation failed')
        if command(['g++', str(test/'caller.cpp'), '-L'+str(test), '-Wl,-rpath,'+str(test), '-ldummy', '-o', str(test/'caller')], 'compile-caller', env=env):
            raise RuntimeError('Caller compilation failed')
        if command([str(test/'caller')], 'uninterposed', env=env, cwd=test) != 0: raise RuntimeError('Uninterposed control failed')
        probe_env = dict(env, LD_PRELOAD=str(shim), LD_BIND_NOW='1')
        code = command([str(test/'caller')], 'interposed', env=probe_env, cwd=test)
        marker = load(test/'cycle-barrier.json') if (test/'cycle-barrier.json').exists() else None
        report['preload_test'] = {'returncode': code, 'marker': marker}
        if not barrier_reached(code, marker) or marker['geometry'] != 7: raise RuntimeError('Preload barrier failed')
        if command(['readelf', '-Ws', str(exe)], 'executable-symbols', env=env): raise RuntimeError('Symbol inspection failed')
        symbols = (root/'executable-symbols-stdout.log').read_text().splitlines()
        if not any(' UND ' in line and line.split()[-1] == 'remag_cycle_execute' for line in symbols):
            raise RuntimeError('Executable cycle entry is not dynamically interposable')
        if command(['nvidia-smi'], 'nvidia-smi', env=env): raise RuntimeError('Device observation failed')
        native = root/'native'; native.mkdir()
        for name in ('input.fdf', 'Ni.psml'): shutil.copyfile(inp/name, native/name)
        # Immediate symbol binding records the actual executable-to-barrier link.
        probe_env.update(LD_DEBUG='bindings', LD_DEBUG_OUTPUT=str(native/'loader'))
        flush(); start = time.monotonic(); timed_out = False; code = None
        with (native/'input.fdf').open('rb') as f:
            try: code = command([str(exe)], 'native', env=probe_env, cwd=native, stdin=f, timeout=60)
            except subprocess.TimeoutExpired: timed_out = True
        report['native'] = dict(summarize(native, code, timed_out), wall_seconds=time.monotonic()-start)
        # Preserve DOT and loader text through the standard verified .log archive.
        for p in list(native.glob('graph-before-*.dot')) + list(native.glob('loader.*')):
            p.rename(p.with_name(p.name+'.log'))
        loader = '\n'.join(p.read_text(errors='replace') for p in native.glob('loader.*.log'))
        report['cycle_binding_verified'] = any(str(exe) in line and str(shim) in line and 'remag_cycle_execute' in line for line in loader.splitlines())
        report['graph_dumps'] = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in native.glob('graph-before-*.dot.log')}
        report['status'] = 'diagnostic_complete' if report['cycle_binding_verified'] else 'diagnostic_binding_unverified'
    except Exception as error:
        report.update(status='diagnostic_error', error={'type': type(error).__name__, 'message': str(error)})
    flush(); package = pack(root, include_native=True); volume.commit(); return package


@app.local_entrypoint()
def main():
    from siesta_remag.discovery.collection import collect
    receipt = folder/'receipt.json'; output = folder/'collected'
    if receipt.exists() or output.exists(): raise ValueError('Existing attempt; never resubmit')
    run_id = 'native-setup-'+uuid.uuid4().hex
    budget.reserve(run_id, policy['reservation_cents'], metadata={'app_id': app.app_id, 'receipt': str(receipt), 'policy_sha256': digest(policy)})
    row = {'status': 'submitting', 'run_id': run_id, 'app_id': app.app_id, 'volume': policy['volume'], 'reserved_cents': policy['reservation_cents']}
    save(receipt, row)
    call = probe.spawn(run_id); row.update(status='running', function_call_id=call.object_id); save(receipt, row); budget.record(run_id, row)
    collect(receipt, output, timeout=240); budget.record(run_id, load(receipt))
