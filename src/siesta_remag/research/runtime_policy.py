"""Bound native deployment policy, distinct from numerical acceptance."""
import ctypes
import json
import re
import subprocess


def validate(value):
    if (not isinstance(value, dict) or set(value) != {
            'format', 'cuda_runtime', 'cuda_driver_api', 'nvidia_driver_versions'}
            or value['format'] != 'REMAG-SCF-RUNTIME-POLICY/1'
            or any(type(value[k]) is not int or not 12040 <= value[k] < 100000
                   for k in ('cuda_runtime', 'cuda_driver_api'))
            or not isinstance(value['nvidia_driver_versions'], list)
            or not 1 <= len(value['nvidia_driver_versions']) <= 32
            or any(not isinstance(s, str) or not re.fullmatch(r'\d+\.\d+\.\d+', s)
                   for s in value['nvidia_driver_versions'])
            or len(set(value['nvidia_driver_versions'])) != len(value['nvidia_driver_versions'])):
        raise ValueError('Invalid compiled runtime admission policy')
    return value


def probe(library):
    """Pure metadata query, including in CPU builders. No CUDA/device calls."""
    lib = ctypes.CDLL(str(library))
    try: call = lib.remag_device_scf_runtime_policy
    except AttributeError: return None  # Historical builds retain their own holds.
    call.argtypes = []; call.restype = ctypes.c_char_p
    raw = call()
    if not raw or len(raw) > 4096: raise ValueError('Invalid compiled runtime policy payload')
    return validate(json.loads(raw))


def observe():
    lib = ctypes.CDLL('libcudart.so.12')
    result = {}
    for symbol, key in [('cudaRuntimeGetVersion', 'cuda_runtime'), ('cudaDriverGetVersion', 'cuda_driver_api')]:
        call = getattr(lib, symbol); call.argtypes = [ctypes.POINTER(ctypes.c_int)]; call.restype = ctypes.c_int
        v = ctypes.c_int(-1); status = call(ctypes.byref(v))
        if status != 0: raise ValueError(f'Runtime version query failed: {symbol} status={status}')
        result[key] = v.value
    response = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=10, check=True)
    result['nvidia_driver_versions'] = [line.strip() for line in response.stdout.splitlines() if line.strip()]
    return result


def check(policy, observed):
    validate(policy)
    if (not isinstance(observed, dict) or set(observed) != {
            'cuda_runtime', 'cuda_driver_api', 'nvidia_driver_versions'}
            or any(type(observed[k]) is not int or observed[k] != policy[k]
                   for k in ('cuda_runtime', 'cuda_driver_api'))
            or not isinstance(observed['nvidia_driver_versions'], list)
            or len(observed['nvidia_driver_versions']) != 1
            or observed['nvidia_driver_versions'][0] not in policy['nvidia_driver_versions']):
        raise ValueError('Native runtime profile is not admitted: '+json.dumps(observed, sort_keys=True))
    return dict(observed=observed, admitted=True, scientific_acceptance=False,
                scope='Runtime admission only; full native and scientific review still required')


def verify_saved(runtime, proof):
    """Recheck a collected worker proof without touching local CUDA hardware."""
    expected = dict(executable_sha256=runtime['executable_sha256'],
                    capabilities=runtime.get('capabilities'), actual_files_verified=True)
    if 'driver_policy' in runtime:
        policy = validate(runtime['driver_policy'])
        deployment = None
        if runtime.get('environment', {}).get('REMAG_DEVICE_SCF') == '1':
            try: observed = proof['deployment']['observed']
            except (KeyError, TypeError) as exc: raise ValueError('Missing native runtime admission evidence') from exc
            deployment = check(policy, observed)
        expected.update(driver_policy=policy, deployment=deployment)
    if proof != expected:
        raise ValueError('Collected native library/capability verification differs or is missing')
    return proof
