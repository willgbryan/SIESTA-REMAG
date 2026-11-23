"""Crash evidence regressions; the deliberate C++ signal is not the CUDA fault."""
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import pytest
import psutil

from siesta_remag.native_diagnostics import NativeProcessDiagnostics, parse_maps
from siesta_remag.corpus import sha256
from siesta_remag.discovery.collection import pack, unpack, verify_archive
from siesta_remag.runner import run_case


@pytest.fixture(scope='module')
def native_probe(tmp_path_factory):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler required')
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp('native-crash-probe')/'probe'
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I'+str(repo/'tests/fixtures/scf_setup_cuda'), '-I'+str(repo/'native'),
                    str(repo/'tests/fixtures/scf_setup_driver.cpp'), '-o', str(output)],
                   check=True, capture_output=True, text=True)
    return output


def checkpoints(root):
    return [json.loads(line) for line in (root/'remag-scf-setup.log').read_text().splitlines()]


@pytest.mark.parametrize('stage,expected_calls', [('instantiate',1),('upload',2),('synchronize',3)])
@pytest.mark.parametrize('failure', ['crash','error'])
def test_setup_evidence_survives_failure_without_retry(native_probe, tmp_path, stage, expected_calls, failure):
    result = subprocess.run([str(native_probe), failure+'-'+stage], cwd=tmp_path, capture_output=True, text=True)
    rows = checkpoints(tmp_path)
    assert result.returncode == (-signal.SIGSEGV if failure == 'crash' else 7)
    assert rows[-1]['phase'] == stage+('_begin' if failure == 'crash' else '_end')
    assert rows[-1]['status'] == (-1 if failure == 'crash' else 7)
    if failure == 'error':
        assert result.stdout.strip() == f'7 {expected_calls}'
    assert all(r['graph_nodes'] == 120985 and r['jobs'] == 1896 for r in rows)
    assert rows[-1]['runtime_version'] == 12060
    assert rows[-1]['driver_cuda_version'] == 13030


def test_supported_runtime_success(native_probe, tmp_path):
    result = subprocess.run([str(native_probe), 'success'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == '0 3'
    rows = checkpoints(tmp_path)
    assert [r['phase'] for r in rows] == ['version_query_begin','instantiate_begin','instantiate_end',
                                        'upload_begin','upload_end','synchronize_begin','synchronize_end']
    assert rows[-1]['driver_cuda_version'] == 13030 and rows[-1]['driver_query_status'] == 0


@pytest.mark.parametrize('mode,version,status', [('unknown-version',-1,9),('unadmitted-driver',13050,0)])
def test_unadmitted_driver_stops_before_graph_calls(native_probe, tmp_path, mode, version, status):
    result = subprocess.run([str(native_probe), mode], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 801 % 256 and result.stdout.strip() == '801 0'
    rows = checkpoints(tmp_path)
    assert [r['phase'] for r in rows] == ['version_query_begin','runtime_rejected']
    assert rows[-1]['driver_cuda_version'] == version and rows[-1]['driver_query_status'] == status


def test_log_failure_does_not_mask_original_api_failure(native_probe, tmp_path):
    (tmp_path/'remag-scf-setup.log').mkdir()
    result = subprocess.run([str(native_probe), 'error-upload'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 7 and result.stdout.strip() == '7 2'
    assert 'cannot open SCF setup checkpoint' in result.stderr


MAPS = ('7f001000-7f003000 r-xp 00002000 08:01 102 /driver/libcuda.so.1\n'
        '7f005000-7f006000 rw-p 00000000 00:00 0\n'
        '7f008000-7f00a000 rw-p 00000000 00:00 0 [stack]\n')


def fake_proc(tmp_path):
    proc = tmp_path/'proc'; child = proc/'42'; child.mkdir(parents=True)
    (child/'maps').write_text(MAPS)
    (child/'limits').write_text('Max stack size            8388608              unlimited            bytes\n')
    driver = proc/'driver/nvidia'; driver.mkdir(parents=True)
    (driver/'version').write_text('NVRM version: fake local test driver\n')
    return proc, child


def test_last_live_mapping_retained_on_exit_race(tmp_path):
    proc, child = fake_proc(tmp_path)
    observer = NativeProcessDiagnostics(42, proc_root=proc, platform='linux')
    observer.sample(.1)
    (child/'maps').unlink(); observer.sample(.2)
    assert observer.record['last_error'] is None  # bounded cadence
    observer.sample(.7)
    record = observer.finish(-11, False)
    assert record['status'] == 'sampled' and record['returncode'] == -11
    assert record['snapshot']['seconds'] == .1
    assert record['snapshot']['maps'][0] == dict(start='0x7f001000',end='0x7f003000',permissions='r-xp',
        file_offset='0x00002000', device='08:01', inode=102, path='/driver/libcuda.so.1')
    assert record['snapshot']['maps'][1]['path'] is None
    assert '8388608' in record['snapshot']['limits']
    assert record['last_error']['seconds'] == .7


def test_bounded_invalid_and_unavailable_procfs_are_explicit(tmp_path):
    proc, child = fake_proc(tmp_path)
    (child/'maps').write_bytes(b'x'*(1024*1024+1))
    observer = NativeProcessDiagnostics(42, proc_root=proc, platform='linux')
    observer.sample(0)
    assert observer.record['snapshot'] is None and observer.record['status'] == 'unavailable'
    assert 'byte limit' in observer.record['last_error']['error']
    (child/'maps').write_text('not a map\n'); observer.sample(1)
    assert 'Unrecognized' in observer.record['last_error']['error']
    for bad in ('', '0010-0000 r-xp 0000 00:00 0'):
        with pytest.raises(ValueError): parse_maps(bad)
    traced = NativeProcessDiagnostics(42, traced=True, proc_root=proc, platform='linux')
    traced.sample(0)
    assert traced.record['snapshot'] is None and 'tracer parent' in traced.record['reason']
    assert NativeProcessDiagnostics(42, platform='darwin').record['status'] == 'unavailable'


def test_missing_limits_keeps_mapping(tmp_path):
    proc, child = fake_proc(tmp_path); (child/'limits').unlink()
    observer = NativeProcessDiagnostics(42, proc_root=proc, platform='linux'); observer.sample(0)
    assert observer.record['status'] == 'sampled'
    assert observer.record['snapshot']['limits'] is None
    assert 'FileNotFoundError' in observer.record['snapshot']['limits_error']


@pytest.fixture
def case(tmp_path):
    # This input is only a provenance fixture; the C++ probe never reads it.
    case = tmp_path/'case'; case.mkdir(); (case/'input.fdf').write_text('fixture\n')
    (case/'manifest.json').write_text(json.dumps(dict(input_sha256=sha256(case/'input.fdf'),
        pseudopotentials={}, spin='collinear', atoms=1)))
    return case


def test_real_native_signal_retained_and_collected(native_probe, case, tmp_path, monkeypatch):
    monkeypatch.setattr('siesta_remag.runner.shutil.which', lambda _: None)
    output = tmp_path/'run'; output.mkdir()
    result = run_case(case, native_probe, output/'scalar', timeout=5)
    assert result['returncode'] == -signal.SIGSEGV and not result['valid'] and not result['timed_out']
    record = json.loads((output/'scalar/native-process.json').read_text())
    assert record['returncode'] == -signal.SIGSEGV
    if sys.platform.startswith('linux'):
        assert record['status'] == 'sampled', record
        assert any(r['path'] == str(native_probe) for r in record['snapshot']['maps'])
    else:
        assert record['status'] == 'unavailable'
    assert checkpoints(output/'scalar')[-1]['phase'] == 'instantiate_begin'
    (output/'report.json').write_text(json.dumps(dict(status='held_error',invocation_sha256='fixture')))
    collected = tmp_path/'collected'; unpack(pack(output), collected)
    assert checkpoints(collected/'scalar') == checkpoints(output/'scalar')
    assert json.loads((collected/'scalar/native-process.json').read_text()) == record
    (collected/'scalar/native-process.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_archive(collected)


def test_failure_context_precedes_native_output_parsing(native_probe, case, tmp_path, monkeypatch):
    monkeypatch.setattr('siesta_remag.runner.shutil.which', lambda _: None)
    def bad_output(_):
        raise ValueError('truncated native output')
    monkeypatch.setattr('siesta_remag.runner.parse_output', bad_output)
    def exited(_):
        raise psutil.NoSuchProcess(42)
    monkeypatch.setattr('siesta_remag.runner.psutil.Process', exited)
    output = tmp_path/'run'
    with pytest.raises(ValueError, match='truncated native output'):
        run_case(case, native_probe, output, timeout=5)
    assert json.loads((output/'execution-status.json').read_text())['returncode'] == -signal.SIGSEGV
    assert json.loads((output/'native-process.json').read_text())['returncode'] == -signal.SIGSEGV
    assert checkpoints(output)[-1]['phase'] == 'instantiate_begin'
