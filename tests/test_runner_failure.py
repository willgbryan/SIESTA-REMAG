"""Real local dummy processes prove failure persistence; no solver or GPU."""
import pytest

from test_discovery import bundle
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load
from siesta_remag.discovery.config import Settings
from siesta_remag.discovery.inputs import prepare_case
from siesta_remag.discovery.pipeline import run_dft
from siesta_remag.runner import run_case


def executable(tmp_path, code):
    p = tmp_path/'dummy-native'
    ending = 'kill -SEGV $$' if code < 0 else f'exit {code}'
    p.write_text('#!/bin/sh\nulimit -c 0\nprintf "{bad-json" > remag-device-scf.json\n'+ending+'\n')
    p.chmod(0o755)
    return p


def case(bundle, tmp_path):
    from ase.io import read
    return prepare_case(read(bundle/'input.extxyz'), bundle/'pseudos', tmp_path/'case', Settings())


@pytest.mark.parametrize('code', [-11, 42, 0])
def test_postprocessing_failure_retains_exit_and_raw_evidence(bundle, tmp_path, code):
    exe = executable(tmp_path, code)
    output = tmp_path/'attempt'
    result = run_case(case(bundle,tmp_path), exe, output, extra_env={'REMAG_DEVICE_SCF':'1'})
    assert result['returncode'] == code and not result['valid'] and not result['timed_out']
    assert result['postprocessing_error']['type'] == 'JSONDecodeError'
    assert result['wall_seconds'] == load(output/'execution-status.json')['wall_seconds']
    assert result['wall_seconds'] > 0 and not result['postprocessing_complete']
    assert (output/'remag-device-scf.json').read_text() == '{bad-json'
    assert load(output/'result.json') == result
    assert (output/'native-process.json').is_file()
    assert 'JSONDecodeError' in load(output/'postprocessing-error.json')['traceback']
    if code == -11:
        assert 'SIGSEGV' in result['parse_errors'][0]
    # Existing attempt evidence must never be rewritten after a failed reuse.
    hashes = {p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError): run_case(tmp_path/'case', exe, output)
    assert hashes == {p.name:sha256(p) for p in output.iterdir() if p.is_file()}


def test_discovery_retains_failed_stage_duration_and_stops_soc(bundle, tmp_path):
    exe = executable(tmp_path, -11)
    report = run_dft(bundle, tmp_path/'run', exe, executable_sha256=sha256(exe),
                     runtime_env={'REMAG_DEVICE_SCF':'1'})
    assert report['status'] == 'held_dft-relax', report
    assert [s['stage'] for s in report['stages']] == ['dft-relax']
    assert report['solver_seconds'] > 0
    assert not report['stages'][0]['accepted']
    assert 'SIGSEGV' in report['reason'] and 'JSONDecodeError' in report['reason']
    assert report['screen'] is None and not report['accepted_artifacts']
    assert not any(p.name.startswith('soc-') for p in (tmp_path/'run').iterdir())


def test_input_validation_failure_is_still_raised(bundle, tmp_path):
    prepared = case(bundle,tmp_path)
    (prepared/'input.fdf').write_text('tampered')
    with pytest.raises(ValueError, match='Input differs'):
        run_case(prepared, executable(tmp_path,0), tmp_path/'attempt')
    assert not (tmp_path/'attempt/result.json').exists()
