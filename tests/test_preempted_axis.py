"""Offline regression for a provider interruption during the final native direction."""
import shutil
import pytest

from test_discovery import backend, bundle
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import digest, load, save
from siesta_remag.discovery.pipeline import run_dft, source_identity
from siesta_remag.research import preempted_axis as axis


@pytest.fixture
def interrupted(tmp_path, bundle, monkeypatch):
    root = tmp_path/'previous'
    exe = tmp_path/'native-fixture'
    exe.write_text('synthetic')
    runtime = dict(executable_sha256=sha256(exe), environment={}, threads=1)
    policy = dict(operation='relaxation', expected_source_code=source_identity(), discovery_diagnostics=True)
    save(root/'report.json', dict(policy=policy, status='running', experiments=[], invocation_sha256=digest(policy),
                                  runtime_verification={'fixture':True}))
    monkeypatch.setattr(axis, 'verify_saved', lambda runtime, proof: proof == {'fixture':True})
    original = backend([])
    def stop_at_y(case, executable, destination, **kw):
        if destination.name == 'soc-y':
            shutil.copytree(case, destination)
            (destination/'stdout.log').write_text('SCF cycle in progress')
            raise KeyboardInterrupt
        result = original(case, executable, destination, **kw)
        for suffix in ('KP', 'EIG'):
            (destination/('remag.'+suffix)).write_text('synthetic')
        return result
    with pytest.raises(KeyboardInterrupt):
        run_dft(bundle, root/'query', exe, executable_sha256=sha256(exe), runner=stop_at_y, diagnostics=True)
    spec = axis.snapshot_axis(root, bundle, runtime, policy, 'physics-'+'1'*32)
    return root, bundle, exe, spec


@pytest.mark.parametrize('changed', [None, 'prefix', 'seed', 'completed'])
def test_only_interrupted_direction_can_reach_solver(interrupted, tmp_path, changed):
    root, bundle, exe, spec = interrupted
    calls = []
    q = root/'query'
    if changed == 'prefix':
        (q/'soc-x/remag.EIG').write_text('changed')
    elif changed == 'seed':
        spec['seeds']['soc-y'] = '0'*64
    elif changed == 'completed':
        (q/'soc-y/0_NORMAL_EXIT').touch()
        with pytest.raises(ValueError, match='interrupted final'):
            axis.axis_runner(root, spec, solver=backend(calls))
        return
    out = tmp_path/'fresh'
    runner = axis.axis_runner(root, spec, solver=backend(calls))
    report = run_dft(bundle, out, exe, executable_sha256=sha256(exe), diagnostics=True, runner=runner)
    if changed:
        assert calls == [] and report['status'] == 'held_error'
        return
    assert [c[0] for c in calls] == ['soc-y'] and report['status'] == 'screened'
    save(tmp_path/'authorization.json', dict(config={'final_axis_checkpoint_sha256':digest(spec)}))
    axis.verify_collected_prefix(tmp_path, out, spec)
    assert load(q/'report.json')['status'] == 'running'
    (out/'soc-z/remag.KP').write_text('changed')
    with pytest.raises(ValueError, match='prefix differs'):
        axis.verify_collected_prefix(tmp_path, out, spec)


def test_final_solver_uses_pinned_seed_bytes_across_reconstruction(interrupted, tmp_path):
    root, bundle, exe, spec = interrupted
    out = tmp_path/'finished'
    used = []
    def observe_solver(case, binary, destination, **kw):
        used.append(sha256(kw['restart']['path']))
        return backend([])(case, binary, destination, **kw)
    dispatch = axis.axis_runner(root, spec, solver=observe_solver)
    def other_host(case, binary, destination, **kw):
        if destination.name == 'soc-y':
            from pathlib import Path
            import struct
            # Perturb one low bit in the synthetic regenerated density.
            p = Path(kw['restart']['path'])
            buf = bytearray(p.read_bytes())
            sizes = []
            i = 0
            while i < len(buf):
                size = struct.unpack_from('<i', buf, i)[0]
                sizes.append((i+4,size)); i += size+8
            start, size = next(s for s in reversed(sizes) if s[1] >= 8)
            buf[start] ^= 1
            p.write_bytes(buf)
            kw['restart'] = {'path':str(p),'sha256':sha256(p)}
        return dispatch(case, binary, destination, **kw)
    report = run_dft(bundle, out, exe, executable_sha256=sha256(exe), diagnostics=True, runner=other_host)
    assert report['status'] == 'screened' and used == [spec['seeds']['soc-y']]
    proof = load(out/'seed-y.reconstruction.json')
    assert proof['original_sha256'] != proof['host_regenerated_sha256']
