"""One authored reference calculation followed by reusable magnetic responses."""
import json
from pathlib import Path
import time
import numpy as np

from .authoring import state_metadata, prepare_authored_input
from .state import import_native_state, file_hash
from .realspace import load_realspace
from .force_theorem import calculate_mft
from .greens import calculate_exchange
from .descriptors import summarize_magnetic_screen


def run_screen(request, runtime, output, *, checkpoint=lambda: None):
    """Run inside the existing single-GPU allocation, never create one here."""
    from ..runner import run_case
    from ..hsx import read_hsx
    root = Path(output)
    started = time.perf_counter()
    def save(name, value):
        (root/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    structure = request['structure']
    save('structure.json', structure)
    inputs = prepare_authored_input(structure, runtime['template'], root/'inputs')
    (root/'rotations.txt').write_text('.7071067811865476 0 .7071067811865476 0\n')
    env = dict(runtime['settings'], REMAG_INITIAL_MESH='0', REMAG_ENERGY_ONLY='1',
        REMAG_FROZEN_ROTATIONS=str(root/'rotations.txt'), REMAG_FROZEN_SOC_FREE='1',
        REMAG_EXPORT_MAGNETIC_STATE='1')
    result = run_case(inputs, Path(runtime['executable']), root/'reference',
                      threads=1, timeout=runtime['scf_timeout_seconds'], extra_env=env)
    ledger = [dict(role='reference_scf', valid=result['valid'], seconds=result['wall_seconds'])]
    save('ledger.json', ledger)
    checkpoint()
    if not result['valid']:
        raise RuntimeError('Reference SCF failed; inspect reference/result.json and native logs')
    metadata = state_metadata(structure, read_hsx(root/'reference/remag.HSX'),
        calculation_id=root.name+'/reference', settings=runtime['electronic_settings'],
        evidence=[dict(path=str(root/'reference/result.json'), sha256=file_hash(root/'reference/result.json'))])
    state = import_native_state(root/'reference', metadata)
    state.save(root/'state')
    model = load_realspace(root/'reference', state.metadata)
    screens = []
    for mesh in request['meshes']:
        start = time.perf_counter()
        sample = model.sample(mesh)
        label = 'k'+'x'.join(map(str, mesh))
        mft = calculate_mft(sample, request['directions'], angular_model=request['angular_model'],
            check_soc_free=False, memory_limit_bytes=8*1024**3, output=root/label)
        exchange = (calculate_exchange(sample, radius_angstrom=request['exchange_radius_angstrom'])
                    if mesh == request['exchange_mesh'] else None)
        screen = summarize_magnetic_screen(sample, mae=mft, exchange=exchange)
        name = label+'/magnetic-screen.json'
        save(name, screen)
        screens.append(dict(mesh=mesh, path=name, sha256=file_hash(root/name)))
        ledger.append(dict(role='frozen_screen', mesh=mesh, valid=True,
            seconds=time.perf_counter()-start, mft_seconds=mft['performance']['total_wall_seconds'],
            exchange_seconds=None if exchange is None else exchange['performance']['total_wall_seconds']))
        save('ledger.json', ledger)
        checkpoint()
        del sample
    return dict(format='REMAG-MAGNETIC-RUN/1', status='complete',
        identity={k:metadata[k] for k in ('gpac_source_hash','evaluated_geometry_hash','calculation_id')},
        reference_converged=True, screens=screens, seconds=time.perf_counter()-started,
        physical_accuracy_qualified=False, reference_settings=runtime['electronic_settings'])
