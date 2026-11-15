"""Source-bound preparation and saved-only feedback through the discovery path."""
from pathlib import Path

import numpy as np

from ..corpus import sha256
from ..discovery.common import load, save
from ..discovery.pipeline import prepare_bundle, verify_bundle
from ..discovery.review import evaluated_structure


def prepare_relaxation(export, pseudos, weights, output, *, settings=None, orb_threads=4):
    """Run the existing bounded CPU Orb preparation; never submit native work."""
    from ase import Atoms
    from ase.io import write
    from gpac.physical import verify_export
    verified=verify_export(export)
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    save(root/'structure.json',verified)
    g=verified['geometry']
    atoms=Atoms(numbers=g['atomic_numbers'],cell=g['cell'],scaled_positions=g['fractional'],pbc=True)
    write(root/'input.extxyz',atoms)
    save(root/'source-binding.json',dict(format='REMAG-RELAXATION-SOURCE/1',
        source_sha256=verified['source']['sha256'],export_sha256=verified['sha256'],
        input_sha256=sha256(root/'input.extxyz'),structure_file_sha256=sha256(root/'structure.json')))
    result=prepare_bundle(root/'input.extxyz',pseudos,weights,root/'bundle',settings,
                          orb_device='cpu',orb_threads=orb_threads)
    return dict(format='REMAG-RELAXATION-PREPARATION/1',source_sha256=verified['source']['sha256'],
        preparation=result,bundle=str(root/'bundle'),native_submitted=False)


def inspect_relaxation(export, bundle, report_path, **options):
    """Review accepted saved relaxation, retaining numeric/exact separation.

    Existing bundles are linked retrospectively by geometry and species/order.
    New preparation adds a source binding. Neither mode modifies the source.
    """
    from ase.io import read
    from gpac.physical import verify_export
    from gpac.feedback import analyze
    verified=verify_export(export);g=verified['geometry']
    bundle,report_path=Path(bundle).resolve(),Path(report_path).resolve()
    receipt=verify_bundle(bundle);request=load(bundle/'request.json');report=load(report_path)
    original=read(bundle/'input.extxyz')
    if (original.numbers.tolist()!=g['atomic_numbers'] or
        not np.allclose(original.cell.array,g['cell'],atol=1e-8,rtol=0) or
        not np.allclose(original.get_scaled_positions(wrap=False),g['fractional'],atol=1e-8,rtol=0)):
        raise ValueError('Authored source differs from discovery input geometry or site order')
    if report['prepared_bundle']!=receipt or report['input_sha256']!=request['input_sha256']:
        raise ValueError('Relaxation report belongs to another preparation')
    source_mode='retrospective geometry match within 1e-8 in cell angstrom/fractional coordinates; species/order exact'
    binding_path=bundle.parent/'source-binding.json'
    if binding_path.exists():
        binding=load(binding_path)
        if (binding.get('format')!='REMAG-RELAXATION-SOURCE/1' or binding['source_sha256']!=verified['source']['sha256'] or
            binding['export_sha256']!=verified['sha256'] or binding['input_sha256']!=request['input_sha256'] or
            sha256(bundle.parent/'input.extxyz')!=binding['input_sha256'] or
            sha256(bundle.parent/'structure.json')!=binding['structure_file_sha256']):
            raise ValueError('Prepared exact source binding differs')
        source_mode='authored source bound before CPU preparation'
    stage=next((s for s in report['stages'] if s['stage']=='dft-relax'),None)
    if stage is None or not stage['accepted'] or 'relaxation' not in report:
        raise ValueError('No accepted native relaxation available')
    artifacts=report['accepted_artifacts']
    required={'dft-relax/remag.STRUCT_OUT','dft-relaxed.extxyz'}
    if not required<=set(artifacts):raise ValueError('Missing accepted relaxed geometry artifacts')
    for name,expected in artifacts.items():
        path=(report_path.parent/name).resolve()
        if not path.is_relative_to(report_path.parent) or sha256(path)!=expected:
            raise ValueError('Accepted relaxation artifact changed: '+name)
    atoms=evaluated_structure(report_path.parent/'dft-relax',g['atomic_numbers'])
    saved=read(report_path.parent/'dft-relaxed.extxyz')
    if (saved.numbers.tolist()!=atoms.numbers.tolist() or
        not np.allclose(saved.cell.array,atoms.cell.array,atol=1e-8,rtol=0) or
        not np.allclose(saved.positions,atoms.positions,atol=1e-7,rtol=0)):
        raise ValueError('Saved relaxation differs from accepted STRUCT_OUT')
    observed=dict(format='GPAC-PHYSICAL-GEOMETRY/1',units='angstrom',cell=atoms.cell.array.tolist(),
        fractional=atoms.get_scaled_positions(wrap=False).tolist(),atomic_numbers=atoms.numbers.tolist(),site_ids=g['site_ids'])
    feedback=analyze(verified,observed,source_sha256=verified['source']['sha256'],
        calculation=dict(id=report['invocation_sha256'],report_sha256=sha256(report_path),
                         executable_sha256=report['executable_sha256']),**options)
    return dict(format='REMAG-RELAXATION-FEEDBACK/1',source_binding=source_mode,
        source_sha256=verified['source']['sha256'],report_sha256=sha256(report_path),
        calculation_status=report['status'],feedback=feedback,solver_executed=False)
