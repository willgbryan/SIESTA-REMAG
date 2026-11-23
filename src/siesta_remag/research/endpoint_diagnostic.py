"""Operator review for a fixed scalar diagnostic on a terminal ionic geometry."""
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms

from ..corpus import sha256
from ..discovery.common import load
from ..discovery.config import Settings
from ..discovery.inputs import prepare_case
from ..discovery.review import evaluated_structure, native_review
from .input_reproduction import verify_prepared_input
from .relaxation_execution import review_relaxation


def endpoint_order(endpoint, proposed):
    """Map exported sites onto STRUCT_OUT, allowing only wrapping and a proper frame rotation."""
    a, b = endpoint.cell.array, proposed.cell.array
    if (len(endpoint) != len(proposed) or np.linalg.det(a)*np.linalg.det(b) <= 0 or
            not np.allclose(a @ a.T, b @ b.T, atol=2e-7, rtol=0)):
        raise ValueError('Fixed scalar query must use the terminal cell metric')
    delta = proposed.get_scaled_positions()[:, None] - endpoint.get_scaled_positions()[None, :]
    distance = np.linalg.norm((delta - np.rint(delta)) @ a, axis=2)
    match = (distance <= 2e-7) & (proposed.numbers[:, None] == endpoint.numbers[None, :])
    if not (np.all(match.sum(axis=0) == 1) and np.all(match.sum(axis=1) == 1)):
        raise ValueError('Fixed scalar query must use the original endpoint sites and origin')
    return np.argmax(match, axis=1)


def review_endpoint_query(root, attempt, owner, plan, bundle):
    """Verify a clean step-limit exit and reconstruct the next scalar input from the endpoint."""
    from gpac.physical import verify_export
    config = plan['config']
    result = load(attempt/'execution.json')
    receipt = load(attempt/'receipt.json')
    grant = load(attempt/'authorization.json')
    report = attempt/'collected/query/report.json'
    stage = report.parent/'dft-relax'
    prior_config = grant['config']
    if (owner.get('status') != 'held_native_review' or owner.get('allocation_stopped') is not True or
            not owner.get('app_id') or owner['app_id'] != config['previous_app_id'] or
            receipt.get('app_id') != owner['app_id'] or
            any(result.get(k) != owner[k] for k in ('app_id','status','allocation_stopped')) or
            result.get('app_lifecycle', {}).get('state', '').lower() != 'stopped' or
            prior_config['authorization_id'] == config['authorization_id'] or
            grant['config_sha256'] != result['config_sha256'] or
            any(config.get(k) != prior_config.get(k) for k in ('runtime_sha256','image','budget_ledger')) or
            sha256(attempt/'runtime.json') != config['runtime_sha256'] or
            sha256(report) != config['relaxation_endpoint_report_sha256'] or
            result['review']['report_sha256'] != sha256(report)):
        raise ValueError('Endpoint diagnostic requires its stopped predecessor and original runtime')
    reservations = plan['ledger'].snapshot()['reservations']
    if not any(r['id'] == receipt['run_id'] and r['metadata'].get('app_id') == owner['app_id'] and
               r['metadata'].get('receipt') == str(attempt/'receipt.json') for r in reservations):
        raise ValueError('Endpoint diagnostic predecessor has no matching reservation')

    old_request = load(attempt/'campaign/query/request.json')
    reproduced = review_relaxation(attempt, old_request, plan['runtime'], require_current_code=False)
    native = load(report)
    settings = Settings.from_dict(old_request['settings'])
    rows = native['stages']
    static = native_review(stage, load(stage/'result.json'), settings)
    moves = re.findall(r'^\s*Begin Broyden opt\. move =\s*(\d+)\s*$', (stage/'stdout.log').read_text(), re.M)
    exit_status = load(stage/'execution-status.json')
    if (reproduced['healthy'] or reproduced['native_status'] != 'held_dft-relax' or
            reproduced['source_binding']['source_sha256'] != result['source_sha256'] or
            len(rows) != 1 or rows[0]['stage'] != 'dft-relax' or rows[0]['accepted'] or
            rows[0]['native_optimizer_converged'] is not False or not static['accepted'] or
            exit_status['timed_out'] is not False or exit_status['returncode'] != 0 or
            list(map(int, moves)) != list(range(settings.dft_steps+1))):
        raise ValueError('Endpoint diagnostic requires an SCF-complete exhausted ionic relaxation')
    manifest = load(stage/'manifest.json')
    request = load(bundle/'experiment.json')
    if (request.get('query') != {'kind':'scalar'} or request.get('fixed_geometry') is not True or
            request.get('basis_spec') is not None or request.get('soc_matrix') or
            any(request[k] != manifest[k] for k in ('kmesh','pseudopotentials','basis_size','basis_energy_shift_ry')) or
            any(request['settings'][k] != v for k,v in old_request['settings'].items()
                if k not in ('soc_seconds','total_dft_seconds')) or
            any(request['settings'][k] > old_request['settings'][k] for k in ('soc_seconds','total_dft_seconds'))):
        raise ValueError('Endpoint diagnostic requires a matched scalar electronic protocol')
    allowed = {'manifest.json', 'input.fdf', *manifest['pseudopotentials']}
    if ({p.name for p in (bundle/'scalar').iterdir()} != allowed or
            any(sha256(bundle/'scalar'/name) != checksum for name,checksum in manifest['pseudopotentials'].items())):
        raise ValueError('Endpoint diagnostic scalar input inventory differs')
    source = load(attempt/'campaign/structure.json')['geometry']
    endpoint = evaluated_structure(stage, source['atomic_numbers'])
    geometry = verify_export(load(bundle/'structure.json'))['geometry']
    proposed = Atoms(numbers=geometry['atomic_numbers'], cell=geometry['cell'],
                     scaled_positions=geometry['fractional'], pbc=True)
    order = endpoint_order(endpoint, proposed)
    spins = np.asarray(manifest['initial_moments_muB'])[order]
    with TemporaryDirectory() as temp:
        scalar = prepare_case(proposed, stage, Path(temp)/'scalar', Settings.from_dict(request['settings']),
                              mesh=manifest['kmesh'], spins=spins, basis_size=manifest['basis_size'],
                              basis_energy_shift_ry=manifest['basis_energy_shift_ry'], allow_psf=True, diagnostics=True)
        verify_prepared_input(scalar, bundle/'scalar')
    return str(report.relative_to(root))
