"""Cheap unconstrained ML pre-relaxation; invoked in a bounded subprocess."""
import importlib.metadata
from pathlib import Path
import sys
import time

import numpy as np

from ..corpus import sha256
from .common import load, save
from .config import Settings
from .inputs import GPA_PER_EV_A3, validate_atoms


def relax(bundle, calculator=None):
    from ase.filters import FrechetCellFilter
    from ase.io import read, write
    from ase.optimize import FIRE

    bundle = Path(bundle)
    request = load(bundle / 'request.json')
    settings = Settings.from_dict(request['settings'])
    atoms = read(bundle / 'input.extxyz')
    validate_atoms(atoms, settings)
    start = time.monotonic()
    initial_volume = atoms.get_volume()
    directory = bundle / 'orb'
    directory.mkdir(exist_ok=False)
    weights = Path(request['orb_weights'])
    if calculator is None:
        if sha256(weights) != request['orb_weights_sha256']:
            raise ValueError('Orb weights changed after preparation')
        import torch
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        torch.set_num_threads(request['orb_threads'])
        model = pretrained.orb_v3_conservative_inf_mpa(weights_path=str(weights),
                    device=request['orb_device'], precision='float32-highest', compile=False)
        calculator = ORBCalculator(model, device=request['orb_device'])
    atoms.calc = calculator
    # Release exact positional symmetries once; no multistart sweep or constraints.
    atoms.rattle(stdev=0.01, seed=0)
    opt = FIRE(FrechetCellFilter(atoms), dt=0.1, maxstep=0.2,
               logfile=str(directory / 'relax.log'), trajectory=str(directory / 'relax.traj'))
    metrics = None
    try:
        for _ in opt.irun(fmax=0.0, steps=settings.orb_steps):
            validate_atoms(atoms, settings)
            ratio = atoms.get_volume() / initial_volume
            if not .5 < ratio < 2:
                raise ValueError('Orb volume changed by more than the bounded screen allows')
            metrics = dict(max_force_ev_ang=float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
                           max_stress_gpa=float(abs(atoms.get_stress()).max() * GPA_PER_EV_A3),
                           energy_ev=float(atoms.get_potential_energy()), volume_a3=float(atoms.get_volume()))
            if not all(np.isfinite(list(metrics.values()))):
                raise ValueError('Nonfinite Orb output')
            passed = metrics['max_force_ev_ang'] <= settings.orb_fmax_ev_ang and metrics['max_stress_gpa'] <= settings.orb_stress_gpa
            if passed or opt.nsteps % 10 == 0:
                write(directory / 'checkpoint.extxyz', atoms)
                save(directory / 'progress.json', dict(steps=opt.nsteps, **metrics))
            if passed:
                break
            if time.monotonic() - start >= settings.orb_seconds:
                raise TimeoutError('Orb time limit reached')
        write(directory / 'relaxed.extxyz', atoms)
        write(directory / 'relaxed.cif', atoms)
        result = dict(status='passed' if passed else 'held_step_limit', accepted=passed,
                      steps=opt.nsteps, metrics=metrics, geometry_sha256=sha256(directory / 'relaxed.extxyz'))
    except Exception as exc:
        write(directory / 'interrupted.extxyz', atoms)
        result = dict(status='held', accepted=False, error=str(exc), steps=opt.nsteps, metrics=metrics)
    result.update(wall_seconds=time.monotonic()-start, model='orb-v3-conservative-inf-mpa',
                  weights_sha256=request['orb_weights_sha256'], rattle_ang=0.01, rattle_seed=0,
                  settings=request['settings'], physical_stability_qualified=False,
                  versions={n: importlib.metadata.version(n) for n in ('ase', 'numpy', 'orb-models', 'torch')})
    save(directory / 'result.json', result)
    return result


if __name__ == '__main__':
    relax(Path(sys.argv[1]))
