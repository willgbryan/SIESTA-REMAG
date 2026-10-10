"""Validate explicit terminal matrix roles and shape-derived cycle exports.

H(Dout), H(Dgen), D, Dold, E, Eold are the six recorded matrix roles.
This audits successful explicit copies; it does not observe library transfers.
"""
import re
import math
from .density_history import settings, boolean


def requested_hamiltonian_exports(input_text, gated):
    """Resolve the fixed/CG regression policy; do not certify extra analysis."""
    if not gated:
        return 1
    values=settings(input_text)
    if values.get('mdtypeofrun',['cg'])[0].lower()!='cg':
        raise ValueError('Hamiltonian export audit has no reference for this MD mode')
    if boolean(values,'writeh',False):
        raise ValueError('Hamiltonian audit requires per-iteration H writes disabled')
    blocks={re.sub(r'[._-]','',m.group(1)).lower() for m in
            re.finditer(r'^\s*%block\s+(\S+)',input_text,re.I|re.M)}
    if blocks-{'chemicalspecieslabel','latticevectors','atomiccoordinatesandatomicspecies','kgridmonkhorstpack','dminitspin'}:
        raise ValueError('Hamiltonian audit has no qualified additional analysis block')
    analysis=('coopwrite','borncharge','opticalcalculation','tdedwfinitialize',
              'siesta2wannier90writemmn','siesta2wannier90writeunk','siesta2wannier90writeamn','siesta2wannier90writeeig')
    if any(boolean(values,key,False) for key in analysis):
        raise ValueError('Hamiltonian audit has no qualified additional analysis consumer')
    return (2*boolean(values,'writehendofcycle',False)+
            4*(boolean(values,'savehs',True) or boolean(values,'writehshistory',False))+
            8*(boolean(values,'tshssave',False) or boolean(values,'writetshshistory',False))+
            64*boolean(values,'scfrecomputehafterscf',False))


def requested_density_exports(input_text, gpu_history, shadow=False):
    """Resolve D consumers for the supported fixed/CG material corpus."""
    values = settings(input_text)
    if values.get('mdtypeofrun', ['cg'])[0].lower() != 'cg':
        raise ValueError('Density export audit has no reference for this MD mode')
    if boolean(values, 'writedm', True):
        raise ValueError('Terminal audit requires per-iteration DM writes disabled')
    dm = boolean(values, 'writedmendofcycle', False)
    soc = values.get('spin', ['none'])[0].lower().replace('-', '') == 'spinorbit'
    return (int(not gpu_history) + 2*dm + 4*boolean(values, 'tsdesave', False) +
            8*(dm and boolean(values, 'cdfsave', False)) +
            16*(boolean(values, 'writedenchar', False) or (soc and boolean(values, 'writeorbmom', False))) +
            32*bool(shadow))


def requested_energy_exports(input_text, compact):
    """Read the relevant booleans from a generated, standalone corpus FDF.

    Reject redirects/includes rather than pretending to implement all of FDF.
    Like FDF, labels ignore case, dots, underscores and hyphens, and the first
    occurrence wins. Per-iteration output is excluded by the device-SCF guard.
    """
    values = {}
    for line in input_text.splitlines():
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        if fields[0].lower() == '%include' or '<' in fields:
            raise ValueError('Export audit requires standalone FDF settings')
        key = re.sub(r'[._-]', '', fields[0]).lower()
        if key in ('tsdesave', 'cdfsave', 'writedmendofcycle', 'writedm') and key not in values:
            if len(fields) != 2 or fields[1].lower() not in ('true', 'false', 't', 'f', '.true.', '.false.'):
                raise ValueError('Invalid export boolean: '+fields[0])
            values[key] = fields[1].lower() in ('true', 't', '.true.')
    if values.get('writedm', True):
        raise ValueError('Terminal audit requires per-iteration DM writes disabled')
    return (int(not compact) + 2*values.get('tsdesave', False) +
            4*(values.get('cdfsave', False) and values.get('writedmendofcycle', False)))


def record_errors(cycles, final_grids, compact, energy_export_reasons, device_initialization=None, density_export_reasons=None, deferred_spectrum=None, hamiltonian_export_reasons=None):
    errors = []
    if (type(energy_export_reasons) is not int or not 0 <= energy_export_reasons <= 7 or
            bool(energy_export_reasons & 1) != (not compact)):
        return ['Invalid expected energy-density export policy']
    if not cycles or len(cycles) != len(final_grids):
        return ['Terminal export/final-grid geometry coverage differs']
    if density_export_reasons is not None and (type(density_export_reasons) is not int or not 0 <= density_export_reasons <= 63):
        return ['Invalid expected density export policy']
    if hamiltonian_export_reasons is not None and (type(hamiltonian_export_reasons) is not int or
            not 0 <= hamiltonian_export_reasons <= 127 or (hamiltonian_export_reasons&1 and hamiltonian_export_reasons!=1)):
        return ['Invalid expected Hamiltonian export policy']
    previous = 0
    for cycle, grid in zip(cycles, final_grids):
        terminal = cycle.get('terminal', {})
        epoch = cycle.get('epoch')
        if type(epoch) is not int or epoch <= previous or epoch != grid.get('epoch'):
            errors.append('Terminal export geometry epoch differs')
        if type(epoch) is int:
            previous = epoch
        capacity, spins, energy_spins, bank = [terminal.get(k) for k in ('capacity', 'spins', 'energy_spins', 'bank_size')]
        if (type(capacity) is not int or capacity < 1 or spins not in (1, 2, 4, 8) or
                energy_spins != min(spins, 4) or type(bank) is not int or bank < 1):
            errors.append('Invalid terminal export dimensions')
            continue
        if capacity != grid.get('capacity') or spins != grid.get('spins'):
            errors.append('Terminal matrix dimensions differ from final-grid state')
        old_energy = terminal.get('old_energy_host')
        if terminal.get('compact') is not compact or type(old_energy) is not bool:
            errors.append('Terminal export policy differs')
        if terminal.get('energy_export_reasons') != energy_export_reasons:
            errors.append('Energy-density export reasons differ from requested consumers')
        if density_export_reasons is not None and terminal.get('density_export_reasons') != density_export_reasons:
            errors.append('Density export reasons differ from requested consumers')
        h, e = 8*capacity*spins, 8*capacity*energy_spins
        wanted = [0, h, h, 0, e if energy_export_reasons else 0, 0] if compact else [h, h, h, h, e, e if old_energy else 0]
        if density_export_reasons is not None:
            wanted[2] = h if density_export_reasons else 0
        if hamiltonian_export_reasons is not None:
            if (terminal.get('hamiltonian_export_reasons',1)!=hamiltonian_export_reasons or
                    terminal.get('gated_hamiltonian',False) is not (hamiltonian_export_reasons!=1)):
                errors.append('Hamiltonian export reasons differ from requested consumers')
            if compact:wanted[1]=h if hamiltonian_export_reasons else 0
        if terminal.get('matrix_bytes') != wanted:
            errors.append('Terminal matrix roles/copy volumes differ')
        final_h = grid.get('hamiltonian_cycle_export_bytes')
        need_h = grid.get('hamiltonian_required') is True or grid.get('comparison_performed') is True
        if final_h != (h if need_h else 0):
            errors.append('Requested final-force Hamiltonian export differs')
        iterations = cycle.get('iterations')
        if type(iterations) is not int or iterations < 1:
            errors.append('Invalid terminal iteration count')
            continue
        # ABI: ScfCycleResult=160 bytes, ScfCycleStep=560 bytes. Spectrum and
        # occupation banks are two FP64 vectors; optional final H is separate.
        deferred = terminal.get('deferred_spectrum', False)
        spectrum = terminal.get('spectrum_bytes', [8*bank, 8*bank])
        if type(deferred) is not bool or (deferred_spectrum is not None and deferred is not deferred_spectrum):
            errors.append('Deferred spectrum mode differs from request')
        wanted_spectrum = [0, 0] if deferred else [8*bank, 8*bank]
        if spectrum != wanted_spectrum:
            errors.append('Cycle spectrum exports differ from requested ownership')
        expected = 160+560*iterations+sum(wanted_spectrum)+sum(wanted)+(h if need_h else 0)
        initial = cycle.get('initialization', {})
        enabled = initial.get('device', False)
        if device_initialization is not None and enabled is not device_initialization:
            errors.append('Initial electronic owner differs from requested mode')
        if initial:
            if type(enabled) is not bool or initial.get('result_bytes') != (32 if enabled else 0):
                errors.append('Initial density result export shape differs')
            if enabled:
                if (initial.get('status') != 0 or initial.get('mix_first') not in (0, 1) or
                        initial.get('note') not in (0, 1) or
                        any(type(initial.get(k)) not in (int, float) or not math.isfinite(initial[k]) or initial[k] == 0
                            for k in ('charge', 'scale'))):
                    errors.append('Invalid device initial density result')
                expected += 32
        profile = cycle.get('stage_profile')
        if profile is not None:
            names = {'assembly_and_eigensolver', 'occupations', 'density_and_normalization',
                     'hamiltonian', 'density_grid_hartree_xc', 'scalar_reductions', 'convergence_and_mixing'}
            stages = profile.get('stages', {}) if isinstance(profile, dict) else {}
            if (not isinstance(profile, dict) or profile.get('clock') != 'PTX globaltimer' or
                    not isinstance(stages, dict) or set(stages) != names or any(not isinstance(v, dict) or
                        type(v.get('calls')) is not int or v['calls'] < 1 or
                        type(v.get('seconds')) not in (int, float) or
                        not math.isfinite(v['seconds']) or v['seconds'] < 0 for v in stages.values())):
                errors.append('Invalid optional SCF stage profile')
            else:
                # ScfStageProfile: one timestamp and seven duration/count pairs,
                # all uint64; exported once after the complete SCF cycle.
                expected += 8*(1+2*len(names))
        if cycle.get('export_bytes') != expected:
            errors.append('Terminal export total does not match matrix/diagnostic/spectrum shapes')
    return errors
