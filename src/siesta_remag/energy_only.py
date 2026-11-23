"""Validate explicit electronic finalization without certifying skipped forces."""
import math


def record_errors(record, cycles, output, text):
    errors = []
    if len(cycles) != 1 or record.get('schema') != 1 or record.get('mode') != 'energy_only':
        return ['Energy-only publication requires exactly one electronic geometry']
    cycle = cycles[0]
    expected = dict(epoch=cycle.get('epoch'), status=0, iterations=cycle.get('iterations'),
                    capacity=cycle.get('terminal', {}).get('capacity'),
                    spins=cycle.get('terminal', {}).get('spins'),
                    forces_calculated=False, stress_calculated=False,
                    hamiltonian_required=False, hamiltonian_cycle_export_bytes=0)
    if any(type(record.get(k)) is not type(v) or record[k] != v for k, v in expected.items()):
        errors.append('Energy-only publication differs from the accepted electronic cycle')
    if cycle.get('status') != 0 or cycle.get('converged') is not True:
        errors.append('Energy-only publication lacks converged device state')
    if 'REMAG: energy-only finalization; forces and stress not calculated' not in text:
        errors.append('Energy-only execution marker missing')
    if output.get('forces_ev_ang') is not None or output.get('stress_ev_ang3') is not None:
        errors.append('Skipped forces or stress were reported as computed')
    for key in ('total_energy_ev', 'free_energy_ev', 'fermi_energy_ev'):
        value = output.get(key)
        if type(value) not in (float, int) or not math.isfinite(value):
            errors.append('Energy-only result lacks finite '+key)
    return errors
