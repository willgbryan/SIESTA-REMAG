"""Validate explicit electronic finalization without certifying skipped forces."""
import math


def publication_records(report):
    """Read legacy single-cycle receipts or the complete ordered sweep receipt."""
    if not isinstance(report, dict) or type(report.get('schema')) is not int or report.get('mode') != 'energy_only':
        return []
    if report['schema'] == 1:
        return [report]
    if report['schema'] == 2:
        records = report.get('cycles')
        if isinstance(records, list) and records and all(isinstance(r, dict) for r in records):
            return records
    return []


def record_errors(record, cycles, output, text):
    errors = []
    records = publication_records(record)
    if len(cycles) != 1 or len(records) != 1:
        return ['Energy-only publication requires exactly one electronic geometry']
    record = records[0]
    if type(record.get('schema')) is not int or record['schema'] != 1 or record.get('mode') != 'energy_only':
        return ['Invalid energy-only publication record']
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


def sweep_record_errors(report, cycles, outputs, texts, *, expected_directions):
    """Validate every direction; callers must supply the ordered per-axis views.

    expected_directions must come from the requested input, not completed rows.
    This does not accept a last-direction result as evidence for a whole sweep.
    Geometry identity, requested axes and final moment checks belong to the
    orientation collector, alongside the ordinary scientific comparison.
    """
    records = publication_records(report)
    if (type(expected_directions) is not int or expected_directions < 1 or
            not (len(records) == len(cycles) == len(outputs) == len(texts) == expected_directions)):
        return ['Energy-only publication does not cover every sweep direction']
    epochs = [c.get('epoch') for c in cycles]
    if any(type(e) is not int or e < 1 for e in epochs) or any(b <= a for a, b in zip(epochs, epochs[1:])):
        return ['Energy-only sweep epochs are missing, duplicated or out of order']
    errors = []
    for i, (record, cycle, output, text) in enumerate(zip(records, cycles, outputs, texts), 1):
        if record.get('schema') != 1 or type(record.get('schema')) is not int:
            errors.append(f'Direction {i}: Invalid energy-only publication record')
            continue
        errors.extend(f'Direction {i}: {error}' for error in record_errors(record, [cycle], output, text))
    return errors
