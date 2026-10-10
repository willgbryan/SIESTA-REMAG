"""Audit the device spectrum retained across geometry steps until final output."""


def record_errors(records, cycles, forces):
    errors = []
    if not records or len(records) != len(cycles) or len(records) != len(forces):
        return ['Final spectrum does not cover all electronic and force geometries']
    previous = 0
    for index, (record, cycle, force) in enumerate(zip(records, cycles, forces)):
        epoch = record.get('epoch')
        terminal = cycle.get('terminal', {})
        bank = terminal.get('bank_size')
        if type(epoch) is not int or epoch <= previous or epoch != cycle.get('epoch') or epoch != force.get('epoch'):
            errors.append('Final spectrum geometry epoch differs')
        if type(epoch) is int:
            previous = epoch
        if type(bank) is not int or bank < 1 or record.get('bank_size') != bank:
            errors.append('Final spectrum bank dimensions differ')
            continue
        if terminal.get('deferred_spectrum') is not True or terminal.get('spectrum_bytes') != [0, 0]:
            errors.append('Deferred spectrum was downloaded at cycle publication')
        if cycle.get('converged') is not True or force.get('status') != 0:
            errors.append('Final spectrum lacks accepted electronic state and forces')
        last = index == len(records)-1
        if (record.get('retained_device_bytes') != 8*bank or record.get('occupation_export_bytes') != 0 or
                record.get('eigenvalue_export_bytes') != (8*bank if last else 0) or
                record.get('consumed') is not last or type(record.get('discarded')) is not bool or
                (not last and record.get('discarded') is not True)):
            errors.append('Final spectrum lifetime or requested output differs')
    return errors
