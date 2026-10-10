"""Validate authoritative dfscf publication; scientific agreement is external."""
import math


def record_errors(record, atoms):
    errors = []
    if (record.get('atoms') != atoms or record.get('authoritative') is not True or
            record.get('comparison_performed') is not False or record.get('passed') is not True or
            record.get('status') != 0 or record.get('graph_nodes') != (3 if record.get('orbital_basis_shadow') else 2 if record.get('orbital_basis_device') else 1) or
            record.get('electronic_inputs_resident') is not True):
        errors.append('Invalid authoritative GPU grid-force result')
    for name, count, unit in (('force', 3*atoms, 'ry_per_bohr'), ('stress', 9, 'ry_per_bohr3')):
        for prefix in ('initial_', 'device_'):
            values = record.get(prefix+name, [])
            if len(values) != count or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
                errors.append('Missing/nonfinite '+prefix+name)
        key = 'max_'+name+'_error_'+unit
        if key not in record or record[key] is not None:
            errors.append('Production force record claims an internal reference comparison')
    shape = record.get('grid_shape', [])
    nh, entries = record.get('spins'), record.get('entries')
    if (len(shape) != 6 or any(type(v) is not int or v <= 0 for v in shape) or
            nh not in (1, 2, 4, 8) or type(entries) is not int or entries <= 0):
        errors.append('Invalid force geometry shape')
    else:
        no, np, nsp, nuo, nnz, cells = shape
        ng = min(nh, 4)
        derivative_bytes=0 if record.get('orbital_basis_device') else 56*entries*nsp
        if record.get('orbital_basis_device') and record.get('orbital_basis_shadow'):errors.append('Conflicting derivative modes')
        if record.get('derivative_upload_bytes')!=derivative_bytes or record.get('host_derivative_storage_bytes')!=derivative_bytes:
            errors.append('Derivative upload/storage differs from selected mode')
        upload = 4*no + derivative_bytes + 8*nuo + 24 + 8*(3*atoms+9) + 4
        if record.get('atomic_terms_authoritative') is True:
            if record.get('atomic_basis_shadow') and record.get('atomic_basis_device'):errors.append('Conflicting atomic derivative modes')
            terms = record.get('atomic_terms', [])
            expected = [1, 2] if record.get('final_partial_core') == 1 else [2]
            if (record.get('shared_accumulator') is not True or record.get('atomic_intermediate_export_bytes') != 0 or
                    record.get('final_partial_core') not in (0, 1) or [r.get('kind') for r in terms] != expected):
                errors.append('Shared atomic-force ownership/consumer coverage failed')
            for term in terms:
                count = term.get('entries')
                if (type(count) is not int or count < 0 or term.get('points') != np*nsp or
                        term.get('diagonal_spins') != (min(2, nh) if term.get('kind') == 1 else 1) or
                        term.get('graph_nodes') != (3 if record.get('atomic_basis_shadow') else 2 if record.get('atomic_basis_device') else 1) or term.get('setup_bytes') != (0 if record.get('atomic_basis_device') else 56*count)+32 or
                        term.get('derivative_upload_bytes') != (0 if record.get('atomic_basis_device') else 56*count) or
                        term.get('host_derivative_storage_bytes') != term.get('derivative_upload_bytes')):
                    errors.append('Invalid shared atomic-force shape/staging record')
                else:
                    upload += term['setup_bytes']
        elif record.get('shared_accumulator') or record.get('atomic_terms'):
            errors.append('Unexpected shared atomic-force contributions')
        retained = 8*(nnz*(nh+ng) + np*nsp*(ng+2))
        if (record.get('topology_validation') != 'retained_geometry_owner' or record.get('retained_entries') != entries or
                record.get('topology_export_bytes') != 0):
            errors.append('Force topology was not validated by its retained owner')
        if (record.get('setup_bytes') != upload or record.get('export_bytes') != 8*(3*atoms+9)+4 or
                record.get('retained_electronic_bytes') != retained):
            errors.append('Shape-derived force transfer/retention sizes disagree')
    return errors
