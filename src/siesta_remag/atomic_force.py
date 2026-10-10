"""Audit retained-input core/neutral force shadows against CPU contributions."""
import math


def shadow_errors(rows, grid_rows, atoms):
    errors = []
    expected = [(r['epoch'], kind) for r in grid_rows
                for kind in ((1, 2) if r.get('final_partial_core') == 1 else (2,))]
    if (not grid_rows or any(r.get('final_partial_core') not in (0, 1) for r in grid_rows) or
            [(r.get('epoch'), r.get('kind')) for r in rows] != expected):
        errors.append('Atomic-force shadows do not cover required consumers in geometry order')
    by_epoch = {r['epoch']: r for r in grid_rows}
    for record in rows:
        grid = by_epoch.get(record.get('epoch'))
        kind = record.get('kind')
        if (kind not in (1, 2) or record.get('atoms') != atoms or record.get('passed') is not True or
                record.get('status') != 0 or record.get('graph_nodes') != 1 or
                record.get('electronic_inputs_resident') is not True):
            errors.append('Invalid native atomic-force shadow')
        mesh, points, nsm, entries = (record.get(k) for k in ('mesh', 'points', 'nsm', 'entries'))
        if (not isinstance(mesh, list) or len(mesh) != 3 or any(type(v) is not int or v <= 0 for v in mesh) or
                type(nsm) is not int or nsm < 1 or type(points) is not int or points <= 0 or
                type(entries) is not int or entries < 0):
            errors.append('Invalid atomic-force geometry shape')
        else:
            if math.prod(mesh) != points or any(m % nsm for m in mesh):
                errors.append('Atomic-force mesh ordering metadata is inconsistent')
            if grid and (points != grid['grid_shape'][1]*grid['grid_shape'][2] or nsm**3 != grid['grid_shape'][2] or
                    record.get('partial_core') != grid.get('final_partial_core') or
                    record.get('retained_electronic_bytes') != grid['retained_electronic_bytes'] or
                    record.get('diagonal_spins') != (min(2, grid['spins']) if kind == 1 else 1)):
                errors.append('Atomic-force fields differ from final electronic geometry/configuration')
            if (record.get('setup_bytes') != 56*entries+32+8*(3*atoms+9)+4 or
                    record.get('export_bytes') != 8*(3*atoms+9)+4):
                errors.append('Shape-derived atomic-force transfer sizes disagree')
        for name, count, tolerance, unit in (('force', 3*atoms, 1e-10, 'ry_per_bohr'), ('stress', 9, 1e-12, 'ry_per_bohr3')):
            a, b = record.get('reference_'+name, []), record.get('device_'+name, [])
            if len(a) != count or len(b) != count or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in a+b):
                errors.append('Missing/nonfinite atomic '+name+' arrays')
                continue
            error = max(abs(x-y) for x, y in zip(a, b))
            if error >= tolerance or error != record.get('max_'+name+'_error_'+unit):
                errors.append('Atomic '+name+' comparison failed or cannot be reproduced')
    return errors
