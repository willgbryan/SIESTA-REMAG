"""Validate final population ownership and shape-derived explicit boundaries."""


def record_errors(records, cycles, atoms, eigenvalues, species):
    errors = []
    if not records or len(records) != len(cycles):
        return ['Population ownership does not cover every electronic geometry']
    try:
        spins = eigenvalues['nspin']
        rows = eigenvalues['nbands']//(2 if spins >= 4 else 1)
        components = min(spins, 4)
        for record, cycle in zip(records, cycles):
            capacity = cycle['terminal']['capacity']
            setup = 36+8*atoms+8*species
            retained = 8*(rows+capacity)
            workspace = setup+retained+8*(components*(rows+atoms+1)+5*rows+6*atoms+5*species+10)+4
            expected = dict(epoch=cycle['epoch'], atoms=atoms, rows=rows, capacity=capacity, spins=spins, species=species,
                            setup_bytes=setup, retained_topology_bytes=retained, workspace_bytes=workspace,
                            evaluated=True, published=True)
            if any(record.get(k) != v for k, v in expected.items()):
                errors.append('Population shape/lifetime/storage differs from its electronic cycle')
            reads, exports = record['reads'], record['export_bytes']
            if len(reads) != 7 or len(exports) != 7 or any(type(v) is not int or v < 0 for v in reads):
                errors.append('Invalid population export/read record')
            elif exports != [8*n*bool(count) for n, count in zip((components*rows, components*atoms, components, 5*rows, 6*atoms, 5*species, 10), reads)]:
                errors.append('Population exports differ from requested shape or repeat-cache policy')
    except (KeyError, TypeError, ValueError):
        errors.append('Incomplete population ownership evidence')
    return errors
