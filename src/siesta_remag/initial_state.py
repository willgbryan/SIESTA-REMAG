"""Initial device density ownership and explicitly requested population exports."""


def record_errors(records, cycles, atoms, eigenvalues, species, histories=None):
    errors = []
    if not records or len(records) != len(cycles):
        return ['Initial density ownership does not cover every electronic geometry']
    if histories is not None and len(histories) != len(records):
        return ['Initial density ownership has incomplete history provenance']
    try:
        spins = eigenvalues['nspin']
        rows = eigenvalues['nbands']//(2 if spins >= 4 else 1)
        components = min(spins, 4)
        for token, (record, cycle) in enumerate(zip(records, cycles), 1):
            from_history = histories[token-1].get('reused') if histories is not None else False
            if type(from_history) is not bool or record.get('density_from_history', False) is not from_history:
                errors.append('Initial density source differs from qualified history ownership')
            capacity = cycle['terminal']['capacity']
            requests = record['requests']
            if type(requests) is not int or not 0 <= requests <= 3:
                errors.append('Invalid initial population request'); continue
            expected = dict(token=token, epoch=cycle['epoch'], atoms=atoms, rows=rows, capacity=capacity,
                spins=spins, species=species, density_upload_bytes=0 if from_history else 8*capacity*spins,
                population_setup_bytes=(36+8*rows+8*atoms+8*species+8*capacity) if requests else 0,
                status_export_bytes=4 if requests else 0, status=0 if requests else -1,
                evaluated=bool(requests), closed=True, borrowed=True, released=True)
            if any(record.get(k) != v for k, v in expected.items()):
                errors.append('Initial density shape, lifetime or staging differs')
            reads, exports = record['reads'], record['export_bytes']
            if (len(reads) != 7 or len(exports) != 7 or
                    any(type(v) is not int or v < 0 for v in reads)):
                errors.append('Invalid initial population export record'); continue
            sizes = (components*rows, components*atoms, components, 5*rows, 6*atoms, 5*species, 10)
            if exports != [8*n*bool(count) for n, count in zip(sizes, reads)]:
                errors.append('Initial population exports differ from shapes or cache policy')
            # The supported Mulliken format reads raw orbital/atomic/total
            # populations for collinear spins. NC/SOC instead reads the
            # derived orbital, atomic and species views; it never requests
            # the raw total. Both routes print partial charges and a summary.
            mulliken_kinds = (0, 1, 2, 4, 6) if spins < 4 else (3, 4, 5, 6)
            if ((not requests and any(reads)) or (requests == 1 and any(reads[:6])) or
                    (requests & 1 and not reads[6]) or
                    (requests & 2 and not all(reads[k] for k in mulliken_kinds))):
                errors.append('Initial population outputs differ from requests')
    except (KeyError, TypeError, ValueError):
        errors.append('Incomplete initial density ownership evidence')
    return errors
