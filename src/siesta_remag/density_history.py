"""Production density history lifetime and explicit boundary accounting.

These checks complement raw upstream population/force/relaxation comparisons;
consistent journals alone do not prove numerical equivalence or full residency.
"""
import re


def settings(text):
    """Read standalone regression FDFs, with first-occurrence label semantics."""
    values = {}
    block = False
    for line in text.splitlines():
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        if fields[0].lower() == '%include' or '<' in fields:
            raise ValueError('History audit requires standalone FDF settings')
        if fields[0].lower() == '%block':
            block = True
        elif fields[0].lower() == '%endblock':
            block = False
        elif not block:
            key = re.sub(r'[._-]', '', fields[0]).lower()
            values.setdefault(key, fields[1:])
    return values


def boolean(values, key, default):
    if key not in values:
        return default
    parts = values[key]
    if len(parts) != 1 or parts[0].lower() not in ('t', 'true', '.true.', 'f', 'false', '.false.'):
        raise ValueError('Invalid history/output boolean: '+key)
    return parts[0].lower() in ('t', 'true', '.true.')


def policy(text):
    values = settings(text)
    # Full native policy still comes from upstream get_allowed_history_depth.
    # This audit currently qualifies the fixed-geometry and CG corpus only.
    if values.get('mdtypeofrun', ['cg'])[0].lower() != 'cg':
        raise ValueError('History material audit has no reference for this MD mode')
    depth = int(values.get('dmhistorydepth', ['1'])[0])
    if not boolean(values, 'dmallowreuse', True):
        depth = 0
    elif not boolean(values, 'dmallowextrapolation', False):
        depth = 1
    if not 0 <= depth <= 32:
        raise ValueError('History depth exceeds the qualified device owner')
    mode = (3 if boolean(values, 'usesvdexperimental', False) else
            2 if boolean(values, 'usesvd', True) else 1) if boolean(values, 'usediisfordmextrapolation', True) else 0
    return dict(depth=depth, mode=mode)


def record_errors(records, cycles, initial, forces, expected):
    errors = []
    if not records or not len(records) == len(cycles) == len(initial) == len(forces):
        return ['History ownership does not cover every electronic/force geometry']
    count = 0
    previous_epoch = 0
    try:
        for token, (record, cycle, owner, force) in enumerate(zip(records, cycles, initial, forces), 1):
            depth, mode = expected['depth'], expected['mode']
            reuse, save = count > 0, depth > 0
            rows, atoms = owner['rows'], owner['atoms']
            capacity, spins = cycle['terminal']['capacity'], cycle['terminal']['spins']
            epoch = cycle['epoch']
            wanted = dict(token=token, epoch=epoch, atoms=atoms, rows=rows, capacity=capacity, spins=spins,
                depth=depth, mode=mode, previous_count=count, count=min(depth, count+1) if save else count,
                reused=reuse, bound=True, finished=True, saved=save, forces_accepted=True, status=0,
                setup_bytes=48*(1+int(token == 1))+8*rows+4*capacity+24*atoms,
                setup_copies=5+int(token == 1), initial_status_copies=1+int(token == 1)+int(reuse),
                initial_status_bytes=40*(1+int(token == 1)+int(reuse)),
                post_status_bytes=40*int(save), post_status_copies=int(save),
                coefficient_export_bytes=8*count if reuse else 0)
            if any(type(record.get(k)) is not type(v) or record[k] != v for k, v in wanted.items()):
                errors.append('History policy, lifetime, topology or boundary accounting differs')
            if (type(epoch) is not int or epoch <= previous_epoch or owner['epoch'] != epoch or force['epoch'] != epoch or
                    cycle.get('converged') is not True or owner.get('density_from_history') is not reuse or
                    owner.get('density_upload_bytes') != (0 if reuse else 8*capacity*spins)):
                errors.append('History does not link accepted device state to the next initial density')
            # The default constrained SVD factors the augmented (count+1)
            # system, including its sum-to-one constraint. Its numerical rank
            # is not the number of retained density matrices. The other modes
            # solve count or count-1 dimensions; singleton reuse is exact.
            rank_limit = (count if count < 2 else
                          count+1 if mode == 2 else count if mode == 1 else count-1)
            rank, fallback = record['rank'], record['fallback']
            invalid_rank = type(rank) is not int or not 0 <= rank <= rank_limit
            if count < 2:
                invalid_rank = invalid_rank or rank != count
            wanted_fallback = int(count > 1 and (rank == 0 if mode == 0 else rank < count)) if mode < 2 else 0
            if (type(record['workspace_bytes']) is not int or record['workspace_bytes'] <= 0 or
                    invalid_rank or type(fallback) is not int or fallback != wanted_fallback):
                errors.append('Invalid history workspace or coefficient diagnostic')
            count = wanted['count']
            previous_epoch = epoch
    except (KeyError, TypeError, ValueError):
        errors.append('Incomplete production history evidence')
    return errors
