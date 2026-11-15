"""Bound saved matrix/moment observables for structural-response research.

Couplings are symmetric-Lowdin block proxies. Band sums are partial occupied
single-particle sums. Neither is a partition of total energy or isolated SOC.
"""
import json
import re
from pathlib import Path

import numpy as np

from ..corpus import sha256
from .mechanisms import _load_matrices, decompose
from .orbital_moments import load_orbital_moments

KINDS = frozenset({'orbital_moment', 'coupling', 'band_sum'})


def _indices(value, limit, name):
    if (not isinstance(value, list) or not value or len(value) > limit
            or any(type(i) is not int or not 0 <= i < limit for i in value)
            or len(set(value)) != len(value)):
        raise ValueError('Distinct bounded ' + name + ' indices required')
    return sorted(value)


def validate_diagnostic(raw):
    if not isinstance(raw, dict) or raw.get('kind') not in KINDS:
        raise ValueError('Unknown mechanism-response diagnostic')
    kind = raw['kind']
    fields = {'orbital_moment': {'selection', 'component'},
              'coupling': {'spin', 'groups', 'metric', 'k_indices'},
              'band_sum': {'spin', 'reference', 'k_indices'}}[kind]
    if set(raw) != {'kind', 'stage'} | fields or not isinstance(raw['stage'], str) or not re.fullmatch(r'scalar|soc-[a-z][a-z0-9_-]{0,31}', raw['stage']):
        raise ValueError('Exact diagnostic fields and a native stage required')
    result = json.loads(json.dumps(raw, allow_nan=False))
    if kind == 'orbital_moment':
        if not raw['stage'].startswith('soc-'):
            raise ValueError('Native orbital moments require an SOC stage')
        if not ((type(raw['component']) is int and raw['component'] in (0, 1, 2)) or raw['component'] == 'norm'):
            raise ValueError('Select an orbital-moment Cartesian component or norm')
        selection = raw['selection']
        if not isinstance(selection, dict) or set(selection) - {'sites', 'l'}:
            raise ValueError('Select parent sites and/or orbital angular momenta')
        result['selection'] = {k: _indices(v, 256 if k == 'sites' else 4, k) for k, v in selection.items()}
        return result, 'muB'
    if type(raw['spin']) is not int or raw['spin'] not in ((0, 1) if raw['stage'] == 'scalar' else (0,)):
        raise ValueError('Select a scalar spin channel or the single SOC spinor channel')
    result['k_indices'] = _indices(raw['k_indices'], 100000, 'native k')
    if len(result['k_indices']) > 8:
        raise ValueError('Matrix responses use at most eight selected native k points')
    if kind == 'band_sum':
        if raw['reference'] not in ('absolute', 'own_fermi'):
            raise ValueError('Declare absolute or own_fermi band-energy reference')
        return result, 'eV/cell'
    groups = raw['groups']
    if (not isinstance(groups, list) or len(groups) != 2 or groups[0] == groups[1]
            or any(not isinstance(g, dict) or set(g) != {'site', 'l'} or type(g['site']) is not int
                   or not 0 <= g['site'] < 256 or type(g['l']) is not int or not 0 <= g['l'] <= 3 for g in groups)
            or raw['metric'] not in ('frobenius', 'largest_singular_value')):
        raise ValueError('Select two distinct parent-site/l groups and a coupling norm')
    return result, 'eV'


def _sampling(meta, arrays, report, report_path, stage):
    """Prove selected arrays against original KP and the complete declared mesh."""
    from .sampling import _bound_file, native_mesh_identity
    from .electronic import _native_weights
    context = dict(report=report, report_path=report_path, stage=stage,
                   metadata={'kp_sha256': meta['evidence']['kp']})
    path = _bound_file(context, 'remag.KP', 32*1024*1024)
    lines = path.read_text().splitlines(); rows = [line.split() for line in lines[1:]]
    if not rows or len(rows) > 100000 or any(len(row) != 5 for row in rows):
        raise ValueError('Invalid complete native KP inventory')
    values = np.asarray([[float(v) for v in row[1:4]] for row in rows])
    cell = np.asarray(arrays['cell_bohr']).reshape(3, 3)
    full = dict(k_fractional=values @ cell.T/(2*np.pi),
                k_weights=_native_weights([row[4] for row in rows])[0])
    identity, proof = native_mesh_identity(full, context)
    indices = _indices(meta['k_indices'], len(rows), 'saved native k')
    if indices != meta['k_indices'] or meta['native_k_count'] != len(rows):
        raise ValueError('Selected matrix native sampling inventory differs')
    for key in ('k_fractional', 'k_weights'):
        if not np.allclose(arrays[key], full[key][indices], rtol=0, atol=1e-12):
            raise ValueError('Selected matrix arrays differ from their original native KP indices')
    return identity, proof


def measure_diagnostic(report, report_path, metadata_path, definition, *, mapping=None):
    """Return one value, its evidence and an internal cross-point signature."""
    o, _ = validate_diagnostic(definition); kind = o['kind']; stage = o['stage']
    path = Path(metadata_path); meta, arrays = (load_orbital_moments(path) if kind == 'orbital_moment' else _load_matrices(path))
    if (meta.get('report_sha256') != sha256(report_path) or meta.get('source_sha256') != report['source_sha256']
            or meta.get('executable_sha256') != report['executable_sha256'] or meta.get('stage') != stage):
        raise ValueError('Diagnostic belongs to another source, report, executable or stage')
    suffixes = {'hsx': 'remag.HSX', 'dm': 'remag.DM', 'stdout': 'stdout.log'} if kind == 'orbital_moment' else {'hsx': 'remag.HSX', 'kp': 'remag.KP', 'eig': 'remag.EIG'}
    for key, name in suffixes.items():
        expected = report['artifacts'].get(stage+'/'+name)
        if not expected or meta.get('evidence', {}).get(key) != expected:
            raise ValueError('Diagnostic native artifact binding differs: ' + name)
    from ase.data import chemical_symbols
    species = [chemical_symbols[z] for z in report['structure']['geometry']['atomic_numbers']]
    mapping = list(range(len(species))) if mapping is None else mapping
    if len(mapping) != len(species) or sorted(mapping) != list(range(len(species))):
        raise ValueError('Complete parent-to-child atom correspondence required')
    inverse = {child: parent for parent, child in enumerate(mapping)}
    orbitals = meta['orbitals']; inventory = []
    for i, orbital in enumerate(orbitals):
        atom = orbital['atom_index']
        if type(atom) is not int or atom not in inverse or orbital['index'] != i or orbital['species'] != species[atom]:
            raise ValueError('Diagnostic orbital inventory differs from the authored atoms')
        inventory.append({k: v for k, v in orbital.items() if k != 'index'} | {'atom_index': inverse[atom]})
    signature = dict(kind=kind, orbitals=sorted(json.dumps(r, sort_keys=True) for r in inventory))
    evidence = dict(diagnostic_sha256=sha256(path), artifact_sha256=meta['artifact_sha256'], native_bindings=meta['evidence'])
    if kind == 'orbital_moment':
        selection = o['selection']; selected_sites = selection.get('sites', list(range(len(mapping))))
        if any(i >= len(mapping) for i in selected_sites):raise ValueError('Parent site outside orbital-moment source')
        if 'l' in selection and not set(selection['l']) <= {r['l'] for r in orbitals}:raise ValueError('Absent orbital angular momentum')
        indices = [i for i, r in enumerate(orbitals) if inverse[r['atom_index']] in selected_sites and ('l' not in selection or r['l'] in selection['l'])]
        if not indices:raise ValueError('Empty orbital-moment selection')
        vector = arrays['orbital_moments_mu_b'][indices].sum(axis=0)
        value = float(np.linalg.norm(vector) if o['component'] == 'norm' else vector[o['component']])
        signature['metadata'] = {k: meta[k] for k in ('spin_components', 'frame', 'unit', 'harmonic_order', 'density_convention', 'approximation')}
        evidence.update(vector_mu_b=vector.tolist(), selected_orbital_count=len(indices),
            scope=meta['approximation'], radial_identity_verified=False,
            sampling='Native integrated density; not a selected-k reconstruction')
        return value, evidence, signature
    if (meta['collinear_spin'] != (o['spin'] if stage == 'scalar' else None)
            or meta['spin_components'] not in ((1, 2) if stage == 'scalar' else (4, 8))):
        raise ValueError('Matrix spin channel differs from the response definition')
    if meta['k_indices'] != o['k_indices']:raise ValueError('Matrix native k selection differs from the declared response')
    for species_name in {r['species'] for r in orbitals}:
        expected=report['artifacts'].get(stage+'/'+species_name+'.ion.xml')
        if not expected or meta.get('ion_sha256', {}).get(species_name) != expected:
            raise ValueError('Radial basis artifact is not bound to the native report')
    signature['metadata'] = {k: meta[k] for k in ('spin_components', 'collinear_spin', 'spin_frame', 'lattice_gauge', 'temperature_ev', 'occupation_model', 'radial_layout', 'k_indices', 'native_k_count')}
    if arrays['radial_functions'].ndim != 1 or not arrays['radial_functions'].size:
        raise ValueError('Actual radial function arrays required')
    signature['radial_functions'] = arrays['radial_functions']
    signature['sampling'], evidence['sampling'] = _sampling(meta, arrays, report, report_path, stage)
    signature['length_conventions'] = {r['length_convention'] for r in evidence['sampling']['admitted_length_conventions']}
    groups = None
    if kind == 'coupling':
        groups = []
        for g in o['groups']:
            if g['site'] >= len(mapping):raise ValueError('Coupling parent site outside this source')
            indices = [i for i, r in enumerate(orbitals) if r['atom_index'] == mapping[g['site']] and r['l'] == g['l']]
            if not indices:raise ValueError('Selected coupling group is absent')
            groups.append([j for i in indices for j in ([2*i, 2*i+1] if meta['spin_components'] > 2 else [i])])
    values = []
    # Recompute from saved H/S; do not trust duplicated JSON coupling/band summaries.
    for h, s, eigenvalues in zip(arrays['hamiltonian_ev'], arrays['overlap'], arrays['eigenvalues_ev']):
        d = decompose(h, s, meta['fermi_ev'], meta['temperature_ev'], degeneracy=2 if meta['spin_components'] == 1 else 1)
        if not np.allclose(d['eigenvalues_ev'], eigenvalues, rtol=0, atol=1e-8):raise ValueError('Saved matrix eigenvalues differ from H/S')
        if kind == 'coupling':
            block = d['lowdin_hamiltonian_ev'][np.ix_(*groups)]
            value = np.linalg.norm(block) if o['metric'] == 'frobenius' else np.linalg.svd(block, compute_uv=False)[0]
        else:
            relative = d['eigenvalues_ev'] - (meta['fermi_ev'] if o['reference'] == 'own_fermi' else 0)
            value = d['occupations'] @ relative
        values.append(float(value))
    weights = arrays['k_weights']; value = float(weights @ values)
    evidence.update(per_k=[dict(native_k_index=i, value=v, weight=float(w)) for i, v, w in zip(o['k_indices'], values, weights)],
        selected_k_weight=float(weights.sum()), weight_renormalized=False,
        scope=('Partial native-weighted symmetric Lowdin block norm; basis-dependent hybridization proxy, not isolated SOC or unique bond energy'
            if kind == 'coupling' else 'Partial occupied self-consistent band sum; explicit energy reference, no force-theorem or total E/F interpretation'))
    return value, evidence, signature


def compare_diagnostics(first, second):
    for key in ('kind', 'metadata', 'orbitals'):
        if first[key] != second[key]:raise ValueError('Response diagnostic convention or parent orbital inventory differs: ' + key)
    if first['kind'] == 'orbital_moment':
        return dict(parent_orbital_inventory_equal=True, radial_identity_verified=False,
            scope='Common onsite convention and orbital inventory; actual radial arrays are not included in moment exports')
    a, b = first['radial_functions'], second['radial_functions']
    if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-10):raise ValueError('Response diagnostic radial functions differ')
    if first['sampling'] != second['sampling'] or not first['length_conventions'] & second['length_conventions']:
        raise ValueError('Response diagnostic native sampling differs')
    return dict(parent_orbital_inventory_equal=True, radial_max_abs_difference=float(np.max(abs(a-b))),
        radial_tolerance=1e-10, native_mesh_identity_equal=True,
        scope='Corresponding atom/l blocks and native samples; geometry/overlap may change, no cross-geometry eigenvector or band matching')
