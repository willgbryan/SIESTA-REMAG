"""Small, artifact-bound electronic answers for research agents and viewers."""
from pathlib import Path
import zipfile

import numpy as np

from ..corpus import sha256
from ..discovery.common import load


def load_spectrum(metadata_path):
    path = Path(metadata_path).resolve()
    meta = load(path)
    if meta.get('format') != 'REMAG-ELECTRONIC/1':
        raise ValueError('Expected an electronic spectrum metadata artifact')
    residual = meta.get('native_max_abs_error_ev')
    if (meta.get('native_comparison_passed') is not True or type(residual) not in (int, float)
            or not np.isfinite(residual) or not 0 <= residual <= 1e-3):
        raise ValueError('Native spectral comparison has not passed')
    artifact = (path.parent/meta['artifact']).resolve()
    if not artifact.is_relative_to(path.parent) or sha256(artifact) != meta['artifact_sha256']:
        raise ValueError('Electronic artifact binding differs')
    with zipfile.ZipFile(artifact) as archive:
        if sum(item.file_size for item in archive.infolist()) > 512*1024*1024:
            raise ValueError('Electronic inspection exceeds 512 MiB; select a smaller exported sampling')
    with np.load(artifact, allow_pickle=False) as a:
        required = {'eigenvalues_ev', 'k_fractional', 'k_weights', 'occupations', 'orbital_mulliken', 'spin_pauli'}
        if not required <= set(a.files):
            raise ValueError('Incomplete electronic arrays')
        data = {key: a[key] for key in required}
    e, k, w, o, p, s = (data[key] for key in ('eigenvalues_ev', 'k_fractional', 'k_weights',
                                           'occupations', 'orbital_mulliken', 'spin_pauli'))
    if (e.ndim != 2 or min(e.shape) < 1 or k.shape != (len(e),3) or w.shape != (len(e),)
            or o.shape != e.shape or p.shape != (len(e), len(meta['orbitals']), e.shape[1])
            or (s.size and s.shape != (*e.shape,3)) or any(not np.isfinite(v).all() for v in data.values())
            or np.any(w < 0) or not np.isclose(w.sum(), 1, rtol=0, atol=1e-10)):
        raise ValueError('Invalid electronic array dimensions, values or sampling')
    if not np.isfinite(meta['fermi_ev']) or not np.isfinite(meta['temperature_ev']):
        raise ValueError('Nonfinite electronic energy reference')
    return meta, data


def _indices(selection, field, count):
    value = selection.get(field, list(range(count)))
    if (not isinstance(value, list) or not value or any(type(i) is not int or not 0 <= i < count for i in value)
            or len(value) != len(set(value))):
        raise ValueError(f'{field} must be distinct in-range indices')
    return value


def _select(meta, arrays, selection):
    if not isinstance(selection, dict) or set(selection)-{'atom_indices', 'orbital_indices', 'l', 'k_indices', 'energy_window_ev', 'limit'}:
        raise ValueError('Unknown electronic selection fields')
    orbitals = meta['orbitals']
    selected = _indices(selection, 'orbital_indices', len(orbitals))
    for field, column in [('atom_indices','atom_index'), ('l','l')]:
        if field in selection:
            values = selection[field]
            available = {o[column] for o in orbitals}
            if (not isinstance(values, list) or not values or any(type(x) is not int or x not in available for x in values)
                    or len(values) != len(set(values))):
                raise ValueError(f'{field} must select existing orbital metadata')
            selected = [i for i in selected if orbitals[i][column] in values]
    if not selected:
        raise ValueError('Orbital selection is empty')
    ks = _indices(selection, 'k_indices', len(arrays['eigenvalues_ev']))
    window = selection.get('energy_window_ev', [-3,3])
    if (not isinstance(window, list) or len(window) != 2 or any(type(x) not in (int,float) or not np.isfinite(x) for x in window)
            or not -100 <= window[0] < window[1] <= 100):
        raise ValueError('Select an increasing energy window within [-100,100] eV relative to Fermi')
    limit = selection.get('limit', 40)
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError('State output limit must be 1..200')
    return selected, ks, window, limit


def inspect_electronic(metadata_path, selection=None):
    meta, data = load_spectrum(metadata_path)
    orbitals, ks, window, limit = _select(meta, data, {} if selection is None else selection)
    relative = data['eigenvalues_ev']-meta['fermi_ev']
    populations = data['orbital_mulliken'][:, orbitals, :].sum(axis=1)
    candidates = [(k,n) for k in ks for n in np.flatnonzero((relative[k]>=window[0]) & (relative[k]<=window[1]))]
    candidates.sort(key=lambda kn: (abs(relative[kn]), kn))
    states = []
    for k,n in candidates[:limit]:
        state = dict(k_index=k, band_index=int(n), k_fractional=data['k_fractional'][k].tolist(),
            k_weight=float(data['k_weights'][k]), energy_minus_fermi_ev=float(relative[k,n]),
            energy_ev=float(data['eigenvalues_ev'][k,n]), occupation=float(data['occupations'][k,n]),
            selected_mulliken_weight=float(populations[k,n]))
        if data['spin_pauli'].size:
            state['spin_pauli'] = data['spin_pauli'][k,n].tolist()
        states.append(state)
    return dict(format='REMAG-ELECTRONIC-SELECTION/1', metadata_sha256=sha256(metadata_path),
        hsx_sha256=meta['hsx_sha256'], artifact_sha256=meta['artifact_sha256'], selection={} if selection is None else selection,
        selected_orbitals=[meta['orbitals'][i] for i in orbitals], states=states,
        matching_states=len(candidates), omitted_states=max(0,len(candidates)-limit),
        selected_k_weight=float(data['k_weights'][ks].sum()), occupations_model=meta['occupations_model'],
        spin_frame=meta['spin_frame'], projection=meta['projection'],
        native_max_abs_error_ev=meta['native_max_abs_error_ev'], physical_accuracy_qualified=False)


def projected_dos(metadata_path, selection=None, *, sigma_ev=.12, points=161):
    meta, data = load_spectrum(metadata_path)
    orbitals, ks, window, _ = _select(meta, data, {} if selection is None else selection)
    if type(sigma_ev) not in (int,float) or not np.isfinite(sigma_ev) or not .001 <= sigma_ev <= 2:
        raise ValueError('Gaussian sigma must be .001..2 eV')
    if type(points) is not int or not 2 <= points <= 801:
        raise ValueError('DOS grid must have 2..801 points')
    grid = np.linspace(*window, points)
    energies = data['eigenvalues_ev'][ks]-meta['fermi_ev']
    pop = data['orbital_mulliken'][ks][:,orbitals,:].sum(axis=1)
    weight = data['k_weights'][ks,None]*pop*(2 if meta['spin_components']==1 else 1)
    # Work one energy sample at a time; never allocate grid×k×band×orbital.
    values = [float(np.sum(np.exp(-.5*((e-energies)/sigma_ev)**2)*weight)/(sigma_ev*np.sqrt(2*np.pi))) for e in grid]
    return dict(format='REMAG-PROJECTED-DOS/1', metadata_sha256=sha256(metadata_path),
        hsx_sha256=meta['hsx_sha256'], energy_minus_fermi_ev=grid.tolist(), density_states_per_ev=values,
        sigma_ev=sigma_ev, selection={} if selection is None else selection, selected_k_weight=float(data['k_weights'][ks].sum()),
        collinear_spin=meta.get('collinear_spin'), projection=meta['projection'],
        scope='Signed Mulliken DOS; selected k weights are not renormalized; single collinear channel unless spin-degenerate')


def compare_electronic(before_path, after_path, selection=None, *, sigma_ev=.12, points=161,before_native=None,after_native=None):
    a, da = load_spectrum(before_path)
    b, db = load_spectrum(after_path)
    for key in ('orbitals', 'spin_components', 'collinear_spin', 'spin_frame', 'projection', 'temperature_ev'):
        if a.get(key) != b.get(key):
            raise ValueError('Electronic comparison metadata differs: '+key)
    from .sampling import compare_sampling,native_report_context
    if (before_native is None)!=(after_native is None):raise ValueError('Supply both native endpoint contexts or neither')
    ca=native_report_context(before_native,a) if before_native is not None else None
    cb=native_report_context(after_native,b) if after_native is not None else None
    sampling=compare_sampling(da,db,ca,cb)
    left = projected_dos(before_path, selection, sigma_ev=sigma_ev, points=points)
    right = projected_dos(after_path, selection, sigma_ev=sigma_ev, points=points)
    return dict(format='REMAG-ELECTRONIC-COMPARISON/1',
        before_metadata_sha256=sha256(before_path), after_metadata_sha256=sha256(after_path),
        before_hsx_sha256=a['hsx_sha256'], after_hsx_sha256=b['hsx_sha256'],
        energy_minus_fermi_ev=left['energy_minus_fermi_ev'], before_dos=left['density_states_per_ev'],
        after_dos=right['density_states_per_ev'],
        delta_dos=(np.array(right['density_states_per_ev'])-left['density_states_per_ev']).tolist(),
        fermi_shift_ev=b['fermi_ev']-a['fermi_ev'], sigma_ev=sigma_ev, selection={} if selection is None else selection,
        sampling_correspondence=sampling,native_contexts=dict(before=ca['binding'],after=cb['binding']) if ca else None,
        scope='Each spectrum aligned to its own Fermi level; basis-dependent populations, no band correspondence or SOC attribution',
        mechanism_verified=False, physical_accuracy_qualified=False)
