"""Selected-k matrix diagnostics; never a partition of self-consistent MAE.

All eigenvectors use the lattice gauge of electronic.bloch_matrices. Density
matrices are reconstructed from those states and HSX Fermi-Dirac occupations,
not read from the native DM. Symmetric orthogonalization makes the chosen
orbital coupling diagnostic explicit without pretending it is basis invariant.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from ..corpus import sha256
from ..discovery.common import load, save
from ..hsx import read_hsx
from ..output import read_eigenvalues
from .electronic import RY_EV, _native_weights, bloch_matrices, orbital_metadata


def decompose(h, s, fermi_ev, temperature_ev, *, degeneracy=1):
    """C†SC=I, D=C f C† and H_L=S^-1/2 H S^-1/2, without regularization."""
    h, s = np.asarray(h), np.asarray(s)
    if (h.ndim != 2 or h.shape[0] != h.shape[1] or s.shape != h.shape
            or not np.isfinite(h).all() or not np.isfinite(s).all()
            or not np.allclose(h, h.conj().T, rtol=0, atol=1e-10)
            or not np.allclose(s, s.conj().T, rtol=0, atol=1e-10)
            or not np.isfinite([fermi_ev, temperature_ev]).all() or temperature_ev < 0
            or type(degeneracy) is not int or degeneracy not in (1, 2)):
        raise ValueError('Finite Hermitian H/S and physical occupation parameters required')
    sv, su = eigh(s)
    if sv[0] <= 0:
        raise ValueError('Overlap is not positive definite; no clipping is allowed')
    inverse_root = (su / np.sqrt(sv)) @ su.conj().T
    values, c = eigh(h, s)
    if not np.allclose(c.conj().T @ s @ c, np.eye(len(s)), atol=2e-7, rtol=0):
        raise ValueError('Generalized eigenvectors are not S-orthonormal')
    f = (1 / (1 + np.exp(np.clip((values-fermi_ev)/temperature_ev, -700, 700)))
         if temperature_ev else (values < fermi_ev).astype(float)) * degeneracy
    d = (c * f) @ c.conj().T
    residual = np.max(np.abs(h @ c - (s @ c) * values))
    return dict(hamiltonian_ev=h, overlap=s, coefficients=c, eigenvalues_ev=values,
        occupations=f, density=d, lowdin_hamiltonian_ev=inverse_root @ h @ inverse_root,
        overlap_eigenvalues=sv, generalized_residual_ev=float(residual))


def _basis(directory, orbitals, verify):
    """Retain actual PAO radial arrays for later same-basis comparisons."""
    descriptors, arrays, hashes = [], [], {}
    for species in sorted({o['species'] for o in orbitals}):
        path = directory/(species+'.ion.xml')
        hashes[species] = verify(path)
        tree = ET.parse(path)
        if tree.findtext('symbol', '').strip() != species:
            raise ValueError('PAO species differs from HSX')
        shells = {}
        for orbital in tree.findall('paos/orbital'):
            key = tuple(int(orbital.get(k)) for k in ('n', 'l', 'z'))
            if key in shells:
                raise ValueError('Repeated PAO radial shell')
            shells[key] = orbital.find('radfunc')
        expected = {(o['n'], o['l'], o['zeta']) for o in orbitals if o['species'] == species}
        if set(shells) != expected:
            raise ValueError('HSX and PAO radial shell inventory differs')
        for key, radial in sorted(shells.items()):
            fields = [np.array([float(x) for x in radial.findtext(field, '').split()])
                      for field in ('npts', 'delta', 'cutoff', 'data')]
            npts, delta, cutoff, values = fields
            if (any(not x.size or not np.isfinite(x).all() for x in fields)
                    or npts.size != 1 or delta.size != 1 or cutoff.size != 1
                    or not 1 <= npts[0] <= 100000 or npts[0] != int(npts[0])
                    or values.size != 2*int(npts[0]) or delta[0] <= 0 or cutoff[0] <= 0):
                raise ValueError('Invalid PAO radial arrays')
            descriptors.append(dict(species=species, n=key[0], l=key[1], zeta=key[2],
                                    lengths=[len(x) for x in fields]))
            arrays.extend(fields)
    return descriptors, np.concatenate(arrays), hashes


def _groups(orbitals, spinor):
    keys = sorted({(o['atom_index'], o['species'], o['l']) for o in orbitals})
    return [dict(atom_index=a, species=species, l=l,
        indices=[j for o in orbitals if (o['atom_index'], o['species'], o['l']) == (a, species, l)
                 for j in ([2*o['index'], 2*o['index']+1] if spinor else [o['index']])])
        for a, species, l in keys]


def coupling_summary(matrix, groups, *, limit=40):
    """Block norms in the chosen symmetric Löwdin representation, in eV."""
    rows = []
    for i, a in enumerate(groups):
        for j in range(i+1, len(groups)):
            block = matrix[np.ix_(a['indices'], groups[j]['indices'])]
            rows.append(dict(left=i, right=j, frobenius_ev=float(np.linalg.norm(block)),
                             largest_singular_value_ev=float(np.linalg.svd(block, compute_uv=False)[0])))
    rows.sort(key=lambda x: (-x['frobenius_ev'], x['left'], x['right']))
    return dict(groups=groups, strongest_couplings=rows[:limit], omitted_couplings=max(0, len(rows)-limit),
        on_group_levels_ev=[np.linalg.eigvalsh(matrix[np.ix_(g['indices'], g['indices'])]).tolist() for g in groups],
        scope='Symmetric Lowdin atom/l blocks, including all zetas and spin components; delocalized, basis-dependent coupling/level proxies, not unique hybridization or crystal-field observables')


def export_matrices(report_path, stage, native_directory, output, *, k_indices=None, collinear_spin=0, collection=None):
    """Reconstruct at most eight native k points from an accepted saved stage.

    Native files may have been recovered separately; their bytes must match the
    original report. Nothing is computed by a native solver or submitted here.
    """
    from .evidence import verify_report as _verified_report
    report_path = Path(report_path).resolve()
    report = _verified_report(report_path, collection)
    matches = [s for s in report['stages'] if s['stage'] == stage]
    if len(matches) != 1 or matches[0]['accepted'] is not True or not matches[0].get('branch', {'passed': True})['passed']:
        raise ValueError('Select an accepted native stage with passing magnetic branch')
    native = Path(native_directory).resolve()
    def verify(path):
        if not (path.resolve().is_relative_to(native) or path.resolve().is_relative_to(report_path.parent)):
            raise ValueError('Native artifact escapes the supplied evidence directories')
        digest = sha256(path)
        if report['artifacts'].get(stage+'/'+path.name) != digest:
            raise ValueError('Native artifact is not bound by the report: '+path.name)
        return digest
    paths = dict(hsx=native/'remag.HSX', kp=report_path.parent/stage/'remag.KP', eig=report_path.parent/stage/'remag.EIG')
    evidence = {key: verify(path) for key, path in paths.items()}
    hsx = read_hsx(paths['hsx'])
    if int(hsx['header'][1]) > 256:
        raise ValueError('Selected-k matrices are bounded to 256 spatial orbitals')
    orbitals = orbital_metadata(hsx)
    radial_layout, radial, ion_hashes = _basis(native, orbitals, verify)
    lines = paths['kp'].read_text().splitlines()
    count = int(lines[0])
    if not 1 <= count <= 100000:
        raise ValueError('Native sampling exceeds the 100000-k-point input bound')
    kp = np.array([[float(x) for x in line.split()] for line in lines[1:]])
    if kp.shape != (count, 5) or not np.isfinite(kp).all() or not np.array_equal(kp[:, 0], np.arange(1, count+1)):
        raise ValueError('Invalid native KP sampling')
    weights, _, _ = _native_weights([line.split()[4] for line in lines[1:]])
    indices = [0] if k_indices is None else k_indices
    if (not isinstance(indices, list) or not 1 <= len(indices) <= 8 or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < count for i in indices)):
        raise ValueError('Select 1..8 distinct in-range native k indices')
    indices = sorted(indices)
    spins = int(hsx['header'][2])
    if type(collinear_spin) is not int or (spins <= 2 and not 0 <= collinear_spin < spins) or (spins > 2 and collinear_spin != 0):
        raise ValueError('Invalid spin channel')
    eig = read_eigenvalues(paths['eig'])
    if eig['nk'] != count or eig['nspin'] != spins:
        raise ValueError('Native KP/EIG/HSX dimensions differ')
    native_eigen = np.array(eig['values_ev'])[indices]
    if spins <= 2:
        native_eigen = native_eigen[:, collinear_spin*eig['nbands']:(collinear_spin+1)*eig['nbands']]
    k = kp[indices, 1:4] @ hsx['cell'].reshape(3, 3).T / (2*np.pi)
    fermi, temperature = hsx['fermi_ry']*RY_EV, hsx['temperature_ry']*RY_EV
    decompositions = [decompose(*bloch_matrices(hsx, point, collinear_spin=collinear_spin), fermi, temperature,
                                degeneracy=2 if spins == 1 else 1) for point in k]
    arrays = {key: np.array([d[key] for d in decompositions]) for key in decompositions[0]}
    if native_eigen.shape != arrays['eigenvalues_ev'].shape or not np.isfinite(native_eigen).all():
        raise ValueError('Native eigenvalue dimensions differ')
    native_error = float(np.max(abs(native_eigen-arrays['eigenvalues_ev'])))
    if native_error > 1e-3:
        raise ValueError('Native selected-k eigenvalue comparison failed')
    arrays.update(k_fractional=k, k_weights=weights[indices], radial_functions=radial,
                  cell_bohr=hsx['cell'], positions_bohr=hsx['positions'], native_eigenvalues_ev=native_eigen,
                  native_eigenvalue_residual_ev=native_eigen-arrays['eigenvalues_ev'])
    groups = _groups(orbitals, spins > 2)
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(root/'matrices.npz', **arrays)
    result = dict(format='REMAG-ELECTRONIC-MATRICES/1', artifact='matrices.npz', artifact_sha256=sha256(root/'matrices.npz'),
        report_sha256=sha256(report_path), source_sha256=report['source_sha256'], executable_sha256=report['executable_sha256'],
        stage=stage, evidence=evidence, ion_sha256=ion_hashes, orbitals=orbitals, radial_layout=radial_layout,
        spin_components=spins, collinear_spin=collinear_spin if spins <= 2 else None,
        spin_frame='Cartesian solver frame; interleaved up/down', lattice_gauge='H[j,i] += H_sparse[i,j,R] exp(-2pi i k.R)',
        fermi_ev=fermi, temperature_ev=temperature, k_indices=indices, native_k_count=count,
        selected_k_weight=float(weights[indices].sum()), native_max_abs_error_ev=native_error,
        reconstruction_max_residual_ev=float(np.max(arrays['generalized_residual_ev'])),
        occupation_model='Fermi-Dirac reconstructed at HSX chemical potential/temperature; not native DM export',
        native_comparison_scope='Reconstruction check against printed EIG/KP, not a physical uncertainty bound or micro-eV qualification',
        density_convention='D=C f C†, Tr(D S)=electron count, Tr(D H)=band sum; each k is unweighted',
        coupling=[coupling_summary(d['lowdin_hamiltonian_ev'], groups) for d in decompositions],
        band_sums_ev=[float(d['occupations'] @ d['eigenvalues_ev']) for d in decompositions],
        electron_counts=[float(d['occupations'].sum()) for d in decompositions],
        scope='Selected-k postprocessing of saved self-consistent H/S; not total-energy anisotropy, isolated SOC, orbital moments or a force-theorem calculation',
        physical_accuracy_qualified=False)
    save(root/'matrices.json', result)
    return result


def _load_matrices(path):
    path = Path(path).resolve(); meta = load(path)
    if meta.get('format') != 'REMAG-ELECTRONIC-MATRICES/1':
        raise ValueError('Expected selected-k matrix metadata')
    artifact = (path.parent/meta['artifact']).resolve()
    if not artifact.is_relative_to(path.parent) or sha256(artifact) != meta['artifact_sha256']:
        raise ValueError('Matrix artifact binding differs')
    with zipfile.ZipFile(artifact) as archive:
        if sum(x.file_size for x in archive.infolist()) > 512*1024*1024:
            raise ValueError('Matrix artifact exceeds inspection size bound')
    with np.load(artifact, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if any(not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError('Nonfinite matrix artifact')
    e = arrays['eigenvalues_ev']; count = len(meta['k_indices'])
    size = len(meta['orbitals'])*(2 if meta['spin_components'] > 2 else 1)
    if (e.shape != (count, size) or not 1 <= count <= 8 or not 1 <= size <= 512
            or any(arrays[k].shape != (count, size, size) for k in ('hamiltonian_ev', 'overlap', 'coefficients', 'density', 'lowdin_hamiltonian_ev'))
            or arrays['occupations'].shape != e.shape or arrays['native_eigenvalue_residual_ev'].shape != e.shape
            or arrays['k_fractional'].shape != (count, 3)
            or arrays['k_weights'].shape != (count,) or np.any(arrays['k_weights'] < 0)
            or not 0 <= meta['native_max_abs_error_ev'] <= 1e-3):
        raise ValueError('Invalid matrix dimensions, sampling or native comparison')
    for s, c in zip(arrays['overlap'], arrays['coefficients']):
        if not np.allclose(c.conj().T @ s @ c, np.eye(size), rtol=0, atol=2e-7):
            raise ValueError('Saved vectors are not S-orthonormal')
    return meta, arrays


def _clusters(energies, tolerance):
    return np.split(np.arange(len(energies)), np.flatnonzero(np.diff(energies) > tolerance)+1)


def state_correspondence(ca, cb, s, ea, eb, *, degeneracy_tolerance_ev=1e-5):
    """Maximum-weight association plus phase/degenerate-subspace invariants.

    Individual vectors inside degenerate spaces have no unique association.
    Projector traces and principal overlap singular values survive their unitary
    mixing. This is an endpoint association, not adiabatic band tracking.
    """
    overlap = ca.conj().T @ s @ cb
    weights = abs(overlap)**2
    if not (np.allclose(weights.sum(axis=0), 1, atol=2e-6, rtol=0)
            and np.allclose(weights.sum(axis=1), 1, atol=2e-6, rtol=0)):
        raise ValueError('Endpoint states are not complete orthonormal bases in a common metric')
    left, right = linear_sum_assignment(-weights)
    a, b = _clusters(ea, degeneracy_tolerance_ev), _clusters(eb, degeneracy_tolerance_ev)
    a_size = {int(i): len(g) for g in a for i in g}; b_size = {int(i): len(g) for g in b for i in g}
    states = []
    for i, j in zip(left, right):
        row_margin = float(weights[i, j]-np.max(np.delete(weights[i], j), initial=0))
        column_margin = float(weights[i, j]-np.max(np.delete(weights[:, j], i), initial=0))
        degenerate = a_size[i] > 1 or b_size[j] > 1
        states.append(dict(before_band=int(i), after_band=int(j), overlap_squared=float(weights[i, j]),
            energy_shift_ev=float(eb[j]-ea[i]), energy_degeneracy_present=degenerate,
            row_overlap_margin=row_margin, column_overlap_margin=column_margin,
            assignment_ambiguous=degenerate or min(row_margin, column_margin) <= 1e-6))
    clusters = []
    for group in a:
        candidates = []
        for target in b:
            singular = np.linalg.svd(overlap[np.ix_(group, target)], compute_uv=False)
            candidates.append(dict(after_bands=target.tolist(), projector_trace=float(np.sum(singular**2)),
                                   principal_overlap_singular_values=singular.tolist()))
        candidates.sort(key=lambda x: -x['projector_trace'])
        clusters.append(dict(before_bands=group.tolist(), matches=candidates[:3],
                             omitted_projector_trace=float(sum(x['projector_trace'] for x in candidates[3:]))))
    return states, clusters, weights


def compare_matrices(before, after, output, *, energy_window_ev=None, degeneracy_tolerance_ev=1e-5, limit=40):
    """Same-geometry, same-radial-basis endpoint association in the laboratory frame."""
    a, da = _load_matrices(before); b, db = _load_matrices(after)
    window = [-3, 3] if energy_window_ev is None else energy_window_ev
    if (not isinstance(window, list) or len(window) != 2 or any(type(x) not in (int, float) or not np.isfinite(x) for x in window)
            or not -100 <= window[0] < window[1] <= 100 or type(limit) is not int or not 1 <= limit <= 200
            or type(degeneracy_tolerance_ev) not in (int, float) or not 1e-10 <= degeneracy_tolerance_ev <= .05):
        raise ValueError('Invalid state comparison selection or degeneracy tolerance')
    for key in ('source_sha256', 'executable_sha256', 'orbitals', 'radial_layout', 'spin_components',
                'collinear_spin', 'spin_frame', 'lattice_gauge', 'k_indices', 'native_k_count', 'temperature_ev'):
        if a[key] != b[key]:
            raise ValueError('Common basis/comparison metadata differs: '+key)
    differences = {}
    for key in ('cell_bohr', 'positions_bohr', 'radial_functions', 'k_fractional', 'k_weights', 'overlap'):
        if da[key].shape != db[key].shape:
            raise ValueError('Common basis/sampling dimensions differ: '+key)
        differences[key] = float(np.max(abs(da[key]-db[key])))
        if differences[key] > 1e-10:
            raise ValueError('Common geometry/radial basis/metric differs: '+key)
    rows, cross, perturbations = [], [], []
    for k, (ca, cb, s, ea, eb, ha, hb) in enumerate(zip(da['coefficients'], db['coefficients'], da['overlap'],
            da['eigenvalues_ev'], db['eigenvalues_ev'], da['hamiltonian_ev'], db['hamiltonian_ev'])):
        states, clusters, weights = state_correspondence(ca, cb, s, ea, eb, degeneracy_tolerance_ev=degeneracy_tolerance_ev)
        delta = ca.conj().T @ (hb-ha) @ ca
        selected = {i for i in range(len(ea)) if window[0] <= ea[i]-a['fermi_ev'] <= window[1]}
        selected_states = [x for x in states if x['before_band'] in selected]
        selected_states.sort(key=lambda x: (abs(ea[x['before_band']]-a['fermi_ev']), x['before_band']))
        for state in selected_states:
            i, j = state['before_band'], state['after_band']
            state.update(before_energy_minus_fermi_ev=float(ea[i]-a['fermi_ev']),
                after_energy_minus_fermi_ev=float(eb[j]-b['fermi_ev']),
                delta_h_expectation_ev=float(delta[i, i].real),
                before_native_reconstruction_residual_ev=float(da['native_eigenvalue_residual_ev'][k,i]),
                after_native_reconstruction_residual_ev=float(db['native_eigenvalue_residual_ev'][k,j]))
        fa, fb = da['occupations'][k], db['occupations'][k]
        rows.append(dict(native_k_index=a['k_indices'][k], states=selected_states[:limit],
            omitted_states=max(0, len(selected_states)-limit),
            subspaces=[g for g in clusters if set(g['before_bands']) & {x['before_band'] for x in selected_states[:limit]}],
            band_sum_change_ev=float(fb @ eb-fa @ ea),
            fermi_relative_band_sum_change_ev=float(fb @ (eb-b['fermi_ev'])-fa @ (ea-a['fermi_ev'])),
            electron_count_change=float(fb.sum()-fa.sum()), k_weight=float(da['k_weights'][k])))
        cross.append(weights); perturbations.append(delta)
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(root/'comparison.npz', state_overlap_squared=np.array(cross), delta_h_before_eigenbasis_ev=np.array(perturbations))
    result = dict(format='REMAG-ELECTRONIC-MECHANISM-COMPARISON/1', before_sha256=sha256(before), after_sha256=sha256(after),
        artifact='comparison.npz', artifact_sha256=sha256(root/'comparison.npz'), common_basis_max_abs_differences=differences,
        energy_window_ev=window, degeneracy_tolerance_ev=degeneracy_tolerance_ev, fermi_shift_ev=b['fermi_ev']-a['fermi_ev'],
        kpoints=rows, selected_k_weight=float(da['k_weights'].sum()),
        weighted_selected_band_sum_change_ev=sum(r['k_weight']*r['band_sum_change_ev'] for r in rows),
        state_scope='One-to-one maximum overlap endpoint association in fixed Cartesian spin frame; not adiabatic tracking. Degenerate-state labels are nonunique; use subspace projector traces. Assignment ambiguity also flags row/column overlap margins <=1e-6; this is a diagnostic, not a match certificate.',
        operator_scope='C_before† (H_after-H_before) C_before includes all self-consistent potential changes; not an isolated SOC operator or a second-order SOC decomposition.',
        energy_scope='Partial weighted self-consistent band sum, no missing-k renormalization; not total E/F anisotropy or force theorem. Raw and own-Fermi-relative sums have explicit energy-reference dependence.',
        mechanism_verified=False, physical_accuracy_qualified=False)
    save(root/'comparison.json', result)
    return result
