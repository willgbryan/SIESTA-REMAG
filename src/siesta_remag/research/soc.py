"""Native spin-dependent nonlocal operator in checked self-consistent states.

This is an operator diagnostic, not a partition of total MAE or a perturbative
energy. The producer subtracts the nonlocal spin trace before HSX encoding.
"""
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from ..corpus import sha256
from ..discovery.common import load, save
from ..hsx import read_hsx
from .electronic import bloch_matrices
from .mechanisms import export_matrices, _load_matrices, coupling_summary, _groups

NATIVE_METADATA = dict(format='REMAG-NATIVE-SOC/1', operator='spin-traceless-offsite-nonlocal',
    producer='pinned-setup_hamiltonian.F', host_nonlocal_h0=True, spin_components=8,
    unit='Ry', encoding='native-HSX-v2')


def validate_native_soc(directory):
    """Check a newly collected operator against its total-H native export."""
    root=Path(directory).resolve()
    for name in ('remag.SOC.HSX','remag.HSX','remag.SOC.json'):
        p=root/name
        if not p.resolve().is_relative_to(root) or p.stat().st_size>512*1024*1024:
            raise ValueError('Native SOC artifact escapes its directory or exceeds the bound')
    if (root/'remag.SOC.json').stat().st_size>4096 or load(root/'remag.SOC.json')!=NATIVE_METADATA:
        raise ValueError('Unsupported native SOC producer or operator definition')
    hsx,operator=read_hsx(root/'remag.HSX'),read_hsx(root/'remag.SOC.HSX')
    if hsx['header'][2]!=8 or operator['precision']!=1:
        raise ValueError('SOC inspection requires a double-precision SOC operator')
    for key in ('header','cell','positions','labels','offsets','counts','columns','overlap'):
        if not np.array_equal(hsx[key],operator[key]):
            raise ValueError('SOC and total-H geometry/basis/topology/overlap differ: '+key)
    for key in ('basis','kpoints','fermi_ry','temperature_ry','charge'):
        if hsx[key]!=operator[key]:raise ValueError('SOC and total-H metadata differ: '+key)
    if np.any(operator['h'][0]+operator['h'][1]) or np.any(operator['h'][4]+operator['h'][5]):
        raise ValueError('Native SOC operator has a nonzero spin trace')
    return operator


def project_soc(operator, overlap, coefficients, occupations):
    """Covariant V transforms as C†VC; no extra overlap factor."""
    v, s, c, f = map(np.asarray, (operator, overlap, coefficients, occupations))
    if (v.ndim != 2 or v.shape[0] != v.shape[1] or v.shape[0] % 2
            or s.shape != v.shape or c.shape != v.shape or f.shape != (len(v),)
            or any(not np.isfinite(x).all() for x in (v,s,c,f))
            or np.any(f < 0) or np.any(f > 1)
            or not np.allclose(v,v.conj().T,rtol=0,atol=1e-9)
            or not np.allclose(s,s.conj().T,rtol=0,atol=1e-10)):
        raise ValueError('Finite Hermitian spinor SOC/overlap and occupations required')
    if not np.allclose(v[::2,::2]+v[1::2,1::2],0,rtol=0,atol=1e-9):
        raise ValueError('Nonlocal SOC operator has a nonzero spin trace')
    w,u = eigh(s)
    if w[0] <= 0 or not np.allclose(c.conj().T@s@c,np.eye(len(v)),rtol=0,atol=2e-7):
        raise ValueError('Positive overlap and S-orthonormal states required')
    root=(u/np.sqrt(w))@u.conj().T
    state=c.conj().T@v@c
    expectation=float(f@state.diagonal().real)
    density=(c*f)@c.conj().T
    if not np.isclose(np.trace(density@v),expectation,rtol=1e-10,atol=1e-10):
        raise ValueError('SOC density and eigenstate contractions differ')
    return dict(state_matrix_ev=state,lowdin_operator_ev=root@v@root,
                occupied_expectation_ev=expectation)


def export_soc(report_path, stage, native_directory, output, *, k_indices=None, collection=None):
    """Export at most eight k points with matching total-H, S and PAO evidence."""
    from .tools import _verified_report
    report_path=Path(report_path).resolve(); native=Path(native_directory).resolve()
    report=_verified_report(report_path,collection)
    paths={key:native/name for key,name in [('operator','remag.SOC.HSX'),('metadata','remag.SOC.json'),('hsx','remag.HSX')]}
    evidence={}
    for key,path in paths.items():
        if not path.resolve().is_relative_to(native) or path.stat().st_size > 512*1024*1024:
            raise ValueError('Native SOC artifact escapes its directory or exceeds the bound')
        evidence[key]=sha256(path)
        if report['artifacts'].get(stage+'/'+path.name)!=evidence[key]:
            raise ValueError('Native SOC artifact is not bound by report: '+path.name)
    operator=validate_native_soc(native)
    if operator['header'][1]>256:raise ValueError('SOC inspection is bounded to 256 spatial orbitals')
    root=Path(output); root.mkdir(parents=True,exist_ok=False)
    states=export_matrices(report_path,stage,native,root/'states',k_indices=k_indices,collection=collection)
    _,arrays=_load_matrices(root/'states/matrices.json')
    operators=[]; projected=[]
    for point,s,c,f in zip(arrays['k_fractional'],arrays['overlap'],arrays['coefficients'],arrays['occupations']):
        v,vs=bloch_matrices(operator,point)
        if not np.allclose(vs,s,rtol=0,atol=1e-12):raise ValueError('Reconstructed SOC overlap differs')
        operators.append(v); projected.append(project_soc(v,s,c,f))
    result_arrays={key:np.array([p[key] for p in projected]) for key in projected[0]}
    result_arrays.update(operator_ev=np.array(operators),k_indices=np.array(states['k_indices']),
                         k_weights=arrays['k_weights'],eigenvalues_ev=arrays['eigenvalues_ev'],occupations=arrays['occupations'])
    np.savez_compressed(root/'soc.npz',**result_arrays)
    groups=_groups(states['orbitals'],True)
    result=dict(format='REMAG-SOC-MATRICES/1',artifact='soc.npz',artifact_sha256=sha256(root/'soc.npz'),
        report_sha256=sha256(report_path),source_sha256=report['source_sha256'],stage=stage,
        executable_sha256=report['executable_sha256'],evidence=evidence,
        implementation_sha256={name:sha256(Path(__file__).parent/name) for name in ('soc.py','mechanisms.py','electronic.py','../hsx.py')},
        states='states/matrices.json',states_sha256=sha256(root/'states/matrices.json'),
        operator=NATIVE_METADATA['operator'],orbitals=states['orbitals'],spin_frame=states['spin_frame'],
        lattice_gauge=states['lattice_gauge'],unit='eV',k_indices=states['k_indices'],native_k_count=states['native_k_count'],
        selected_k_weight=states['selected_k_weight'],native_max_abs_error_ev=states['native_max_abs_error_ev'],
        occupied_expectations_ev=result_arrays['occupied_expectation_ev'].tolist(),
        selected_weighted_expectation_ev=float(arrays['k_weights']@result_arrays['occupied_expectation_ev']),
        state_diagonal_ev=[p['state_matrix_ev'].diagonal().real.tolist() for p in projected],
        coupling=[coupling_summary(p['lowdin_operator_ev'],groups) for p in projected],
        projection='C† V_SOC C, C† S C=I; no extra overlap. Lowdin operator S^-1/2 V_SOC S^-1/2.',
        occupation_model=states['occupation_model'],
        scope='Spin-dependent off-site nonlocal pseudopotential only; spin trace removed even when native SR/SO split is disabled. Excludes exchange and other total-H terms. State entries depend on phases and degenerate-subspace choice; atom/l Lowdin blocks are basis-dependent. Occupied expectation is not total MAE, a force-theorem energy or second-order perturbation theory. Selected k weights are not renormalized.',
        physical_accuracy_qualified=False)
    save(root/'soc.json',result)
    return result
