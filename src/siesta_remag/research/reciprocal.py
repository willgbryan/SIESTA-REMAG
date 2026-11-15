"""Periodic sampled regions and ordered saved-k band paths; no new eigensolve."""
from pathlib import Path
import numpy as np

from ..corpus import sha256
from ..discovery.common import save,digest
from ..hsx import read_hsx


def region_indices(k_fractional, region):
    """Componentwise reciprocal torus box, not a Cartesian sphere or first BZ."""
    if not isinstance(region,dict) or set(region)!={'center','half_width'}:
        raise ValueError('Reciprocal region needs center and half_width triples')
    for key in region:
        v=region[key]
        if (not isinstance(v,list) or len(v)!=3 or
                any(type(x) not in (int,float) or not np.isfinite(x) for x in v)):
            raise ValueError('Reciprocal region requires finite numerical triples')
    center,width=np.array(region['center']),np.array(region['half_width'])
    if np.any(abs(center)>10000) or np.any(width<0) or np.any(width>.5):
        raise ValueError('Region center must be bounded and half widths within [0,0.5]')
    k=np.asarray(k_fractional)
    if k.ndim!=2 or k.shape[1]!=3 or not np.isfinite(k).all():raise ValueError('Finite fractional k points required')
    delta=(k-center+.5)%1-.5
    return np.flatnonzero(np.all(abs(delta)<=width+1e-12,axis=1)).tolist()


def path_geometry(k_fractional, cell_angstrom, k_indices, breaks=None):
    ks=np.asarray(k_fractional);cell=np.asarray(cell_angstrom,float)
    if (ks.ndim!=2 or ks.shape[1]!=3 or not np.isfinite(ks).all() or cell.shape!=(3,3)
            or not np.isfinite(cell).all() or abs(np.linalg.det(cell))<1e-12):
        raise ValueError('Finite k coordinates and nonsingular cell required')
    if (not isinstance(k_indices,list) or not 2<=len(k_indices)<=256 or
            any(type(k) is not int or not 0<=k<len(ks) for k in k_indices)):
        raise ValueError('Band path needs 2..256 ordered saved k indices')
    breaks=[] if breaks is None else breaks
    if (not isinstance(breaks,list) or any(type(i) is not int or not 1<=i<len(k_indices) for i in breaks)
            or breaks!=sorted(set(breaks))):raise ValueError('Path breaks must be sorted distinct interior sample indices')
    reciprocal=2*np.pi*np.linalg.inv(cell).T
    distances=np.linalg.norm(np.diff(ks[k_indices],axis=0)@reciprocal,axis=1)
    for i in breaks:distances[i-1]=0
    if not np.any(distances>1e-12):raise ValueError('Band path has no nonzero connected segment')
    return np.r_[0,np.cumsum(distances)],reciprocal


def export_band_path(report_path, stage, spectrum_path, native_hsx, output, *, k_indices, breaks=None, selection=None, collection=None):
    from .inspection import load_spectrum,_select
    from .sampling import native_report_context,BOHR_ANGSTROM,_bound_file
    from .electronic import _native_weights,orbital_metadata,RY_EV
    from ..output import read_eigenvalues
    meta,arrays=load_spectrum(spectrum_path)
    endpoint=dict(report=report_path,stage=stage)
    if collection is not None:endpoint['collection']=collection
    context=native_report_context(endpoint,meta)
    native_hsx=Path(native_hsx)
    if native_hsx.stat().st_size>512*1024*1024 or sha256(native_hsx)!=meta['hsx_sha256']:
        raise ValueError('Band path native Hamiltonian binding differs')
    hsx=read_hsx(native_hsx)
    # Reproduce sampling and eigenvalue acceptance from the bound text outputs;
    # a self-consistent NPZ/metadata pair alone cannot certify native agreement.
    lines=_bound_file(context,'remag.KP',32*1024*1024).read_text().splitlines()
    kp=np.array([[float(x) for x in line.split()] for line in lines[1:]])
    count=int(lines[0])
    if kp.shape!=(count,5) or not np.isfinite(kp).all() or not np.array_equal(kp[:,0],np.arange(1,count+1)):
        raise ValueError('Invalid bound native KP sampling')
    expected_k=kp[:,1:4]@hsx['cell'].reshape(3,3).T/(2*np.pi)
    weights,_,_=_native_weights([line.split()[4] for line in lines[1:]])
    eigen=read_eigenvalues(_bound_file(context,'remag.EIG',256*1024*1024));native=np.array(eigen['values_ev'])
    spin=meta.get('collinear_spin')
    if eigen['nspin']<=2:
        if type(spin) is not int or not 0<=spin<eigen['nspin']:raise ValueError('Invalid native collinear channel')
        native=native[:,spin*eigen['nbands']:(spin+1)*eigen['nbands']]
    if (eigen['nspin']!=int(hsx['header'][2]) or meta['spin_components']!=eigen['nspin'] or eigen['nk']!=count
            or arrays['k_fractional'].shape!=expected_k.shape or not np.allclose(arrays['k_fractional'],expected_k,rtol=0,atol=1e-12)
            or arrays['k_weights'].shape!=weights.shape or not np.allclose(arrays['k_weights'],weights,rtol=0,atol=1e-12)
            or native.shape!=arrays['eigenvalues_ev'].shape or not np.isfinite(native).all()
            or np.max(abs(native-arrays['eigenvalues_ev']))>1e-3 or meta['orbitals']!=orbital_metadata(hsx)
            or not np.allclose([meta['fermi_ev'],meta['temperature_ev']],np.array([hsx['fermi_ry'],hsx['temperature_ry']])*RY_EV,rtol=0,atol=1e-12)):
        raise ValueError('Band spectrum differs from bound native sampling, eigenvalues or basis')
    residual=float(np.max(abs(native-arrays['eigenvalues_ev'])))
    cell=hsx['cell'].reshape(3,3)*BOHR_ANGSTROM['codata2018']
    distance,reciprocal=path_geometry(arrays['k_fractional'],cell,k_indices,breaks)
    selection={} if selection is None else selection
    if not isinstance(selection,dict) or set(selection)-{'atom_indices','orbital_indices','l'}:
        raise ValueError('Band projections select atoms, orbitals or angular momentum; path supplies k points')
    orbitals,_,_,_=_select(meta,arrays,selection)
    selected=arrays['orbital_mulliken'][k_indices][:,orbitals,:].sum(axis=1)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(root/'bands.npz',k_indices=k_indices,k_fractional=arrays['k_fractional'][k_indices],
        distance_inv_angstrom=distance,eigenvalues_ev=arrays['eigenvalues_ev'][k_indices],
        selected_mulliken=selected,occupations=arrays['occupations'][k_indices],
        spin_pauli=arrays['spin_pauli'][k_indices] if arrays['spin_pauli'].size else np.array([]))
    result=dict(format='REMAG-BAND-PATH/1',**context['binding'],spectrum_sha256=sha256(spectrum_path),
        hsx_sha256=meta['hsx_sha256'],spectrum_artifact_sha256=meta['artifact_sha256'],
        artifact='bands.npz',artifact_sha256=sha256(root/'bands.npz'),k_indices=k_indices,breaks=breaks or [],
        fermi_ev=meta['fermi_ev'],energy_units='eV',distance_units='angstrom^-1',cell_angstrom=cell.tolist(),
        reciprocal_rows_inv_angstrom=reciprocal.tolist(),bohr_angstrom=BOHR_ANGSTROM['codata2018'],
        selected_orbitals=[meta['orbitals'][i] for i in orbitals],selection=selection,
        band_count=arrays['eigenvalues_ev'].shape[1],native_max_abs_error_ev=residual,
        native_sampling_reproduced=True,
        projection=meta['projection'],occupations_model=meta['occupations_model'],spin_frame=meta['spin_frame'],
        scope='Saved sampled eigenvalues in energy rank order; straight segments guide the eye. No intermediate states, band tracking or new solver execution.',
        quadrature_defined=False,physical_accuracy_qualified=False)
    save(root/'bands.json',result)
    return result


def sampled_view(meta, arrays, atom_count):
    """Bound the complete browser payload; never silently downsample a mesh."""
    energies=arrays['eigenvalues_ev']-meta['fermi_ev']
    count=energies.size*(atom_count+5)+len(energies)*4
    if count>2_000_000 or energies.shape[0]>4096 or energies.shape[1]>2048:
        return dict(available=False,reason='Full interactive sample exceeds 4096 k points, 2048 states or two million numbers; use artifact tools',number_count=count)
    pop=np.stack([arrays['orbital_mulliken'][:,[i for i,o in enumerate(meta['orbitals']) if o['atom_index']==a],:].sum(axis=1)
                  for a in range(atom_count)],axis=1)
    return dict(available=True,k_fractional=arrays['k_fractional'].tolist(),k_weights=arrays['k_weights'].tolist(),
        energy_minus_fermi_ev=energies.tolist(),occupations=arrays['occupations'].tolist(),
        atom_mulliken=pop.tolist(),spin_pauli=arrays['spin_pauli'].tolist(),
        degeneracy=2 if meta['spin_components']==1 else 1,artifact_sha256=meta['artifact_sha256'],identity_sha256=digest(meta),
        complete_exported_sampling=True,number_count=count)
