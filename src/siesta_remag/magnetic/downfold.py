"""Optional PAO projection and Wannier90 import; GPU Fourier interpolation."""
from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np
from .gpu import cuda, adjoint, cell_gauge, generalized_eigh
from .state import MagneticState, file_hash
from .operators import RY_EV


@dataclass
class ReducedHamiltonian:
    metadata: dict
    translations: np.ndarray
    coefficients: dict
    report: dict

    def interpolate(self,kpoints):
        """Batched GPU sum_R exp(+2pi i k.R) H(R), common operator gauge."""
        cp,_=cuda();k=np.asarray(kpoints,float)
        if k.ndim!=2 or k.shape[1]!=3 or not np.isfinite(k).all():raise ValueError('Finite fractional k coordinates required')
        phase=cp.exp(2j*np.pi*cp.asarray(k)@cp.asarray(self.translations).T)
        result={}
        for name,coeff in self.coefficients.items():
            a=cp.einsum('kr,rij->kij',phase,cp.asarray(coeff))
            # Hermitian trigonometric interpolation handles the even-mesh
            # Nyquist term symmetrically, rather than inventing complex bands.
            result[name]=.5*(a+adjoint(a))
        return result

    def magnetic_state(self,kpoints,weights):
        if not {'h_base','h_mag','h_soc','overlap'}<=self.coefficients.keys():
            raise ValueError('Magnetic use requires consistently projected base, magnetic and SOC operators')
        cp,_=cuda();banks=self.interpolate(kpoints)
        m=dict(self.metadata,mesh_complete=True,kpoint_convention='fractional_reciprocal',
               bloch_gauge='cell_periodic_positive',representation='reduced_pao')
        return MagneticState(m,**{key:cp.asnumpy(banks[key]) for key in ('h_base','h_mag','h_soc','overlap')},
                             kpoints=kpoints,weights=weights)

    def spectrum(self,kpoints):
        cp,_=cuda();a=self.interpolate(kpoints)
        h=a.get('hamiltonian')
        if h is None:h=a['h_base']+a['h_mag']+a['h_soc']
        s=a.get('overlap',cp.broadcast_to(cp.eye(h.shape[-1]),h.shape))
        e,_,_,checks=generalized_eigh(h,s)
        return cp.asnumpy(e)*RY_EV,checks

    def save(self,directory):
        path=Path(directory);path.mkdir(parents=True,exist_ok=False)
        np.savez(path/'reduced.npz',translations=self.translations,**self.coefficients)
        manifest=dict(metadata=self.metadata,report=self.report,array_sha256=file_hash(path/'reduced.npz'))
        (path/'reduced.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        return path/'reduced.json'


def build_reduced_hamiltonian(state, *, orbital_indices, electrons=None, target='magnetic_response'):
    """Cholesky-orthogonalized selected PAO subspace, no disentanglement.

    Truncation requires its own electron count. Sampling is a complete uniform
    mesh, explicitly supplied in metadata['settings']['kmesh']. The returned
    model is provisional until the desired response is compared with the PAO
    calculation; no automatic replacement or performance claim is made.
    """
    if target!='magnetic_response':raise ValueError('Only magnetic_response is supported')
    started=time.perf_counter();cp,cupyx=cuda();banks,k=cell_gauge(state)
    ids=np.asarray(orbital_indices)
    norb=state.shape[-1]//2
    if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or not len(ids) or len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=norb):
        raise ValueError('Distinct valid spatial orbital indices required')
    if electrons is None:
        if len(ids)!=norb:raise ValueError('Specify the retained electron count for a truncated PAO model')
        electrons=state.metadata['electrons']
    mesh=np.asarray(state.metadata.get('settings',{}).get('kmesh'))
    if mesh.shape!=(3,) or not np.issubdtype(mesh.dtype,np.integer) or np.any(mesh<1) or np.prod(mesh)!=len(k):
        raise ValueError('Complete uniform source kmesh dimensions required for Fourier interpolation')
    coords=np.mod(cp.asnumpy(k),1.)
    if not np.allclose(state.weights,1/len(k),rtol=0,atol=1.e-12):raise ValueError('Fourier reduction requires uniform full-mesh weights')
    for a in range(3):
        values=np.unique(np.round(coords[:,a],10))
        if len(values)!=mesh[a] or not np.allclose(np.diff(np.r_[values,values[0]+1]),1/mesh[a],atol=1.e-9):
            raise ValueError('Source k points are not the declared complete uniform mesh')
    if len(np.unique(np.round(coords,10),axis=0))!=len(k):raise ValueError('Duplicate source k points')
    selected=np.ravel(np.column_stack((2*ids,2*ids+1)))
    ix=cp.asarray(selected);s=banks[3][:,ix,:][:,:,ix]
    with cupyx.errstate(linalg='raise'):
        l=cp.linalg.cholesky(s)
        q=cp.linalg.solve(adjoint(l),cp.broadcast_to(cp.eye(len(ix)),s.shape))
    reduced={name:adjoint(q)@bank[:,ix,:][:,:,ix]@q for name,bank in zip(('h_base','h_mag','h_soc','overlap'),banks,strict=True)}
    translations=np.array(list(__import__('itertools').product(*(range(-(int(v)//2),int(v)-int(v)//2) for v in mesh))))
    phase=cp.exp(-2j*np.pi*cp.asarray(translations)@k.T)/len(k)
    coefficients={name:cp.asnumpy(cp.einsum('rk,kij->rij',phase,a)) for name,a in reduced.items()}
    metadata=dict(state.metadata,electrons=electrons,representation='reduced_pao',parent_calculation_id=state.metadata['calculation_id'],
                  original_spatial_orbital_indices=ids.tolist(),kpoint_convention='fractional_reciprocal',bloch_gauge='cell_periodic_positive')
    if metadata.get('orbitals') is not None:metadata['orbitals']=[metadata['orbitals'][i] for i in ids]
    report=dict(status='screening',approximation='selected_pao_subspace',original_spinor_dimension=2*norb,
        reduced_spinor_dimension=len(selected),basis_mapping=ids.tolist(),target=target,
        response_error=None,measured_speedup=None,numerical_sensitivity=None,raw_evidence=[],
        performance=dict(construction_seconds=time.perf_counter()-started),
        limitations=['Orbital truncation is not energy-window disentanglement or exact elimination of excluded states.',
            'Use only after comparing the intended bands and magnetic response with the complete PAO model.',
            'Interpolation inherits finite-mesh aliasing; projected operators share one frozen gauge.',
            'No automatic PAO replacement: a reduced model may cost more if setup dominates.'])
    model=ReducedHamiltonian(metadata,translations,coefficients,report)
    replay=model.interpolate(cp.asnumpy(k))
    report['source_mesh_reconstruction_max_abs_ry']=float(max(cp.max(cp.abs(replay[name]-a)).get() for name,a in reduced.items()))
    if report['source_mesh_reconstruction_max_abs_ry']>1.e-9:raise RuntimeError('Reduced Fourier reconstruction failed at source k points')
    return model


def import_wannier90(path, *, metadata):
    """Read standard spinor seedname_hr.dat in eV; no localization optimizer.

    The Hamiltonian alone supports spectra/Green functions. It cannot provide
    a magnetic decomposition: separately projected operators are required.
    """
    path=Path(path)
    with path.open() as stream:
        stream.readline();n=int(stream.readline());nr=int(stream.readline())
        if n<2 or n%2 or nr<1:raise ValueError('An even-dimensional spinor Wannier Hamiltonian is required')
        degeneracy=[]
        while len(degeneracy)<nr:
            line=stream.readline()
            if not line:raise ValueError('Truncated Wannier degeneracies')
            degeneracy.extend(map(int,line.split()))
        if len(degeneracy)!=nr or min(degeneracy)<1:raise ValueError('Invalid Wannier degeneracies')
        translations=[];matrices=[]
        for r in range(nr):
            matrix=np.empty((n,n),complex);seen=set();translation=None
            for _ in range(n*n):
                fields=stream.readline().split()
                if len(fields)!=7:raise ValueError('Truncated/malformed Wannier matrix')
                this=tuple(map(int,fields[:3]));i,j=map(int,fields[3:5])
                if not 1<=i<=n or not 1<=j<=n or (i,j) in seen or (translation is not None and this!=translation):
                    raise ValueError('Invalid Wannier matrix indices or translations')
                translation=this;seen.add((i,j));matrix[i-1,j-1]=complex(float(fields[5]),float(fields[6]))/(degeneracy[r]*RY_EV)
            translations.append(translation);matrices.append(matrix)
        if stream.read().strip() or not np.isfinite(matrices).all() or len(set(translations))!=nr:
            raise ValueError('Invalid/trailing Wannier data')
    mapping=dict(zip(translations,matrices,strict=True))
    for r,a in mapping.items():
        partner=mapping.get(tuple(-v for v in r))
        if partner is None or np.max(abs(a-partner.conj().T))>1.e-9:
            raise ValueError('Wannier H(R) does not satisfy H(-R)=H(R)^dagger')
    report=dict(status='screening',approximation='imported_wannier90',reduced_spinor_dimension=n,
        measured_speedup=None,numerical_sensitivity=None,
        raw_evidence=[dict(path=str(path.resolve()),sha256=file_hash(path))],
        limitations=['Wannier spinor/orbital/site identity must be supplied in the common gauge.',
                    'H alone does not identify magnetic/SOC operators; MFT is unavailable until they are supplied.'])
    return ReducedHamiltonian(dict(metadata,representation='wannier90'),np.array(translations),
                               {'hamiltonian':np.array(matrices)},report)


def contract_radial_manifolds(realspace, training_state, *, radial_rank=2, window_above_fermi_ev=8.):
    """Material-specific local PAO radial contraction in one constant gauge.

    Learns radial combinations from occupied and low-lying unoccupied spinor
    states, averaging m within each (site,l,n) shell. It keeps the complete
    angular manifold and projects every finite-range operator with the SAME
    local map. This is a variational subspace, not exact elimination of remote
    bands; the response must be measured before using its cheaper predictions.
    """
    from .realspace import RealSpaceHamiltonian
    from .greens import _fermi
    if type(radial_rank) is not int or radial_rank<1 or window_above_fermi_ev<=0:
        raise ValueError('Positive radial rank and energy window required')
    cp,_=cuda();start=time.perf_counter();m=training_state.metadata
    orbitals=m.get('orbitals')
    if not orbitals or any(not all(k in o for k in ('n','l','m','zeta')) for o in orbitals):
        raise ValueError('Native PAO quantum numbers required for radial contraction')
    h=cp.asarray(training_state.h_base+training_state.h_mag+training_state.h_soc);s=cp.asarray(training_state.overlap)
    e,c,_,checks=generalized_eigh(h,s)
    mu,_=_fermi(cp,e,cp.asarray(training_state.weights),m['temperature_ry'],m['electrons'])
    retain=(e<=mu+window_above_fermi_ev/RY_EV).astype(cp.float64)
    covariance=cp.einsum('kib,kjb,kb,k->ij',c,c.conj(),retain,cp.asarray(training_state.weights))
    covariance=.5*(covariance[::2,::2]+covariance[1::2,1::2])
    shells={}
    for i,o in enumerate(orbitals):shells.setdefault((o['site_id'],o['n'],o['l']),{}).setdefault(o['m'],[]).append(i)
    columns=[];mapping=[];discarded=[]
    for (site,n,l),by_m in shells.items():
        ids=[sorted(v,key=lambda i:orbitals[i]['zeta']) for _,v in sorted(by_m.items())]
        if len(ids)!=2*l+1 or len({len(v) for v in ids})!=1:raise ValueError('Incomplete radial/angular PAO shell')
        nz=len(ids[0]);rank=min(radial_rank,nz)
        density=sum(covariance[cp.ix_(cp.asarray(v),cp.asarray(v))] for v in ids).real/len(ids)
        _,rotation=cp.linalg.eigh(density);rotation=cp.asnumpy(rotation[:,-rank:])
        for angular,v in zip(sorted(by_m),ids,strict=True):
            for radial in range(rank):
                column=np.zeros(len(orbitals));column[v]=rotation[:,radial];columns.append(column)
                mapping.append(dict(site_id=site,n=n,l=l,m=angular,zeta=radial+1,
                    parent_indices=v,parent_coefficients=rotation[:,radial].tolist()))
        discarded.append(dict(site_id=site,n=n,l=l,original_radial_count=nz,retained_radial_count=rank))
    transform=np.column_stack(columns)
    if 2*transform.shape[1]<=m['electrons']:raise ValueError('Reduced basis cannot hold the reference electron count')
    t=cp.kron(cp.asarray(transform),cp.eye(2));coeff={name:cp.asnumpy(adjoint(t)@cp.asarray(a)@t) for name,a in realspace.coefficients.items()}
    metadata=dict(realspace.metadata,orbitals=mapping,representation='contracted_native_pao',
        parent_spinor_dimension=training_state.shape[-1],radial_rank=radial_rank,window_above_fermi_ev=window_above_fermi_ev)
    model=RealSpaceHamiltonian(metadata,realspace.translations,coeff)
    report=dict(status='screening',approximation='local_radial_pao_contraction',
        original_spinor_dimension=training_state.shape[-1],reduced_spinor_dimension=2*transform.shape[1],
        basis_mapping=mapping,shells=discarded,training_window_above_fermi_ev=window_above_fermi_ev,
        training_checks=checks,response_error=None,measured_speedup=None,
        performance=dict(construction_seconds=time.perf_counter()-start),numerical_sensitivity=None,raw_evidence=[],
        limitations=['Contraction is variational; excluded radial flexibility can change small anisotropy differences.',
            'One common local real map projects H, S, exchange and SOC; angular manifolds remain complete.',
            'Retain the full PAO model if measured magnetic response changes materially.'])
    return model,report
