"""GPU Green blocks and finite-smearing spectral magnetic response.

The response integrates the two-Green-function bubble analytically using
spectral divided differences. It avoids an energy-quadrature parameter. This
is the grand-canonical frozen-potential Hessian of a declared onsite exchange
model; nonlocal magnetic matrix elements are not assigned to local spins.
"""
from itertools import product
import time
import numpy as np
from .gpu import cuda, adjoint, cell_gauge, generalized_eigh
from .operators import RY_EV, rotation_between


class GreensFunction:
    def __init__(self, h, overlap, *, eigenpairs=None):
        self.cp,self.cupyx=cuda()
        self.h,self.s=self.cp.asarray(h),self.cp.asarray(overlap)
        self.eigenpairs=eigenpairs
        self.checks=None

    def spectral(self):
        if self.eigenpairs is None:
            e,c,_,self.checks=generalized_eigh(self.h,self.s)
            self.eigenpairs=e,c
        return self.eigenpairs

    def block(self,z,rows,columns,*,method='spectral'):
        """G_rows,columns(k,z), contravariant PAO Green matrix, stays on GPU."""
        if not np.isfinite(z) or complex(z).imag==0:
            raise ValueError('A finite complex energy off the real axis is required')
        if method=='spectral':
            e,c=self.spectral()
            return (c[:,rows,:]/(z-e)[:,None,:])@adjoint(c[:,columns,:])
        if method!='solve':raise ValueError('Select spectral or solve')
        cp=self.cp
        rhs=cp.broadcast_to(cp.eye(self.h.shape[-1],dtype=cp.complex128)[:,columns],
                             (len(self.h),self.h.shape[-1],len(columns)))
        with self.cupyx.errstate(linalg='raise'):
            solution=cp.linalg.solve(z*self.s-self.h,rhs)
        if not bool(cp.all(cp.isfinite(solution)).get()):raise RuntimeError('Complex Green solve failed')
        residual=cp.linalg.norm((z*self.s-self.h)@solution-rhs)/cp.linalg.norm(rhs)
        if float(residual.get())>1.e-8:raise RuntimeError('Complex Green solve residual failed')
        return solution[:,rows,:]


def _fermi(cp,e,weights,temperature,electrons):
    lower=float(cp.min(e).get())-100*temperature;upper=float(cp.max(e).get())+100*temperature
    for _ in range(150):
        mu=.5*(lower+upper)
        f=1/(1+cp.exp(cp.clip((e-mu)/temperature,-100,100)))
        charge=float(cp.sum(weights[:,None]*f).get())
        if abs(charge-electrons)<=1.e-10:return mu,f
        if charge>electrons:upper=mu
        else:lower=mu
    raise RuntimeError('Green response fixed-count chemical potential failed')


def _pairs(metadata,radius):
    sites=metadata.get('sites');geometry=metadata.get('geometry')
    if not sites or not geometry:raise ValueError('Exchange requires atomic sites and evaluated geometry')
    cell=np.asarray(geometry['cell_angstrom'],float)
    if cell.shape!=(3,3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell))<1.e-12:
        raise ValueError('Invalid lattice for exchange distances')
    positions=np.array([s['position_angstrom'] for s in sites])
    # Reciprocal bound remains valid for skew cells and unwrapped positions.
    spans=np.max(np.abs((positions[:,None,:]-positions[None,:,:])@np.linalg.inv(cell)),axis=(0,1))
    bounds=np.ceil(radius*np.linalg.norm(np.linalg.inv(cell),axis=0)+spans).astype(int)
    if np.prod(2*bounds+1)*len(sites)**2>1_000_000:
        raise ValueError('Requested interaction radius generates too many pairs; reduce it')
    pairs=[]
    for i,a in enumerate(sites):
        for j,b in enumerate(sites):
            for r in product(*(range(-x,x+1) for x in bounds)):
                if (i,j,*r)>=(j,i,*[-x for x in r]):continue
                displacement=positions[j]+np.array(r)@cell-positions[i]
                distance=float(np.linalg.norm(displacement))
                if 1.e-9<distance<=radius:
                    pairs.append(dict(i=i,j=j,site_i=a['site_id'],site_j=b['site_id'],translation=list(r),
                                      distance_angstrom=distance,displacement_angstrom=displacement.tolist()))
    return pairs


def calculate_exchange(state, radius_angstrom=8., *, relativistic=False, directions=None,
                       method='realspace', poles=64, site_roles=None):
    """Local rigid-field spectral Green response, E=-sum_pairs e_i.J_ij.e_j.

    SOC tensors use three frozen references (x/y/z), retaining only transverse
    derivatives from each. This is an explicit frozen-reference reconstruction,
    not a general noncollinear formalism or an independently SCF tensor fit.
    """
    if method=='realspace':
        return _realspace_exchange(state,radius_angstrom,relativistic=relativistic,
                                   directions=directions,poles=poles,site_roles=site_roles)
    if method!='spectral_pairs':raise ValueError('Choose realspace or spectral_pairs')
    if not np.isfinite(radius_angstrom) or radius_angstrom<=0:raise ValueError('Positive radius required')
    if directions is not None:raise ValueError('Custom/noncollinear tensor reference sets are not supported')
    cp,_=cuda();started=time.perf_counter();m=state.metadata
    orbitals=m.get('orbitals')
    if not orbitals:raise ValueError('Exchange requires the spatial orbital/site mapping')
    banks,kpoints=cell_gauge(state);base,mag,soc,s=banks
    weights=cp.asarray(state.weights);nk,n,_=base.shape
    sites=m['sites'];indices=[];fields=[]
    roles=site_roles or {v['site_id']:v.get('magnetic_role','unassigned') for v in sites}
    if any(roles.get(v['site_id']) not in ('local_moment','induced','unassigned','nonmagnetic') for v in sites):
        raise ValueError('Declare site roles as local_moment, induced, unassigned or nonmagnetic')
    direction=np.asarray(m['reference_direction'])
    # Onsite exchange from the explicit native decomposition, excluding SOC.
    p0=cp.sum(weights[:,None,None]*mag,axis=0)
    vector=cp.stack([.5*(p0[::2,1::2]+p0[1::2,::2]),.5j*(p0[::2,1::2]-p0[1::2,::2]),.5*(p0[::2,::2]-p0[1::2,1::2])])
    p=cp.einsum('a,aij->ij',cp.asarray(direction),vector)
    for site in sites:
        orbital=np.array([i for i,o in enumerate(orbitals) if o['site_id']==site['site_id']],dtype=int)
        if not len(orbital):raise ValueError('A site has no orbital mapping')
        indices.append(np.ravel(np.column_stack((2*orbital,2*orbital+1))))
        fields.append(p[cp.ix_(cp.asarray(orbital),cp.asarray(orbital))])
    pairs=_pairs(m,radius_angstrom)
    if not pairs:raise ValueError('No site pairs within radius')
    pauli=cp.asarray([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],dtype=cp.complex128)
    axes=np.eye(3) if relativistic else np.array([[0.,0.,1.]])
    responses=cp.zeros((len(pairs),3,3),dtype=cp.float64);counts=np.zeros((3,3),int);checks=[]
    for axis in axes:
        q=rotation_between(direction,axis);w,x,y,z=q
        u=cp.asarray([[w-1j*z,-y-1j*x],[y-1j*x,w+1j*z]])
        u=cp.kron(cp.eye(n//2),u)
        h=base+u@mag@adjoint(u)+(soc if relativistic else 0)
        e,c,_,check=generalized_eigh(h,s);checks.append(check)
        mu,f=_fermi(cp,e,weights,m['temperature_ry'],m['electrons'])
        active=[a for a in range(3) if axis[a]==0]
        for a in active:
            for b in active:counts[a,b]+=1
        operators=[[cp.kron(field,pauli[a]) for a in active] for field in fields]
        # One (k,k') block at a time bounds the response workspace. All band,
        # orbital and site-pair contractions execute on GPU.
        for k in range(nk):
            for kp in range(nk):
                delta=e[k,:,None]-e[kp,None,:]
                df=f[k,:,None]-f[kp,None,:]
                average=.5*(f[k,:,None]+f[kp,None,:])
                divided=cp.where(cp.abs(delta)>1.e-10,df/cp.where(cp.abs(delta)>1.e-10,delta,1),
                                 -average*(1-average)/m['temperature_ry'])
                local=[]
                for ids,ops in zip(indices,operators,strict=True):
                    ca,cb=c[k,ids,:],c[kp,ids,:]
                    local.append([adjoint(ca)@op@cb for op in ops])
                for pair_index,pair in enumerate(pairs):
                    phase=cp.exp(2j*np.pi*cp.dot(kpoints[k]-kpoints[kp],cp.asarray(pair['translation'])))
                    for ai,a in enumerate(active):
                        for bi,b in enumerate(active):
                            value=cp.sum(divided*local[pair['i']][ai]*local[pair['j']][bi].conj())
                            responses[pair_index,a,b]-=(weights[k]*weights[kp]*phase*value).real
    if not relativistic:
        isotropic=.5*(responses[:,0,0]+responses[:,1,1])
        responses=isotropic[:,None,None]*cp.eye(3)
    else:
        responses/=cp.asarray(counts)[None,:,:]
    tensors=cp.asnumpy(responses)*RY_EV
    if not np.isfinite(tensors).all():raise RuntimeError('Nonfinite Green response tensor')
    for pair,tensor in zip(pairs,tensors,strict=True):
        iso=float(np.trace(tensor)/3);sym=.5*(tensor+tensor.T)-iso*np.eye(3)
        d=.5*np.array([tensor[1,2]-tensor[2,1],tensor[2,0]-tensor[0,2],tensor[0,1]-tensor[1,0]])
        pair.update(exchange_tensor_ev=tensor.tolist(),isotropic_ev=iso,symmetric_anisotropic_ev=sym.tolist(),dmi_ev=d.tolist())
        a,b=roles[pair['site_i']],roles[pair['site_j']]
        pair.update(site_roles=[a,b],independent_spin_pair=(a==b=='local_moment'),
                    interpretation='spin_coupling' if a==b=='local_moment' else 'electronic_response_channel')
    ordered=sorted(pairs,key=lambda p:abs(p['isotropic_ev']),reverse=True)
    return dict(status='screening',method='gpu_spectral_greens_hessian',approximation='onsite_rigid_exchange_frozen_potential',
        convention='E=-sum_unique_pairs e_i^T J_ij e_j; unit spins; positive J is ferromagnetic',
        pairs=pairs,site_roles=roles,network=dict(pair_count=len(pairs),
            independent_spin_pair_count=sum(p['independent_spin_pair'] for p in pairs),
            dominant_ferromagnetic=[p for p in ordered if p['independent_spin_pair'] and p['isotropic_ev']>0][:8],
            dominant_antiferromagnetic=[p for p in ordered if p['independent_spin_pair'] and p['isotropic_ev']<0][:8],
            induced_or_unassigned_response_channels=[p for p in ordered if not p['independent_spin_pair']][:8],
            competing_signs=any(p['independent_spin_pair'] and p['isotropic_ev']>0 for p in pairs)
                and any(p['independent_spin_pair'] and p['isotropic_ev']<0 for p in pairs)),
        local_single_ion_anisotropy=None,checks=checks,numerical_sensitivity=None,raw_evidence=[],
        performance=dict(total_wall_seconds=time.perf_counter()-started),
        limitations=['Onsite projection defines the local exchange perturbation; nonlocal exchange-potential terms are omitted.',
            'Induced and unassigned moments are electronic response channels, not independently orientable local spins.',
            'Fixed-chemical-potential Hessian at the reference electron count and electronic temperature.',
            'Finite k meshes alias real-space interactions; radius does not establish real-space convergence.',
            'SOC tensors combine transverse responses of three frozen reference directions; general noncollinear response is unsupported.',
            'Exchange paths are representation-dependent diagnostics; competing signs do not by themselves prove frustration.'])


def _fermi_poles(cp, count):
    """Ozaki continued fraction (PRB 75, 035123): GPU tridiagonal eigenproblem."""
    j=cp.arange(1,2*count,dtype=cp.float64)
    off=1/(2*cp.sqrt((2*j-1)*(2*j+1)))
    values,vectors=cp.linalg.eigh(cp.diag(off,1)+cp.diag(off,-1))
    positive=values>0;lam=values[positive]
    return 1/lam,-vectors[0,positive]**2/(4*lam**2)


def _realspace_exchange(state,radius,*,relativistic,directions,poles,site_roles):
    """Fourier-transform G once per pole; no (k,k') pair loop.

    Retains the spectral implementation's finite-T onsite frozen-field Hessian.
    Induced/unassigned sites remain response channels, not independent spins.
    """
    if directions is not None:raise ValueError('Custom noncollinear tensor references are unsupported')
    if not np.isfinite(radius) or radius<=0 or type(poles) is not int or not 8<=poles<=512:
        raise ValueError('Positive radius and 8..512 initial poles required')
    cp,_=cuda();start=time.perf_counter();m=state.metadata
    banks,kpoints=cell_gauge(state);base,mag,soc,s=banks;nk,n,_=base.shape
    sites=m['sites'];orbitals=m['orbitals'];weights=cp.asarray(state.weights)
    roles=site_roles or {v['site_id']:v.get('magnetic_role','unassigned') for v in sites}
    if any(roles.get(v['site_id']) not in ('local_moment','induced','unassigned','nonmagnetic') for v in sites):
        raise ValueError('Declare site roles as local_moment, induced, unassigned or nonmagnetic')
    pairs=_pairs(m,radius)
    if not pairs:raise ValueError('No site pairs within radius')
    indices=[np.ravel(np.column_stack((2*np.array([i for i,o in enumerate(orbitals) if o['site_id']==v['site_id']]),
                                      2*np.array([i for i,o in enumerate(orbitals) if o['site_id']==v['site_id']])+1))) for v in sites]
    direction=np.asarray(m['reference_direction']);p0=cp.sum(weights[:,None,None]*mag,axis=0)
    vec=cp.stack([.5*(p0[::2,1::2]+p0[1::2,::2]),.5j*(p0[::2,1::2]-p0[1::2,::2]),.5*(p0[::2,::2]-p0[1::2,1::2])])
    field=cp.einsum('a,aij->ij',cp.asarray(direction),vec)
    pauli=cp.asarray([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],dtype=cp.complex128)
    fields=[field[cp.ix_(cp.asarray(ids[::2]//2),cp.asarray(ids[::2]//2))] for ids in indices]
    axes=np.eye(3) if relativistic else np.array([[0.,0.,1.]])
    tensors=cp.zeros((len(pairs),3,3));counts=np.zeros((3,3),int);checks=[];pole_reports=[]
    translations=np.unique(np.array([p['translation'] for p in pairs]+[[-x for x in p['translation']] for p in pairs]),axis=0)
    rmap={tuple(r):i for i,r in enumerate(translations)}
    # G_ij(R)=sum_k w_k exp(-ik.R) G_ij(k) in the cell-positive convention.
    phase=cp.exp(-2j*np.pi*cp.asarray(translations)@kpoints.T)*weights[None,:]
    for axis in axes:
        w,x,y,z=rotation_between(direction,axis);u=cp.kron(cp.eye(n//2),cp.asarray([[w-1j*z,-y-1j*x],[y-1j*x,w+1j*z]]))
        h=base+u@mag@adjoint(u)+(soc if relativistic else 0)
        e,c,_,check=generalized_eigh(h,s);checks.append(check)
        temperature=m['temperature_ry'];mu,f=_fermi(cp,e,weights,temperature,m['electrons'])
        used=poles
        while True:
            zp,rp=_fermi_poles(cp,used);xx=(e-mu)/temperature
            approx=.5+cp.sum(2*rp[:,None,None]*xx[None]/(xx[None]**2+zp[:,None,None]**2),axis=0)
            error=float(cp.max(cp.abs(approx-f)).get())
            if error<1.e-9:break
            if used>=512:raise RuntimeError('Fermi pole approximation insufficient for the occupied spectrum')
            used=min(2*used,512)
        pole_reports.append(dict(poles=used,fermi_function_max_error=error,
                                 chemical_potential_ev=float(mu)*RY_EV,
                                 temperature_ev=temperature*RY_EV))
        active=[a for a in range(3) if axis[a]==0]
        operators=[[cp.kron(v,pauli[a]) for a in active] for v in fields]
        for a in active:
            for b in active:counts[a,b]+=1
        for pole,residue in zip(cp.asnumpy(zp),cp.asnumpy(rp),strict=True):
            g=(c/(mu+1j*temperature*pole-e)[:,None,:])@adjoint(c)
            real=(phase@g.reshape(nk,-1)).reshape(len(translations),n,n)
            for pi,pair in enumerate(pairs):
                i,j=pair['i'],pair['j'];r=tuple(pair['translation']);minus=tuple(-v for v in r)
                gij=real[rmap[r]][cp.ix_(cp.asarray(indices[i]),cp.asarray(indices[j]))]
                gji=real[rmap[minus]][cp.ix_(cp.asarray(indices[j]),cp.asarray(indices[i]))]
                for ai,a in enumerate(active):
                    for bi,b in enumerate(active):
                        tensors[pi,a,b]+=2*temperature*residue*cp.trace(operators[i][ai]@gij@operators[j][bi]@gji).real
        del e,c,h
    if relativistic:tensors/=cp.asarray(counts)[None]
    else:tensors=.5*(tensors[:,0,0]+tensors[:,1,1])[:,None,None]*cp.eye(3)
    values=cp.asnumpy(tensors)*RY_EV
    if not np.isfinite(values).all():raise RuntimeError('Nonfinite real-space Green response')
    for pair,tensor in zip(pairs,values,strict=True):
        iso=float(np.trace(tensor)/3);a,b=roles[pair['site_i']],roles[pair['site_j']]
        pair.update(exchange_tensor_ev=tensor.tolist(),isotropic_ev=iso,
            symmetric_anisotropic_ev=(.5*(tensor+tensor.T)-iso*np.eye(3)).tolist(),
            dmi_ev=(.5*np.array([tensor[1,2]-tensor[2,1],tensor[2,0]-tensor[0,2],tensor[0,1]-tensor[1,0]])).tolist(),
            site_roles=[a,b],independent_spin_pair=(a==b=='local_moment'),
            interpretation='spin_coupling' if a==b=='local_moment' else 'electronic_response_channel')
    ranked=sorted(pairs,key=lambda p:abs(p['isotropic_ev']),reverse=True)
    return dict(status='screening',method='gpu_realspace_pole_greens_hessian',approximation='onsite_rigid_exchange_frozen_potential',
        convention='E=-sum_unique_pairs e_i^T J_ij e_j; unit spins; positive J is ferromagnetic',
        pairs=pairs,site_roles=roles,network=dict(pair_count=len(pairs),
            independent_spin_pair_count=sum(p['independent_spin_pair'] for p in pairs),
            dominant_ferromagnetic=[p for p in ranked if p['independent_spin_pair'] and p['isotropic_ev']>0][:8],
            dominant_antiferromagnetic=[p for p in ranked if p['independent_spin_pair'] and p['isotropic_ev']<0][:8],
            induced_or_unassigned_response_channels=[p for p in ranked if not p['independent_spin_pair']][:8],
            competing_signs=any(p['independent_spin_pair'] and p['isotropic_ev']>0 for p in pairs)
                and any(p['independent_spin_pair'] and p['isotropic_ev']<0 for p in pairs)),
        checks=checks,pole_approximation=pole_reports,local_single_ion_anisotropy=None,numerical_sensitivity=None,raw_evidence=[],
        performance=dict(total_wall_seconds=time.perf_counter()-start,kpoints=nk,algorithm='O(nk * poles), excluding eigensolution and orbital contractions'),
        limitations=['Onsite projected exchange only; finite k meshes alias real-space interactions.',
            'Induced and unassigned moments are electronic response channels, not independently orientable local spins.',
            'Fixed-chemical-potential Hessian; SOC tensors use three frozen global references, not general noncollinear response.',
            'Interaction strengths do not alone establish ordering temperature or a causal structural mechanism.'])
