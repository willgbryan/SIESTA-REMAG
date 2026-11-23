"""Localized 4f Fock-space multiplets: GPU assembly and diagonalization.

Atomic angular coefficients are small analytic input data, not a CPU solver.
The Coulomb convention is H=1/2 sum U_pqrs a†p a†q a_s a_r with normalized
spherical harmonics C^k_q. No DFT-derived parameters or double counting are
inferred. Supplied crystal fields remain separate from parameter extraction.
"""
from functools import lru_cache
from math import factorial, sqrt
import time
import numpy as np
from ..angular_energy import unit_directions
from .gpu import cuda, adjoint
from .operators import hermitian

KB_EV = 8.617333262145e-5


def _three_j(j1,j2,j3,m1,m2,m3):
    if m1+m2+m3 or any(abs(m)>j for m,j in ((m1,j1),(m2,j2),(m3,j3))) or not abs(j1-j2)<=j3<=j1+j2:
        return 0.
    prefactor=(-1.)**(j1-j2-m3)*sqrt(
        factorial(j1+j2-j3)*factorial(j1-j2+j3)*factorial(-j1+j2+j3)/factorial(j1+j2+j3+1)*
        np.prod([factorial(j+m)*factorial(j-m) for j,m in ((j1,m1),(j2,m2),(j3,m3))],dtype=float))
    total=0.
    for t in range(max(0,j2-j3-m1,j1-j3+m2),min(j1+j2-j3,j1-m1,j2+m2)+1):
        denominator=factorial(t)*factorial(j1+j2-j3-t)*factorial(j1-m1-t)*factorial(j2+m2-t)*factorial(j3-j2+m1+t)*factorial(j3-j1-m2+t)
        total+=(-1.)**t/denominator
    return prefactor*total


@lru_cache(maxsize=32)
def spherical_tensor(rank,q):
    if rank not in (0,2,4,6) or abs(q)>rank:raise ValueError('4f tensors require k=0,2,4,6 and |q|<=k')
    a=np.zeros((7,7),complex)
    for i,m in enumerate(range(-3,4)):
        for j,mp in enumerate(range(-3,4)):
            a[i,j]=(-1.)**m*7*_three_j(3,rank,3,0,0,0)*_three_j(3,rank,3,-m,q,mp)
    a.setflags(write=False)
    return a


def angular_operators():
    plus=np.zeros((7,7),complex)
    for i,m in enumerate(range(-3,3)):plus[i+1,i]=sqrt(12-m*(m+1))
    orbital=[(plus+plus.T)*.5,(plus-plus.T)/2j,np.diag(np.arange(-3,4))]
    spin=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])*.5
    l=np.array([np.kron(a,np.eye(2)) for a in orbital]);s=np.array([np.kron(np.eye(7),a) for a in spin])
    return l,s,sum(a@b for a,b in zip(l,s,strict=True))


def crystal_field(parameters):
    if parameters.get('crystal_field_convention')!='Wybourne_Ckq':
        raise ValueError('Supply explicit Wybourne_Ckq coefficients in eV; Stevens coefficients are not interchangeable')
    terms=parameters.get('crystal_field_ev')
    if terms is None:raise ValueError('Missing crystal-field parameters (an explicit empty list denotes zero CF)')
    cf=np.zeros((7,7),complex)
    seen=set()
    for term in terms:
        k,q=term['k'],term['q']
        if type(k) is not int or type(q) is not int or k not in (2,4,6) or (k,q) in seen:
            raise ValueError('Distinct k=2,4,6 crystal-field coefficients required')
        seen.add((k,q));value=complex(term['real_ev'],term.get('imag_ev',0.))
        cf+=value*spherical_tensor(k,q)
    return hermitian(cf,'4f crystal field',1.e-12)


def coulomb_terms(slater):
    """Antisymmetrized pair coefficients for p<q, r<s in 14 spin orbitals."""
    f={int(k):float(v) for k,v in slater.items()}
    if set(f)!={0,2,4,6} or any(not np.isfinite(v) or v<0 for v in f.values()):
        raise ValueError('Finite nonnegative F0,F2,F4,F6 in eV are required')
    orbital=np.zeros((7,7,7,7),complex)
    for k,value in f.items():
        for q in range(-k,k+1):
            c=spherical_tensor(k,q)
            orbital+=value*np.einsum('pr,qs->pqrs',c,c.conj().T)
    def u(p,q,r,s):
        return orbital[p//2,q//2,r//2,s//2] if p%2==r%2 and q%2==s%2 else 0.
    terms=[]
    for p in range(14):
        for q in range(p+1,14):
            for r in range(14):
                for s in range(r+1,14):
                    value=u(p,q,r,s)-u(p,q,s,r)
                    if abs(value)>1.e-14:terms.append((p,q,r,s,value))
    return terms


_ASSEMBLE = r'''
struct Z {double x,y;};
__device__ int apply(int &mask,int orbital,bool create,int &sign) {
    int present=(mask>>orbital)&1;
    if(present==int(create))return 0;
    if(__popc(unsigned(mask)&((1u<<orbital)-1))) {
        if(__popc(unsigned(mask)&((1u<<orbital)-1))&1)sign=-sign;
    }
    mask^=(1<<orbital);return 1;
}
extern "C" __global__ void assemble(int dim,int nt,const int* basis,const int* lookup,
    const int* terms,const Z* coefficients,int* rows,int* columns,Z* values) {
    for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<dim*nt;i+=blockDim.x*gridDim.x) {
        int col=i%dim,term=i/dim,mask=basis[col],sign=1;
        int p=terms[4*term],q=terms[4*term+1],r=terms[4*term+2],s=terms[4*term+3];
        bool valid=apply(mask,r,false,sign);
        if(valid&&s>=0)valid=apply(mask,s,false,sign);
        if(valid&&q>=0)valid=apply(mask,q,true,sign);
        if(valid)valid=apply(mask,p,true,sign);
        rows[i]=valid?lookup[mask]:0;columns[i]=valid?col:0;
        Z v=coefficients[term];values[i]=valid?Z{sign*v.x,sign*v.y}:Z{0.,0.};
    }
}
'''


class MultipletModel:
    """Cache occupancy-dependent operators and atomic terms for environment sweeps."""
    def __init__(self,occupancy,slater_ev,soc_ev):
        if type(occupancy) is not int or not 0<=occupancy<=14:raise ValueError('Integer 4f occupancy 0..14 required')
        if not np.isfinite(soc_ev) or soc_ev<0:raise ValueError('Finite nonnegative atomic SOC parameter required')
        self.cp,self.cupyx=cuda();cp=self.cp
        self.parameters=(occupancy,{int(k):float(v) for k,v in slater_ev.items()},float(soc_ev))
        from cupyx.scipy import sparse
        self.sparse=sparse
        states=np.array([v for v in range(1<<14) if v.bit_count()==occupancy],dtype=np.int32)
        lookup=np.full(1<<14,-1,dtype=np.int32);lookup[states]=np.arange(len(states),dtype=np.int32)
        self.basis=cp.asarray(states);self.lookup=cp.asarray(lookup);self.dimension=len(states)
        self.kernel=cp.RawKernel(_ASSEMBLE,'assemble',options=('--std=c++17',))
        l,s,soc=angular_operators()
        self.spin=[self.one_body(a) for a in s];self.orbital=[self.one_body(a) for a in l]
        self.atomic=self.assemble(coulomb_terms(slater_ev))+self.one_body(soc_ev*soc)

    def assemble(self,terms):
        cp=self.cp;n=self.dimension
        if not terms:return self.sparse.csr_matrix((n,n),dtype=cp.complex128)
        indices=cp.asarray(np.array([t[:4] for t in terms],dtype=np.int32))
        coefficients=cp.asarray(np.array([t[4] for t in terms],dtype=np.complex128))
        count=n*len(terms)
        rows,cols=cp.empty(count,dtype=cp.int32),cp.empty(count,dtype=cp.int32)
        values=cp.empty(count,dtype=cp.complex128)
        self.kernel((min(1024,(count+255)//256),),(256,),
                    (np.int32(n),np.int32(len(terms)),self.basis,self.lookup,indices,coefficients,rows,cols,values))
        result=self.sparse.coo_matrix((values,(rows,cols)),shape=(n,n)).tocsr()
        result.sum_duplicates();result.eliminate_zeros()
        return result

    def one_body(self,matrix):
        return self.assemble([(int(p),-1,int(r),-1,matrix[p,r]) for p,r in np.argwhere(abs(matrix)>1.e-14)])

    def solve(self,cf,exchange_field_ev,directions,*,temperature_k=0.,levels=64):
        cp=self.cp
        if not np.isfinite(exchange_field_ev) or not np.isfinite(temperature_k) or temperature_k<0:
            raise ValueError('Finite exchange field and nonnegative temperature required')
        if type(levels) is not int or levels<1:raise ValueError('Positive number of multiplet levels required')
        axes=unit_directions(directions);base=self.atomic+self.one_body(np.kron(cf,np.eye(2)))
        points=[]
        for axis in axes:
            h=base.copy()
            for a in range(3):h=h-2*exchange_field_ev*float(axis[a])*self.spin[a]
            defect=h-h.conj().T
            if defect.nnz and float(cp.max(cp.abs(defect.data)).get())>1.e-10:raise RuntimeError('4f Hamiltonian is not Hermitian')
            full=self.dimension<=max(levels+2,32)
            with self.cupyx.errstate(linalg='raise'):
                if full:e,v=cp.linalg.eigh(h.toarray())
                else:
                    from cupyx.scipy.sparse.linalg import eigsh
                    e,v=eigsh(h,k=min(levels,self.dimension-3),which='SA',tol=1.e-11,maxiter=10000)
                    order=cp.argsort(e);e,v=e[order],v[:,order]
            residual=cp.linalg.norm(h@v-v*e[None,:],axis=0)
            max_residual=float(cp.max(residual).get())
            if not bool(cp.all(cp.isfinite(e)).get()) or not np.isfinite(max_residual) or max_residual>1.e-8:
                raise RuntimeError('4f eigenpairs failed finite/residual checks')
            gap=e-e[0];t=KB_EV*temperature_k
            if t:
                probability=cp.exp(-gap/t);partition=float(cp.sum(probability).get());probability/=partition
                tail=0. if full else (self.dimension-len(e))*float(cp.exp(-gap[-1]/t).get())/partition
                free=float(e[0].get())-t*np.log(partition) if tail<1.e-4 else None
            else:
                ground=gap<1.e-8;probability=ground/cp.sum(ground)
                tail=0. if full or float(gap[-1].get())>1.e-8 else None
                free=float(e[0].get()) if tail is not None else None
            def moments(ops):return [float(cp.sum(probability*cp.sum(v.conj()*(op@v),axis=0).real).get()) for op in ops]
            spin=moments(self.spin);orbital=moments(self.orbital)
            def square(ops):return float(cp.sum(probability*sum(cp.sum(cp.abs(op@v)**2,axis=0) for op in ops)).get())
            points.append(dict(direction=axis.tolist(),free_energy_ev=free,
                excitation_energies_ev=cp.asnumpy(gap).tolist(),spin_expectation=spin,orbital_expectation=orbital,
                magnetic_moment_muB=(-np.array(orbital)-2*np.array(spin)).tolist(),
                S_squared=square(self.spin),L_squared=square(self.orbital),
                J_squared=square([l+s for l,s in zip(self.orbital,self.spin,strict=True)]),
                partition_tail_relative_bound=tail,full_spectrum=full,max_residual_ev=max_residual))
        for point in points:
            point['delta_free_ev']=None if point['free_energy_ev'] is None or points[0]['free_energy_ev'] is None else point['free_energy_ev']-points[0]['free_energy_ev']
        return points


def solve_4f_multiplet(parameters, directions=((0,0,1),(1,0,0)), *, temperature_k=0.,levels=64,model=None):
    """Standalone local model, also usable before DFT parameter extraction exists."""
    started=time.perf_counter()
    required=('species','ionic_valence','occupancy','slater_ev','soc_ev','crystal_field_convention','crystal_field_ev','exchange_field_ev','double_counting')
    missing=[k for k in required if parameters.get(k) is None]
    if missing:return dict(status='incomplete',missing_parameters=missing,points=None,summary=None,
                           limitations=['Local parameters are not inferred from incomplete DFT evidence.'],raw_evidence=[])
    if parameters['double_counting'] not in ('separate_open_core_contribution','local_model_only'):
        raise ValueError('Declare separate_open_core_contribution or local_model_only; no automatic addition to DFT energies')
    cf=crystal_field(parameters)
    if model is None:model=MultipletModel(parameters['occupancy'],parameters['slater_ev'],parameters['soc_ev'])
    if model.parameters!=(parameters['occupancy'],{int(k):float(v) for k,v in parameters['slater_ev'].items()},float(parameters['soc_ev'])):
        raise ValueError('Cached atomic model parameters differ from the requested ion')
    points=model.solve(cf,parameters['exchange_field_ev'],directions,temperature_k=temperature_k,levels=levels)
    complete=all(p['free_energy_ev'] is not None for p in points)
    minimum=min(points,key=lambda p:float('inf') if p['free_energy_ev'] is None else p['free_energy_ev'])
    return dict(status='screening' if complete else 'incomplete',method='gpu_4f_fock_space',
        approximation='localized_atomic_multiplet_supplied_crystal_field',parameters=parameters,
        basis='m=-3..3, up/down; fixed-number Slater determinants',dimension=model.dimension,
        temperature_k=temperature_k,points=points,
        summary=dict(preferred_sampled_direction=minimum['direction'] if complete else None,
                     local_anisotropy_ev=max(p['free_energy_ev'] for p in points)-minimum['free_energy_ev'] if complete else None),
        numerical_sensitivity=None,raw_evidence=[],performance=dict(total_wall_seconds=time.perf_counter()-started),
        limitations=['Local 4f response is not the full magnet temperature-dependent anisotropy.',
            'Parameters and ionic occupancy are supplied; general first-principles crystal-field extraction is unavailable.',
            'Magnetic moment uses the electron sign -muB(L+2S); the exchange coupling is -2 B_ex dot S.',
            'Truncated thermal sums are withheld when their omitted-state bound is insufficient; increase levels.',
            'Do not add this contribution to a DFT calculation containing the same explicit 4f electrons.'])


def calculate_4f_multiplets(state, sites='localized_4f', *, parameters=None, **kwargs):
    selected=[s['site_id'] for s in state.metadata.get('sites',[]) if s.get('localized_4f')] if sites=='localized_4f' else list(sites)
    if not selected:return dict(status='incomplete',sites={},summary=None,missing_parameters=['localized_4f site identity'],raw_evidence=[])
    parameters=parameters or {}
    results={site:solve_4f_multiplet(parameters.get(site,{}),**kwargs) for site in selected}
    return dict(status='screening' if all(r['status']=='screening' for r in results.values()) else 'incomplete',
        sites=results,summary={k:r.get('summary') for k,r in results.items()},
        approximation='localized_4f_site_models',numerical_sensitivity=None,raw_evidence=[],
        limitations=['Site contributions are kept separate and are not automatically added to itinerant DFT energies.'])
