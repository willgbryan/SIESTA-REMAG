"""Finite-range native generator interpolation, independent of the SCF k mesh."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .gpu import cuda
from .operators import BOHR_ANGSTROM
from .state import MagneticState, file_hash


def uniform_mesh(mesh):
    mesh=np.asarray(mesh)
    if mesh.shape!=(3,) or not np.issubdtype(mesh.dtype,np.integer) or np.any(mesh<1):
        raise ValueError('Three positive integer mesh dimensions required')
    # Match the source's SIESTA k_displ=0 grid, including Gamma for even n.
    return np.stack(np.meshgrid(*[(np.arange(n)-n//2)/n for n in mesh],indexing='ij'),axis=-1).reshape(-1,3)


@dataclass
class RealSpaceHamiltonian:
    metadata: dict
    translations: np.ndarray
    coefficients: dict

    def interpolate(self,kpoints, *, batch_size=256):
        cp,_=cuda();k=np.asarray(kpoints,float)
        if k.ndim!=2 or k.shape[1]!=3 or not np.isfinite(k).all():raise ValueError('Finite fractional k points required')
        r=cp.asarray(self.translations);coeff={key:cp.asarray(value) for key,value in self.coefficients.items()}
        for start in range(0,len(k),batch_size):
            phase=cp.exp(2j*np.pi*cp.asarray(k[start:start+batch_size])@r.T)
            banks={name:(phase@a.reshape(len(r),-1)).reshape(len(phase),*a.shape[1:]) for name,a in coeff.items()}
            # Match the native lower-triangle eigensolver convention, including
            # its small independently accumulated upper/lower differences.
            banks={name:cp.tril(a)+cp.tril(a,-1).swapaxes(-1,-2).conj() for name,a in banks.items()}
            for a in banks.values():
                ii=cp.arange(a.shape[-1]);a[:,ii,ii]=a[:,ii,ii].real
            yield start,banks

    def sample(self,mesh, *, batch_size=256):
        cp,_=cuda();k=uniform_mesh(mesh);values={name:[] for name in self.coefficients}
        for _,banks in self.interpolate(k,batch_size=batch_size):
            for name,a in banks.items():values[name].append(cp.asnumpy(a))
        m=dict(self.metadata,mesh_complete=True,kpoint_convention='fractional_reciprocal',
               bloch_gauge='cell_periodic_positive',interpolation='native_finite_range_generator',
               settings={**self.metadata.get('settings',{}),'kmesh':list(map(int,mesh))})
        return MagneticState(m,**{key:np.concatenate(a) for key,a in values.items()},
                             kpoints=k,weights=np.full(len(k),1/len(k)))


def load_realspace(directory, metadata):
    """Load native active sparse entries; preserve complex spin blocks and SOC."""
    root=Path(directory);manifest=root/'remag-magnetic-realspace.json'
    info=json.loads(manifest.read_text());path=root/'remag-magnetic-realspace.bin'
    if (info.get('complete') is not True or info.get('schema')!=1 or
            info.get('reference')!='Final SCF generator Hamiltonian' or
            info.get('layout')!='rows,columns,xij,generator,offsite,overlap'):
        raise ValueError('Complete native real-space generator required')
    count,n=info['entries'],info['spatial_orbitals']
    if count<1 or n<1 or path.stat().st_size!=168*count:raise ValueError('Invalid real-space export size')
    with path.open('rb') as stream:
        rows=np.fromfile(stream,'<i4',count);cols=np.fromfile(stream,'<i4',count)
        xij=np.fromfile(stream,'<f8',3*count).reshape(-1,3)
        h=np.fromfile(stream,'<f8',8*count).reshape(-1,8)
        off=np.fromfile(stream,'<c16',4*count).reshape(-1,4)
        overlap=np.fromfile(stream,'<f8',count)
    if np.any(rows<0) or np.any(cols<0) or max(rows.max(),cols.max())>=n or not all(np.isfinite(a).all() for a in (xij,h,off,overlap)):
        raise ValueError('Invalid active sparse generator')
    sites={s['site_id']:s['position_angstrom'] for s in metadata['sites']}
    positions=np.array([sites[o['site_id']] for o in metadata['orbitals']])/BOHR_ANGSTROM
    cell=np.asarray(metadata['geometry']['cell_angstrom'])/BOHR_ANGSTROM
    shift=(xij-(positions[cols]-positions[rows]))@np.linalg.inv(cell)
    if np.max(abs(shift-np.rint(shift)))>1.e-6:raise ValueError('Native displacement does not match the supplied geometry/site mapping')
    translations,inverse=np.unique(np.rint(shift).astype(int),axis=0,return_inverse=True)
    # Native packed h and lj use distinct offdiagonal imaginary conventions.
    full=np.empty((count,2,2),complex)
    full[:,0,0]=h[:,0]+1j*h[:,4];full[:,1,1]=h[:,1]+1j*h[:,5]
    full[:,0,1]=h[:,2]-1j*h[:,3];full[:,1,0]=h[:,6]+1j*h[:,7]
    soc=np.empty_like(full)
    soc[:,0,0]=off[:,0];soc[:,1,1]=off[:,1];soc[:,0,1]=off[:,2].conj();soc[:,1,0]=off[:,3].conj()
    trace=.5*(soc[:,0,0]+soc[:,1,1]);soc[:,0,0]-=trace;soc[:,1,1]-=trace
    electronic=full-soc if info['reference_has_soc'] else full
    base=np.zeros_like(full);base[:,0,0]=base[:,1,1]=.5*(electronic[:,0,0]+electronic[:,1,1])
    magnetic=electronic-base;s=np.zeros_like(full);s[:,0,0]=s[:,1,1]=overlap
    coefficients={}
    for name,values in [('h_base',base),('h_mag',magnetic),('h_soc',soc),('overlap',s)]:
        a=np.zeros((len(translations),2*n,2*n),complex)
        for i in range(2):
            for j in range(2):np.add.at(a,(inverse,2*rows+i,2*cols+j),values[:,i,j])
        coefficients[name]=a
    m=dict(metadata,evidence=list(metadata.get('evidence',[]))+[
        dict(path=str(manifest.resolve()),sha256=file_hash(manifest)),dict(path=str(path.resolve()),sha256=file_hash(path))])
    return RealSpaceHamiltonian(m,translations,coefficients)
