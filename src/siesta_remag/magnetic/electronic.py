"""Named PAO response descriptors for structural hypotheses, not causal claims."""
import numpy as np
from .operators import RY_EV


def orbital_groups(metadata):
    sites={s['site_id']:s.get('species',s['site_id']) for s in metadata.get('sites',[])}
    groups={}
    for i,orbital in enumerate(metadata.get('orbitals',[])):
        l=orbital.get('l');shell='spdf'[l] if isinstance(l,int) and 0<=l<4 else 'unspecified'
        name=f"{orbital['site_id']}:{sites.get(orbital['site_id'],'?')}:{shell}"
        groups.setdefault(name,[]).append(i)
    return groups


def band_descriptors(metadata,energies,mu,weights,populations,*,width_ev=.25):
    groups=orbital_groups(metadata)
    if not groups:return None
    relative=(energies-mu)*RY_EV
    gaussian=np.exp(-.5*(relative/width_ev)**2)/(np.sqrt(2*np.pi)*width_ev)
    spectral=weights[:,None]*gaussian;group_weights={k:populations[:,:,ids].sum(axis=-1) for k,ids in groups.items()}
    projected=[]
    for name,p in group_weights.items():
        density=float(np.sum(spectral*p));centroid=None if abs(density)<1.e-12 else float(np.sum(spectral*p*relative)/density)
        projected.append(dict(group=name,fermi_projected_dos_states_per_ev_cell=density,near_fermi_centroid_ev=centroid))
    # A co-weight of named orbital groups on the same near-Fermi eigenstates.
    # Absolute Mulliken weights normalize this diagnostic; it is not a hopping,
    # bond energy, unique orbital partition or proof of hybridization causality.
    denominator=sum(abs(p) for p in group_weights.values());denominator=np.maximum(denominator,1.e-30)
    keys=list(groups);mixing=[];norm=max(float(spectral.sum()),1.e-30)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            if a.split(':')[0]==b.split(':')[0]:continue
            value=float(np.sum(spectral*4*abs(group_weights[a]*group_weights[b])/denominator**2)/norm)
            mixing.append(dict(groups=[a,b],near_fermi_co_weight=value))
    return dict(window_width_ev=width_ev,projected_dos=projected,
                intersite_band_mixing=sorted(mixing,key=lambda v:v['near_fermi_co_weight'],reverse=True),
                convention='Gaussian near-Fermi PAO Mulliken DOS; absolute-weight normalized band co-weight',
                interpretation='Representation-dependent electronic diagnostics; a structural mechanism remains a hypothesis')
