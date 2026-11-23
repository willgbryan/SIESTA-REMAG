"""Shared GPU matrix operations for optional response methods (no CPU solve).

CuPy dispatches to cuBLAS/cuSOLVER. The SCF/MFT engine remains the native REMAG
library. Only requested response blocks and compact diagnostics are downloaded.
"""
import numpy as np


def cuda():
    try:
        import cupy as cp
        import cupyx
    except ImportError as exc:
        raise RuntimeError('Optional response methods require siesta-remag[magnetic] on a CUDA host') from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('A CUDA device is required; no CPU response fallback exists')
    return cp, cupyx


def adjoint(a):
    return a.swapaxes(-1,-2).conj()


def generalized_eigh(h, s, *, factor=None):
    cp, cupyx = cuda()
    with cupyx.errstate(linalg='raise'):
        l = cp.linalg.cholesky(s) if factor is None else factor
        a = cp.linalg.solve(l,h)
        a = adjoint(cp.linalg.solve(l,adjoint(a)))
        a = (a+adjoint(a))*.5
        e,u = cp.linalg.eigh(a)
        c = cp.linalg.solve(adjoint(l),u)
    residual = h@c-(s@c)*e[...,None,:]
    denom = (cp.linalg.norm(h,axis=(-2,-1))[...,None]+cp.abs(e)*cp.linalg.norm(s,axis=(-2,-1))[...,None])*cp.linalg.norm(c,axis=-2)
    r = float(cp.max(cp.linalg.norm(residual,axis=-2)/cp.maximum(denom,1.e-300)).get())
    orth = float(cp.max(cp.abs(adjoint(c)@s@c-cp.eye(h.shape[-1]))).get())
    if not bool(cp.all(cp.isfinite(e)).get()) or not np.isfinite(r) or r>1.e-8 or not np.isfinite(orth) or orth>1.e-8:
        raise RuntimeError('GPU generalized eigensystem failed finite/residual/orthogonality checks')
    return e,c,l,dict(max_residual=r,max_orthogonality_error=orth)


def cell_gauge(state):
    """H(k)=sum_R H(R) exp(+2pi i k.R), with orbitals in one unit cell.

    Native SIESTA uses exp(-ik.(R+rj-ri)); first remove orbital phases, then
    relabel k -> -k. This is a gauge change, with the same sampled eigenvalues.
    """
    cp,_=cuda();m=state.metadata
    banks=[cp.asarray(getattr(state,key)) for key in ('h_base','h_mag','h_soc','overlap')]
    if m.get('bloch_gauge')=='cell_periodic_positive' and m['kpoint_convention']=='fractional_reciprocal':
        return banks,cp.asarray(state.kpoints)
    if m.get('bloch_gauge')!='siesta_atomic_negative' or m['kpoint_convention']!='cartesian_radians_per_bohr':
        raise ValueError('Response interpolation requires an explicit supported Bloch gauge')
    geometry=m.get('geometry') or {};orbitals=m.get('orbitals');sites=m.get('sites')
    if not geometry or not orbitals or not sites:
        raise ValueError('Native cell-gauge conversion requires geometry and orbital-to-site identity')
    from .operators import BOHR_ANGSTROM
    positions={site['site_id']:site['position_angstrom'] for site in sites}
    r=np.array([positions[o['site_id']] for o in orbitals])/BOHR_ANGSTROM
    phase=cp.exp(1j*cp.asarray(state.kpoints@r.T))
    phase=cp.repeat(phase,2,axis=1)
    # H_atomic = D H_negative D^dagger, D_i=exp(+ik.ri).
    banks=[phase[:,:,None].conj()*a*phase[:,None,:] for a in banks]
    cell=np.asarray(geometry['cell_angstrom'])/BOHR_ANGSTROM
    return banks,cp.asarray(-state.kpoints@cell.T/(2*np.pi))
