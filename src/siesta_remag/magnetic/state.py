"""Portable REMAG-MAGNETIC-STATE/1, with explicit native operator provenance.

The initial representation stores full sampled-k matrices. It supports reuse
of an SCF state; it does not infer unsampled k points or replace a generator
capture with a terminal HSX. NPZ contains arrays only (no pickled objects).
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from ..angular_energy import unit_directions
from .operators import decompose, hermitian, spin_scalar, spin_trace, native_lower_hermitian

SCHEMA = "REMAG-MAGNETIC-STATE/1"
ARRAYS = ("h_base", "h_mag", "h_soc", "overlap", "kpoints", "weights", "reference_spectrum")


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@dataclass
class MagneticState:
    metadata: dict
    h_base: np.ndarray
    h_mag: np.ndarray
    h_soc: np.ndarray
    overlap: np.ndarray
    kpoints: np.ndarray
    weights: np.ndarray
    reference_spectrum: np.ndarray | None = None

    def __post_init__(self):
        # Copy so caller mutations cannot silently change a loaded state.
        self.metadata = json.loads(json.dumps(self.metadata, allow_nan=False))
        m = self.metadata
        if m.get('schema') != SCHEMA or m.get('converged') is not True:
            raise ValueError('A converged REMAG-MAGNETIC-STATE/1 is required')
        if (m.get('energy_unit') != 'Ry' or m.get('spinor_order') != 'orbital_up_down'
                or m.get('occupation_model') != 'fermi_dirac'):
            raise ValueError('Supported state: Ry, interleaved spinors, Fermi–Dirac occupations')
        if m.get('magnetic_terms') != {'xc': 'rigid_spin_rotation', 'hubbard': 'absent',
                                       'constraint': 'absent', 'external_field': 'absent'}:
            raise ValueError('Initial decomposition supports XC exchange without Hubbard, constraints or external fields')
        if m.get('soc_source') != 'native_offsite' or type(m.get('reference_has_soc')) is not bool:
            raise ValueError('Native offsite SOC provenance and reference SOC treatment are required')
        if not m.get('calculation_id') or not m.get('evidence'):
            raise ValueError('Calculation identity and original evidence are required')
        m['reference_direction'] = unit_directions([m['reference_direction']])[0].tolist()
        for key in ('h_base', 'h_mag', 'h_soc', 'overlap'):
            setattr(self, key, hermitian(getattr(self, key), key))
        shape = self.h_base.shape
        if len(shape) != 3 or shape[0] < 1 or shape[1] < 2 or shape[1] % 2:
            raise ValueError('Matrices must have shape (nk, 2*norb, 2*norb)')
        if any(getattr(self, k).shape != shape for k in ('h_mag', 'h_soc', 'overlap')):
            raise ValueError('H/S/operator shapes differ')
        for name in ('h_base', 'overlap'):
            a = getattr(self, name)
            if np.max(abs(a-spin_scalar(spin_trace(a)))) > 1.e-10:
                raise ValueError(f'{name} must be spin independent')
        for name in ('h_mag', 'h_soc'):
            if np.max(abs(spin_trace(getattr(self, name)))) > 1.e-10:
                raise ValueError(f'{name} must be spin traceless')
        # A global collinear exchange field is B_orbital * n.sigma.
        n = np.asarray(m['reference_direction'])
        vector = np.stack([.5*(self.h_mag[:, ::2, 1::2]+self.h_mag[:, 1::2, ::2]),
                           .5j*(self.h_mag[:, ::2, 1::2]-self.h_mag[:, 1::2, ::2]),
                           .5*(self.h_mag[:, ::2, ::2]-self.h_mag[:, 1::2, 1::2])])
        parallel = np.einsum('a,akij->kij', n, vector)
        if np.max(abs(vector-n[:, None, None, None]*parallel)) > m.get('collinear_tolerance_ry', 1.e-8):
            raise ValueError('Initial MFT interface requires a collinear reference exchange field')
        self.kpoints = np.array(self.kpoints, dtype=float, copy=True)
        self.weights = np.array(self.weights, dtype=float, copy=True)
        nk, bands, _ = shape
        if (self.kpoints.shape != (nk, 3) or not np.isfinite(self.kpoints).all()
                or self.weights.shape != (nk,) or not np.isfinite(self.weights).all()
                or np.any(self.weights <= 0) or abs(self.weights.sum()-1) > 1.e-10):
            raise ValueError('Complete k mesh with finite coordinates and positive normalized weights required')
        if m.get('kpoint_convention') not in ('fractional_reciprocal', 'cartesian_radians_per_bohr'):
            raise ValueError('Explicit k-point convention required')
        if m.get('mesh_complete') is not True:
            raise ValueError('Selected-k captures cannot be used as a full magnetic screening state')
        temperature, electrons = m.get('temperature_ry'), m.get('electrons')
        if (not isinstance(temperature, (int, float)) or not np.isfinite(temperature) or temperature < 1.e-6
                or not isinstance(electrons, (int, float)) or not np.isfinite(electrons) or not 0 < electrons < bands):
            raise ValueError('Finite positive electron count and temperature >= 1e-6 Ry required')
        orbitals = m.get('orbitals')
        if orbitals is not None and (len(orbitals) != bands//2 or
                any(not isinstance(o, dict) or not o.get('site_id') for o in orbitals)):
            raise ValueError('Orbital mapping must identify every spatial orbital and site')
        sites=m.get('sites')
        geometry=m.get('geometry')
        if geometry is not None:
            cell=np.asarray(geometry.get('cell_angstrom'),dtype=float)
            if cell.shape!=(3,3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell))<1.e-12:
                raise ValueError('A finite nonsingular evaluated lattice is required')
            volume=float(abs(np.linalg.det(cell)))
            if m.get('volume_angstrom3') is not None and not np.isclose(m['volume_angstrom3'],volume,rtol=1.e-8,atol=1.e-8):
                raise ValueError('Declared volume differs from the evaluated lattice')
            m['volume_angstrom3']=volume
        if sites is not None:
            identities=[site.get('site_id') for site in sites]
            if not all(identities) or len(set(identities))!=len(identities):
                raise ValueError('Distinct atomic site identities required')
            positions=np.asarray([site.get('position_angstrom') for site in sites],dtype=float)
            if positions.shape!=(len(sites),3) or not np.isfinite(positions).all():
                raise ValueError('Finite evaluated positions required for every site')
            if orbitals and any(o['site_id'] not in identities for o in orbitals):
                raise ValueError('An orbital refers to an unknown site')
        for key in ('gpac_source_hash','evaluated_geometry_hash'):
            value=m.get(key)
            if value is not None and (not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError(f'{key} must be a SHA256 hex digest or null')
        if self.reference_spectrum is not None:
            self.reference_spectrum = np.array(self.reference_spectrum, dtype=float, copy=True)
            if self.reference_spectrum.shape != (nk, bands) or not np.isfinite(self.reference_spectrum).all():
                raise ValueError('Reference spectrum must cover every k/band')
        for key in ARRAYS:
            a = getattr(self, key)
            if a is not None:
                a.setflags(write=False)

    @property
    def shape(self):
        return self.h_base.shape

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        arrays = {key: getattr(self, key) for key in ARRAYS if getattr(self, key) is not None}
        np.savez(path/'matrices.npz', **arrays)
        meta = dict(self.metadata, arrays_file='matrices.npz', arrays_sha256=file_hash(path/'matrices.npz'))
        (path/'magnetic-state.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
        return path/'magnetic-state.json'


def load_magnetic_state(calculation):
    path = Path(calculation)
    if path.is_dir():
        path = path/'magnetic-state.json'
    meta = json.loads(path.read_text())
    name = meta.get('arrays_file')
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError('State arrays must be a local NPZ file')
    arrays = path.parent/name
    if file_hash(arrays) != meta.get('arrays_sha256'):
        raise ValueError('Magnetic-state array hash differs from its manifest')
    with np.load(arrays, allow_pickle=False) as archive:
        values = {key: archive[key] for key in ARRAYS if key in archive}
    return MagneticState(meta, **values)


def import_native_state(directory, metadata):
    """Attach geometry/site/GPAC identity to the complete native generator export.

    Metadata must come from the actual calculation and structure handoff. No
    GPAC hash, XC convention or magnetic direction is inferred from filenames.
    """
    root = Path(directory)
    path = root/'remag-magnetic-export.json'
    export = json.loads(path.read_text())
    if export.get('schema') != 1 or export.get('complete') is not True:
        raise ValueError('Incomplete native magnetic generator export')
    if (export.get('reference')!='Final SCF generator Hamiltonian' or export.get('dtype')!='<c16'
            or export.get('order')!='F' or export.get('matrices')!=['generator','overlap','offsite']
            or type(export.get('reference_has_soc')) is not bool):
        raise ValueError('Unsupported native generator layout or operator provenance')
    nk, n = export['kpoints'], export['bands']
    if type(nk) is not int or type(n) is not int or nk < 1 or n < 2 or n % 2:
        raise ValueError('Invalid native dimensions')
    binary = root/'remag-magnetic-export.bin'
    expected = nk*(3*n*n*16+(n+4)*8)
    if binary.stat().st_size != expected:
        raise ValueError('Native export is truncated or oversized')
    h, s, soc = (np.empty((nk, n, n), complex) for _ in range(3))
    kpoints, weights, spectrum = np.empty((nk, 3)), np.empty(nk), np.empty((nk, n))
    with binary.open('rb') as stream:
        for k in range(nk):
            for bank in (h, s, soc):
                bank[k] = np.fromfile(stream, dtype='<c16', count=n*n).reshape(n, n, order='F')
            kpoints[k] = np.fromfile(stream, dtype='<f8', count=3)
            weights[k] = np.fromfile(stream, dtype='<f8', count=1)[0]
            spectrum[k] = np.fromfile(stream, dtype='<f8', count=n)
    triangle_errors={name:float(np.max(abs(a-a.swapaxes(-1,-2).conj())))
                     for name,a in (('generator',h),('overlap',s),('offsite',soc))}
    h,s,soc=(native_lower_hermitian(a) for a in (h,s,soc))
    base, magnetic, soc = decompose(h, soc, reference_has_soc=export['reference_has_soc'])
    # Verify reconstruction before releasing the original generator bank.
    reconstructed = base+magnetic+(soc if export['reference_has_soc'] else 0)
    error = float(np.max(abs(h-reconstructed)))
    if error > 1.e-10:
        raise ValueError('Native operator decomposition does not reconstruct the generator')
    m = dict(metadata, schema=SCHEMA, energy_unit='Ry', spinor_order='orbital_up_down',
             converged=True, occupation_model='fermi_dirac', soc_source='native_offsite',
             reference_has_soc=export['reference_has_soc'], electrons=export['electrons'],
             temperature_ry=export['temperature_ry'], mesh_complete=True,
             bloch_gauge='siesta_atomic_negative',
             kpoint_convention='cartesian_radians_per_bohr',
             reconstruction_max_abs_ry=error)
    m['native_solver_triangle']='lower'
    m['native_raw_triangle_discrepancies']=triangle_errors
    m['evidence'] = list(m.get('evidence', []))+[
        {'path': str(path.resolve()), 'sha256': file_hash(path)},
        {'path': str(binary.resolve()), 'sha256': file_hash(binary)}]
    return MagneticState(m, base, magnetic, soc, s, kpoints, weights, spectrum)
