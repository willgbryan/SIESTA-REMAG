"""Native-density orbital moments in the pinned moments.F approximation.

This reproduces the solver's same-atom, same-l, same-zeta, zero-cell
contraction. It is not a complete orbital-magnetization theory or a Mulliken
partition of a general nonorthogonal angular-momentum operator.
"""
from itertools import groupby
from pathlib import Path
import re
import zipfile

import numpy as np

from ..corpus import sha256
from ..density import read_dm
from ..discovery.common import load, save
from ..hsx import read_hsx
from .electronic import orbital_metadata


def magnetic_orbital_metadata(hsx):
    """Recover m only for complete shells in pinned atom.F's -l..l ordering.

HSX itself stores n/l/zeta, not m. Both comBasis and comPOL in the pinned
producer enumerate real harmonics in increasing m. Reject incomplete or
ambiguous repeated shell runs instead of guessing another producer's order.
"""
    rows = orbital_metadata(hsx)
    for _, group in groupby(rows, key=lambda x: (x['atom_index'], x['n'], x['l'], x['zeta'])):
        shell = list(group)
        l = shell[0]['l']
        if not 0 <= l <= 3 or len(shell) != 2*l+1 or shell[0]['zeta'] < 1:
            raise ValueError('Orbital moments require complete pinned s/p/d/f real-harmonic shells')
        for m, orbital in zip(range(-l, l+1), shell):
            orbital['m'] = m
    return rows


def _angular(l, mi, mj):
    """int_so_ang from the pinned spinorbit.f, restricted to s/p/d/f."""
    if l == 0:
        return np.zeros(3)
    a = np.sqrt(l*(l+1)/2)
    b = np.sqrt(l*(l+1)-2)/2
    c = np.sqrt(l*(l+1)-6)/2 if l >= 3 else 0.
    entries = {
        (0,-1):(2,a), (0,1):(1,a), (1,-2):(2,b), (1,0):(1,-a), (1,2):(1,b),
        (-1,-2):(1,b), (-1,0):(2,-a), (-1,2):(2,-b), (2,-3):(2,c), (2,-1):(2,b),
        (2,1):(1,-b), (2,3):(1,c), (-2,-3):(1,c), (-2,-1):(1,-b), (-2,1):(2,-b),
        (-2,3):(2,-c), (3,-2):(2,c), (3,2):(1,-c), (-3,-2):(1,-c), (-3,2):(2,-c),
    }
    result = np.array([float(mj) if mi+mj == 0 else 0., 0., 0.])
    if (mi, mj) in entries:
        axis, value = entries[mi, mj]; result[axis] = value
    return result


def moment_kernels(orbitals):
    """Real contraction weights, not covariant operators for a general basis.

moment[i,axis] = sum_j kernel[axis,i,j]*(native_DM[4,i,j]+native_DM[5,i,j]).
The native convention includes cross-n terms if l/zeta agree.
"""
    result = np.zeros((3, len(orbitals), len(orbitals)))
    for i, a in enumerate(orbitals):
        for j, b in enumerate(orbitals):
            if a['l'] and all(a[k] == b[k] for k in ('atom_index','l','zeta')) and a['m'] != b['m']:
                angular = _angular(a['l'], b['m'], a['m'])
                result[:, i, j] = angular[[2, 1, 0]] * [1, -1, -1]
    return result


def native_moments(hsx, dm):
    n, spins = map(int, hsx['header'][1:3])
    if n > 512 or spins != 8 or not np.array_equal(dm['header'], hsx['header'][[1,2,4,5,6]]):
        raise ValueError('Orbital moments require matching SOC DM/HSX dimensions, at most 512 orbitals')
    if not np.array_equal(hsx['offsets'][:3], [0,0,0]):
        raise ValueError('Pinned unit-cell orbital indexing is required')
    if not np.array_equal(dm['counts'], hsx['counts']) or not np.array_equal(dm['columns'], hsx['columns']):
        raise ValueError('Native DM and HSX sparse topologies differ')
    orbitals = magnetic_orbital_metadata(hsx)
    rows = np.repeat(np.arange(n), dm['counts'])
    mask = dm['columns'] <= n  # Same jo == indxuo(jo) test as native moments.F.
    rows, cols = rows[mask], dm['columns'][mask]-1
    if len(set(zip(rows.tolist(), cols.tolist()))) != len(rows):
        raise ValueError('Repeated native unit-cell density entries')
    unit = np.zeros((8,n,n))
    unit[:,rows,cols] = dm['density'][:,mask]
    kernels = moment_kernels(orbitals)
    per_orbital = np.einsum('aij,ij->ia', kernels, unit[4]+unit[5])
    per_atom = np.zeros((int(hsx['header'][0]),3))
    np.add.at(per_atom, [o['atom_index'] for o in orbitals], per_orbital)
    return orbitals, dict(native_unitcell_density_components=unit, moment_kernels=kernels,
                         orbital_moments_mu_b=per_orbital, atom_moments_mu_b=per_atom)


def _printed_moments(path, orbitals):
    """Check the final native table, including each orbital, atom and total."""
    if path.stat().st_size > 64*1024*1024:
        raise ValueError('Native log exceeds orbital-moment inspection bound')
    text = path.read_text()
    marker = 'moments: Magnetic moments from orbital angular momenta:'
    if marker not in text:
        raise ValueError('Final native orbital-moment table is missing')
    table = text.rsplit(marker, 1)[1]
    by_orbital, by_atom, total = {}, {}, None
    for line in table.splitlines():
        fields = line.split()
        if len(fields) == 7 and fields[0].isdigit() and fields[1].isdigit():
            i, a = int(fields[1])-1, int(fields[0])-1
            if i in by_orbital or not 0 <= i < len(orbitals) or orbitals[i]['atom_index'] != a:
                raise ValueError('Native orbital-moment labels differ')
            by_orbital[i] = [float(x) for x in fields[-3:]]
        elif len(fields) == 6 and fields[0].isdigit() and fields[1] == 'Total':
            a = int(fields[0])-1
            if a in by_atom:
                raise ValueError('Repeated native atom moment')
            by_atom[a] = [float(x) for x in fields[-3:]]
        elif len(fields) == 5 and fields[0] == 'Sum':
            total = [float(x) for x in fields[-3:]]
            break
    atoms = sorted({o['atom_index'] for o in orbitals})
    if sorted(by_orbital) != list(range(len(orbitals))) or sorted(by_atom) != atoms or total is None:
        raise ValueError('Incomplete final native orbital-moment table')
    result = dict(orbital=np.array([by_orbital[i] for i in range(len(orbitals))]),
                  atom=np.array([by_atom[i] for i in atoms]), total=np.array(total))
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError('Nonfinite printed native moments')
    return result


def export_orbital_moments(report_path, stage, native_directory, output, *, collection=None):
    """Inspect an accepted saved native SOC stage without running a solver."""
    from .tools import _verified_report
    report_path = Path(report_path).resolve()
    if not isinstance(stage, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', stage):
        raise ValueError('Invalid native stage name')
    report = _verified_report(report_path, collection)
    stages = [s for s in report['stages'] if s['stage'] == stage]
    if len(stages) != 1 or stages[0]['accepted'] is not True or not stages[0].get('branch', {}).get('passed', False):
        raise ValueError('Orbital moments require an accepted SOC stage with passing branch')
    native = Path(native_directory).resolve()
    paths = dict(hsx=native/'remag.HSX', dm=native/'remag.DM', stdout=report_path.parent/stage/'stdout.log')
    evidence = {}
    for key, path in paths.items():
        if not (path.resolve().is_relative_to(native) or path.resolve().is_relative_to(report_path.parent)):
            raise ValueError('Native moment evidence escapes its directories')
        evidence[key] = sha256(path)
        if report['artifacts'].get(stage+'/'+path.name) != evidence[key]:
            raise ValueError('Native moment artifact is not bound by report: '+path.name)
    if paths['hsx'].stat().st_size > 512*1024*1024:
        raise ValueError('Native HSX exceeds orbital-moment inspection bound')
    hsx = read_hsx(paths['hsx'])
    orbitals, arrays = native_moments(hsx, read_dm(paths['dm']))
    printed = _printed_moments(paths['stdout'], orbitals)
    total = arrays['atom_moments_mu_b'].sum(axis=0)
    residual = {key:float(np.max(abs(printed[key]-value))) for key,value in
                [('orbital',arrays['orbital_moments_mu_b']),('atom',arrays['atom_moments_mu_b']),('total',total)]}
    # Native tables print exactly three decimals. This is a reproduction check,
    # not a statement of the physical accuracy of the approximation.
    if any(x > .00050001 for x in residual.values()):
        raise ValueError('Reconstructed orbital moments differ from final native table')
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(root/'orbital-moments.npz', **arrays)
    result = dict(format='REMAG-ORBITAL-MOMENTS/1', artifact='orbital-moments.npz',
        artifact_sha256=sha256(root/'orbital-moments.npz'), report_sha256=sha256(report_path),
        implementation_sha256={name:sha256(Path(__file__).parent/name) for name in
            ('orbital_moments.py','electronic.py','../density.py','../hsx.py')},
        source_sha256=report['source_sha256'], executable_sha256=report['executable_sha256'], stage=stage,
        evidence=evidence, spin_components=8, frame='Cartesian solver frame', unit='mu_B, native moments.F sign convention',
        orbitals=orbitals, orbital_moments_mu_b=arrays['orbital_moments_mu_b'].tolist(),
        atom_moments_mu_b=arrays['atom_moments_mu_b'].tolist(), total_moment_mu_b=total.tolist(),
        native_table_max_abs_residual_mu_b=residual, native_table_rounding_tolerance_mu_b=.00050001,
        harmonic_order='m=-l..l real harmonics recovered from complete HSX shells using pinned atom.F comBasis/comPOL order; m is not stored in HSX',
        density_convention='Native real-space DM component/row/column order, zero-cell block; imaginary spin trace uses components 5+6 (one-based); native k integration already included',
        approximation='Pinned moments.F: same atom, l and zeta, zero cell; no extra S multiplication or intersite/cross-zeta terms. Kernels are contraction weights, not general nonorthogonal angular-momentum operators.',
        scope='Saved native SOC density contraction; atom/orbital partitions are approximation- and basis-dependent. Not bulk orbital magnetization, isolated SOC, total MAE or physical-accuracy qualification.',
        physical_accuracy_qualified=False)
    save(root/'orbital-moments.json', result)
    return result


def load_orbital_moments(path):
    """Verify a saved moment artifact and recompute its displayed contractions."""
    path = Path(path).resolve(); meta = load(path)
    if meta.get('format') != 'REMAG-ORBITAL-MOMENTS/1' or meta.get('spin_components') != 8:
        raise ValueError('Expected native SOC orbital-moment metadata')
    artifact = (path.parent/meta['artifact']).resolve()
    if not artifact.is_relative_to(path.parent) or sha256(artifact) != meta['artifact_sha256']:
        raise ValueError('Orbital-moment artifact binding differs')
    with zipfile.ZipFile(artifact) as archive:
        if sum(x.file_size for x in archive.infolist()) > 48*1024*1024:
            raise ValueError('Orbital-moment arrays exceed inspection bound')
    orbitals = meta['orbitals']; n = len(orbitals)
    if not 1 <= n <= 512:
        raise ValueError('Invalid orbital-moment basis size')
    for i,o in enumerate(orbitals):
        if (any(type(o[k]) is not int for k in ('index','atom_index','atom_orbital_index','n','l','m','zeta'))
                or o['index'] != i or not 0 <= o['atom_index'] < n or not 0 <= o['l'] <= 3
                or o['n'] < 1 or o['zeta'] < 1):
            raise ValueError('Invalid orbital-moment labels')
    for _,group in groupby(orbitals,key=lambda x:(x['atom_index'],x['n'],x['l'],x['zeta'])):
        shell = list(group); l = shell[0]['l']
        if [o['m'] for o in shell] != list(range(-l,l+1)):
            raise ValueError('Orbital-moment harmonic ordering differs')
    atom_count = max(o['atom_index'] for o in orbitals)+1
    if sorted({o['atom_index'] for o in orbitals}) != list(range(atom_count)):
        raise ValueError('Noncontiguous orbital-moment atoms')
    with np.load(artifact,allow_pickle=False) as loaded:
        arrays = {key:loaded[key] for key in loaded.files}
    shapes = dict(native_unitcell_density_components=(8,n,n),moment_kernels=(3,n,n),
                  orbital_moments_mu_b=(n,3),atom_moments_mu_b=(atom_count,3))
    if set(arrays) != set(shapes) or any(arrays[k].shape != shape or arrays[k].dtype.kind != 'f' or
            not np.isfinite(arrays[k]).all() for k,shape in shapes.items()):
        raise ValueError('Invalid orbital-moment arrays')
    kernels = moment_kernels(orbitals)
    unit = arrays['native_unitcell_density_components']
    values = np.einsum('aij,ij->ia',kernels,unit[4]+unit[5])
    atoms = np.zeros((atom_count,3)); np.add.at(atoms,[o['atom_index'] for o in orbitals],values)
    for key,expected in [('moment_kernels',kernels),('orbital_moments_mu_b',values),('atom_moments_mu_b',atoms)]:
        if not np.allclose(arrays[key],expected,rtol=0,atol=1e-12):
            raise ValueError('Orbital-moment contraction differs from saved arrays')
    return meta, arrays
