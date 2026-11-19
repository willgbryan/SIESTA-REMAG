"""Verify portable input regeneration without equating hashes to regeneration.

ASE converts Cartesian positions back to fractional coordinates with a linear
solve. Different BLAS implementations can differ at the last binary places.
Retained bytes/hashes remain exact. Reconstructed coordinates may differ by
eight binary64 epsilons. A nonzero SOC axis component in the manifest may
differ by one representable step or reproduce exactly from the same lattice
using a different binary64 reduction. If the rendered angles differ, both
spin blocks must regenerate from their respective axes. Different axes in
this case must both regenerate from the same lattice.
This is a serialization bound, never a native convergence tolerance.
"""
from pathlib import Path
from fractions import Fraction
from itertools import permutations
import math
import re
import numpy as np
from ..corpus import sha256
from ..discovery.common import load, digest

FRACTIONAL_ROUNDOFF = 8 * np.finfo(float).eps
START = '%block AtomicCoordinatesAndAtomicSpecies\n'
END = '%endblock AtomicCoordinatesAndAtomicSpecies\n'


def _dot_values(a, b):
    """Binary64 three-term dot products with separate or fused multiply-add."""
    terms = [x*y for x,y in zip(a,b)]
    def fused(i, partial):
        return float(Fraction(a[i])*Fraction(b[i]) + Fraction(partial))
    values = {math.fsum(terms)}
    for i,j,k in permutations(range(3)):
        for partial in (terms[i]+terms[j], fused(i,terms[j])):
            values.update((partial+terms[k], fused(k,partial)))
    return values


def _unit_vectors(v):
    lengths = {math.sqrt(s) for s in _dot_values(v,v)} | {math.hypot(*v)}
    return {tuple(x/d for x in v) for d in lengths if d > 0}


def _reconstructed_axes(cell, stage):
    """Repeat discovery.inputs.axes and prepare_case's final normalization."""
    if stage not in ('soc-x','soc-y','soc-z'):
        return set()
    a, _, c = cell
    axes = set()
    for z in _unit_vectors(c):
        if stage == 'soc-z':
            axes.update(_unit_vectors(z))
            continue
        for projection in _dot_values(a,z):
            residual = tuple(x-projection*y for x,y in zip(a,z))
            for x_axis in _unit_vectors(residual):
                if stage == 'soc-x':
                    axes.update(_unit_vectors(x_axis))
                else:
                    axes.update(_unit_vectors(tuple(np.cross(z,x_axis).tolist())))
    return axes


def _spin_renderings(manifest):
    moments = manifest['initial_moments_muB']
    x, y, z = manifest['direction']
    z = min(1.0, max(-1.0, z))
    polar = math.acos(z)
    alternate = math.atan2(math.sqrt((1.0-z)*(1.0+z)), z)
    phi = math.degrees(math.atan2(y, x))
    angles = (polar, alternate) if abs(alternate-polar) <= math.ulp(polar) else (polar,)
    for angle in angles:
        theta = math.degrees(angle)
        yield [f'{i} {abs(m):.12f} {theta if m >= 0 else 180-theta:.12f} '
               f'{phi if m >= 0 else phi+180:.12f}\n' for i, m in enumerate(moments, 1)]


def _matches_polar_rendering(expected, actual, left, right):
    """Validate each complete block against its own source-derived direction."""
    start, end = '%block DM.InitSpin\n', '%endblock DM.InitSpin\n'
    if expected.count(start) != 1 or expected.count(end) != 1:
        return False
    first, last = expected.index(start) + 1, expected.index(end)
    if expected[:first] != actual[:first] or expected[last:] != actual[last:]:
        return False
    if left['direction'] != right['direction']:
        if left['geometry']['lattice_A'] != right['geometry']['lattice_A']:
            return False
        axes = _reconstructed_axes(left['geometry']['lattice_A'], left['name'])
        if not all(tuple(m['direction']) in axes for m in (left, right)):
            return False
    return expected[first:last] in _spin_renderings(left) and actual[first:last] in _spin_renderings(right)


def verify_prepared_input(expected, actual):
    expected, actual = Path(expected), Path(actual)
    manifests = [load(p/'manifest.json') for p in (expected, actual)]
    texts = [(p/'input.fdf').read_bytes().decode('utf-8').splitlines(keepends=True) for p in (expected, actual)]
    coordinates = []
    stripped = []
    for folder, manifest, lines in zip((expected, actual), manifests, texts):
        if (manifest['input_sha256'] != sha256(folder/'input.fdf') or
                manifest['geometry_sha256'] != digest(manifest['geometry']) or
                lines.count(START) != 1 or lines.count(END) != 1):
            raise ValueError('Native input or geometry hash/block differs')
        begin, end = lines.index(START), lines.index(END)
        rows = lines[begin+1:end]
        if len(rows) != manifest['atoms'] or any(
                not re.fullmatch(r'-?\d+\.\d{16} -?\d+\.\d{16} -?\d+\.\d{16} [1-9]\d*\n', row) for row in rows):
            raise ValueError('Unexpected generated fractional coordinate layout')
        values = np.asarray([[float(v) for v in row.split()[:3]] for row in rows])
        geometry = np.asarray(manifest['geometry']['fractional_coordinates'], dtype=float)
        if (geometry.shape != values.shape or not np.isfinite(geometry).all() or
                [[f'{v:.16f}' for v in row] for row in geometry] != [row.split()[:3] for row in rows]):
            raise ValueError('Input coordinates do not reproduce their retained geometry')
        coordinates.append(geometry)
        # Preserve species/site order, whitespace, all other blocks and options.
        stripped.append(lines[:begin+1] + [row.split()[-1]+'\n' for row in rows] + lines[end:])
    a, b = manifests
    ignored = {'input_sha256', 'geometry', 'geometry_sha256'}
    if a['direction'] != b['direction'] and a['spin'] == b['spin'] == 'spin-orbit':
        left, right = (np.asarray(m['direction'], dtype=float) for m in manifests)
        if (left.shape == right.shape == (3,) and np.isfinite(left).all() and np.isfinite(right).all()
                and np.all((left == right) | ((left * right > 0) & (np.nextafter(left, right) == right)))):
            # Adjacent components alone only admit identical rendered input.
            ignored.add('direction')
        elif left.shape == right.shape == (3,) and a['geometry']['lattice_A'] == b['geometry']['lattice_A']:
            axes = _reconstructed_axes(a['geometry']['lattice_A'], a['name'])
            if tuple(left) in axes and tuple(right) in axes:
                ignored.add('direction')
    spin_roundoff = (stripped[0] != stripped[1] and a['spin'] == b['spin'] == 'spin-orbit' and
                     (a['direction'] == b['direction'] or 'direction' in ignored) and
                     _matches_polar_rendering(*stripped, a, b))
    if ({k:v for k,v in a.items() if k not in ignored} != {k:v for k,v in b.items() if k not in ignored} or
            {k:v for k,v in a['geometry'].items() if k != 'fractional_coordinates'} !=
            {k:v for k,v in b['geometry'].items() if k != 'fractional_coordinates'} or
            (stripped[0] != stripped[1] and not spin_roundoff) or coordinates[0].shape != coordinates[1].shape):
        raise ValueError('Native input does not reproduce from the source-bound preparation')
    delta = float(np.max(np.abs(coordinates[0]-coordinates[1])))
    if not np.isfinite(delta) or delta > FRACTIONAL_ROUNDOFF:
        raise ValueError('Native input coordinates exceed portable reconstruction roundoff')
    proof = dict(exact_input_bytes=texts[0] == texts[1], max_fractional_reconstruction_delta=delta,
                 fractional_roundoff_bound=float(FRACTIONAL_ROUNDOFF), retained_hashes_verified=True)
    if spin_roundoff:
        proof['soc_polar_rendering_reproduced'] = True
    return proof
