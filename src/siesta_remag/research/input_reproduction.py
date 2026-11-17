"""Verify portable input regeneration without equating hashes to regeneration.

ASE converts Cartesian positions back to fractional coordinates with a linear
solve. Different BLAS implementations can differ at the last binary places.
Retained bytes/hashes remain exact. Reconstructed coordinates may differ by
eight binary64 epsilons. A nonzero SOC axis component in the manifest may
differ by one representable step when the generated spin instructions match.
This is a serialization bound, never a native convergence tolerance.
"""
from pathlib import Path
import re
import numpy as np
from ..corpus import sha256
from ..discovery.common import load, digest

FRACTIONAL_ROUNDOFF = 8 * np.finfo(float).eps
START = '%block AtomicCoordinatesAndAtomicSpecies\n'
END = '%endblock AtomicCoordinatesAndAtomicSpecies\n'


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
            # The full DM.InitSpin block, including every rendered angle, is
            # compared byte for byte below. The source-derived axis is expected.
            ignored.add('direction')
    if ({k:v for k,v in a.items() if k not in ignored} != {k:v for k,v in b.items() if k not in ignored} or
            {k:v for k,v in a['geometry'].items() if k != 'fractional_coordinates'} !=
            {k:v for k,v in b['geometry'].items() if k != 'fractional_coordinates'} or
            stripped[0] != stripped[1] or coordinates[0].shape != coordinates[1].shape):
        raise ValueError('Native input does not reproduce from the source-bound preparation')
    delta = float(np.max(np.abs(coordinates[0]-coordinates[1])))
    if not np.isfinite(delta) or delta > FRACTIONAL_ROUNDOFF:
        raise ValueError('Native input coordinates exceed portable reconstruction roundoff')
    return dict(exact_input_bytes=texts[0] == texts[1], max_fractional_reconstruction_delta=delta,
                fractional_roundoff_bound=float(FRACTIONAL_ROUNDOFF), retained_hashes_verified=True)
