"""Bounded split-valence PAO specifications, never arbitrary FDF text."""
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from ..corpus import sha256


def validate_basis(specification, symbols):
    """Return a detached canonical definition covering exactly the supplied species.

    Radii are bohr; zero asks the pinned SIESTA generator to derive a radius.
    Polarization uses the existing nonperturbative scheme. Numerical adequacy
    and pseudopotential shell compatibility still require native evidence.
    """
    if (not isinstance(specification, dict) or set(specification) != {'format', 'species'}
        or specification['format'] != 'REMAG-PAO-BASIS/1'
        or not isinstance(specification['species'], dict)
        or set(specification['species']) != set(symbols)):
        raise ValueError('Explicit PAO basis must cover exactly the calculation species')
    result = dict(format='REMAG-PAO-BASIS/1', species={})
    for symbol in sorted(set(symbols)):
        shells = specification['species'][symbol]
        if not isinstance(shells, list) or not 1 <= len(shells) <= 16:
            raise ValueError('Each PAO species requires 1..16 shells')
        seen = set(); canonical = []
        for shell in shells:
            if not isinstance(shell, dict) or set(shell) != {'n', 'l', 'zeta', 'polarization', 'radii_bohr'}:
                raise ValueError('Unsupported PAO shell fields')
            n, l, zeta, pol = [shell[k] for k in ['n', 'l', 'zeta', 'polarization']]
            if (any(type(v) is not int for v in [n, l, zeta, pol]) or not 0 <= l <= 3
                or not l < n <= 10 or not 1 <= zeta <= 4 or not 0 <= pol <= 2
                or (pol and l == 3) or (n, l) in seen):
                raise ValueError('Invalid or repeated PAO shell quantum numbers')
            radii = shell['radii_bohr']
            if (not isinstance(radii, list) or len(radii) != zeta
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 30 for v in radii)
                or (radii[0] > 0 and any(v > radii[0] for v in radii[1:]))):
                raise ValueError('One finite PAO cutoff per zeta, in 0..30 bohr, is required')
            seen.add((n, l))
            canonical.append(dict(n=n, l=l, zeta=zeta, polarization=pol, radii_bohr=list(radii)))
        angular = [s['l'] for s in canonical]
        generated = [s['l']+1 for s in canonical if s['polarization']]
        if set(angular) & set(generated) or len(set(generated)) != len(generated):
            raise ValueError('Explicit and generated polarization shells overlap')
        result['species'][symbol] = sorted(canonical, key=lambda s: (s['l'], s['n']))
    return result


def basis_lines(specification, symbols):
    spec = validate_basis(specification, symbols)
    lines = ['PAO.BasisType split', '%block PAO.Basis']
    for symbol in dict.fromkeys(symbols):
        shells = spec['species'][symbol]
        lines.append(f'{symbol} {len(shells)}')
        for shell in shells:
            line = f"n={shell['n']} {shell['l']} {shell['zeta']}"
            if shell['polarization']: line += f" P {shell['polarization']}"
            lines.extend([line, ' '.join(f'{r:.16g}' for r in shell['radii_bohr']),
                          ' '.join('1.0' for _ in shell['radii_bohr'])])
    return lines + ['%endblock PAO.Basis']


def orbital_count(specification, symbols):
    spec = validate_basis(specification, symbols)
    return sum(sum(s['zeta']*(2*s['l']+1) + s['polarization']*(2*s['l']+3)
                   for s in spec['species'][symbol]) for symbol in symbols)


def audit_native_basis(directory, specification, symbols):
    """Check actual shells/counts/radii; this is integrity, not convergence."""
    directory = Path(directory); spec = validate_basis(specification, symbols)
    expected = orbital_count(spec, symbols)
    counts = re.findall(r'initatomlists: Number of atoms, orbitals, and projectors:\s+(\d+)\s+(\d+)\s+(\d+)',
                        (directory/'stdout.log').read_text(errors='replace'))
    if not counts or any(int(a) != len(symbols) or int(b) != expected for a, b, _ in counts):
        raise ValueError('Native PAO orbital/atom count differs from requested basis')
    species = []
    for symbol, shells in spec['species'].items():
        path = directory/f'{symbol}.ion.xml'; tree = ET.parse(path)
        if tree.findtext('symbol', '').strip() != symbol:
            raise ValueError('Native PAO species differs')
        actual = []
        for o in tree.findall('paos/orbital'):
            item = dict(n=int(o.attrib['n']), l=int(o.attrib['l']), zeta=int(o.attrib['z']),
                        cutoff_bohr=float(o.findtext('radfunc/cutoff')))
            if not math.isfinite(item['cutoff_bohr']) or item['cutoff_bohr'] <= 0:
                raise ValueError('Invalid native PAO cutoff')
            actual.append(item)
        keys = {(o['n'], o['l'], o['zeta']) for o in actual}
        if len(keys) != len(actual): raise ValueError('Duplicate native PAO radial orbital')
        base = {(s['n'], s['l'], z) for s in shells for z in range(1, s['zeta']+1)}
        if not base <= keys: raise ValueError('Requested PAO shell missing from native basis')
        extra = keys-base
        for shell in shells:
            if not shell['polarization']: continue
            found = {key for key in extra if key[1] == shell['l']+1}
            if (len({key[0] for key in found}) != 1
                or {key[2] for key in found} != set(range(1, shell['polarization']+1))):
                raise ValueError('Native PAO polarization multiplicity differs')
            extra -= found
        if extra: raise ValueError('Unexpected native PAO shells')
        species.append(dict(symbol=symbol, ion_sha256=sha256(path), orbitals=sorted(actual, key=lambda o: (o['l'], o['n'], o['zeta']))))
    return dict(passed=True, spatial_orbitals=expected, species=species,
                scope='Native basis identity and finite positive radii only; no conditioning or completeness certificate')
