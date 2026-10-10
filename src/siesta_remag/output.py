"""Parse observables emitted by the pinned SIESTA release, failing closed on missing data."""
import re
import struct
from pathlib import Path
import numpy as np

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"

def floats(text):
    return [float(v.replace("D", "E").replace("d", "e")) for v in re.findall(NUMBER, text)]

def parse_output(text):
    def last(pattern):
        matches = re.findall(pattern, text, re.MULTILINE)
        return float(matches[-1].replace("D", "E")) if matches else None
    energy = last(r"^siesta:\s+Total\s*=\s*(" + NUMBER + ")")
    fermi = last(r"^siesta:\s+Fermi\s*=\s*(" + NUMBER + ")")
    soc = last(r"^siesta:\s+Eso\s*=\s*(" + NUMBER + ")")
    force_blocks = re.findall(r"siesta: Atomic forces \(eV/Ang\):[^\n]*\n((?:(?:siesta:)?[ \t]+\d+[ \t]+[^\n]+\n)+)", text)
    forces = [floats(line)[1:] for line in force_blocks[-1].splitlines()] if force_blocks else None
    stress_blocks = re.findall(r"siesta: Stress tensor \(static\) \(eV/Ang\*\*3\):\s*\n((?:siesta:[^\n]+\n){3})", text)
    stress = [floats(line) for line in stress_blocks[-1].splitlines()] if stress_blocks else None
    spins = re.findall(r"spin moment:\s*\{S\}\s*,\s*\|S\|\s*=\s*\{([^}]+)\}", text)
    spin = floats(spins[-1]) if spins else None
    # Collinear output reports Qup-Qdown, in Bohr magnetons.
    if spin is None:
        collinear = last(r"^\s* spin moment:\s*(" + NUMBER + ")")
        if collinear is not None:
            spin = [0., 0., collinear]
    # A verified spin-degenerate calculation has identically zero moments.
    # Do not interpret missing magnetic output as zero for other spin layouts.
    spin_components = last(r"^redata:\s+Number of spin components\s*=\s*(\d+)\s*$")
    configurations = re.findall(r"^redata:\s+Spin configuration\s*=\s*(\S+)", text, re.MULTILINE)
    spin_degenerate = spin_components == 1 and bool(configurations) and configurations[-1].lower() == 'none'
    if spin is None and spin_degenerate:
        spin = [0., 0., 0.]
    iterations = [int(i) for i in re.findall(r"^\s*scf:\s*(\d+)", text, re.MULTILINE)]
    converged = bool(re.search(r"SCF cycle converged", text)) and "SCF_NOT_CONV" not in text
    atomic_tables = re.findall(r"Mulliken Atomic Populations:\s*\n([^\n]+)\n((?:\s*\d+[^\n]+\n)+)", text)
    atomic_moments, atomic_charges = None, None
    if atomic_tables:
        header, table = atomic_tables[-1]
        rows = [floats(line.rsplit(None, 1)[0]) for line in table.splitlines()]
        atomic_charges = [row[1] for row in rows]
        if 'Sx' in header and all(len(row) == 7 for row in rows):
            atomic_moments = [row[4:7] for row in rows]
        elif 'Sz' in header and all(len(row) == 4 for row in rows):
            atomic_moments = [[0., 0., row[3]] for row in rows]
        elif spin_degenerate and 'valence' in header and 'Sz' not in header and all(len(row) == 3 for row in rows):
            atomic_moments = [[0., 0., 0.] for row in rows]
    return dict(total_energy_ev=energy, fermi_energy_ev=fermi, soc_energy_ev=soc,
                soc_energy_representation="included_in_nonlocal" if "Eso(nil)" in text else "split",
                forces_ev_ang=forces, stress_ev_ang3=stress, total_moment_muB=spin,
                scf_iterations=len(iterations), converged=converged,
                completed="Job completed" in text, atom_moments_muB=atomic_moments,
                atom_net_charges_e=atomic_charges,
                spin_components=int(spin_components) if spin_components is not None else None,
                spin_degenerate=spin_degenerate)

def read_eigenvalues(path):
    lines = Path(path).read_text().splitlines()
    ef = float(lines[0])
    nbands, nspin, nk = map(int, lines[1].split())
    tokens = " ".join(lines[2:]).split()
    if nspin not in (1, 2, 4, 8) or min(nbands, nk) < 1:
        raise ValueError("Unsupported EIG header")
    # SIESTA writes the density spin dimension (4/8) for a single spinor band list.
    stride = 1 + nbands * (nspin if nspin <= 2 else 1)
    if len(tokens) != nk * stride:
        raise ValueError(f"Invalid EIG dimensions in {path}")
    values = []
    for k in range(nk):
        if int(tokens[k*stride]) != k+1:
            raise ValueError("Unexpected EIG k-point order")
        values.append([float(v) for v in tokens[k*stride+1:(k+1)*stride]])
    return {"fermi_ev": ef, "nbands": nbands, "nspin": nspin, "nk": nk, "values_ev": values}

def grid_artifact(path):
    """Locate a nominal RHO/DRHO field in its upstream-selected file format."""
    path = Path(path)
    if path.is_file():
        return path
    netcdf = path.with_suffix('.nc')
    if path.suffix in ('.RHO', '.DRHO') and netcdf.is_file():
        from netCDF4 import Dataset
        with Dataset(netcdf) as dataset:
            name='Rho' if path.suffix=='.RHO' else 'RhoDelta'
            if 'GRID' in dataset.groups and name in dataset.groups['GRID'].variables:
                return netcdf
    raise FileNotFoundError(path)


def read_grid(path):
    """SIESTA binary RHO: Fortran records, FP64 cell, int32 mesh, FP32 x-lines.

    CDF.Save stores the same requested fields in GRID/Rho and GRID/RhoDelta
    of the system NetCDF file. Preserve native precision and spin components.
    """
    path = Path(path)
    artifact = grid_artifact(path)
    if artifact != path:
        from netCDF4 import Dataset
        with Dataset(artifact) as dataset:
            variable = dataset.groups['GRID'].variables['Rho' if path.suffix == '.RHO' else 'RhoDelta']
            cell = dataset.variables['cell'][:]
            grid = variable[:]
            if (variable.dimensions != ('spin', 'nz', 'ny', 'nx') or
                    variable.getncattr('unit') != 'e/Bohr**3' or
                    dataset.variables['cell'].getncattr('unit') != 'Bohr' or
                    np.ma.is_masked(cell) or np.ma.is_masked(grid)):
                raise ValueError('Invalid or unwritten NetCDF density field')
            cell, grid = np.asarray(cell), np.asarray(grid)
            if cell.shape != (3, 3) or min(grid.shape)<1 or grid.shape[0] not in (1, 2, 4) or not np.isfinite(cell).all() or not np.isfinite(grid).all():
                raise ValueError('Invalid NetCDF density values or shape')
            return cell.copy(), grid.copy()
    with artifact.open("rb") as stream:
        first = stream.read(4)
        endian = "<" if first == struct.pack("<i", 72) else ">"
        stream.seek(0)
        def record():
            lead = stream.read(4)
            if len(lead) != 4:
                raise ValueError("Truncated grid record")
            size = struct.unpack(endian+"i", lead)[0]
            if size <= 0 or size > 2**28:
                raise ValueError("Unsupported grid record format")
            payload = stream.read(size)
            if len(payload) != size or stream.read(4) != lead:
                raise ValueError("Corrupt grid record")
            return payload
        cell = np.frombuffer(record(), dtype=endian+"f8").reshape(3, 3)
        nx, ny, nz, nspin = np.frombuffer(record(), dtype=endian+"i4")
        if min(nx, ny, nz, nspin) <= 0:
            raise ValueError("Invalid grid dimensions")
        grid = np.empty((nspin, nz, ny, nx), dtype=np.float32)
        for spin in range(nspin):
            for z in range(nz):
                for y in range(ny):
                    row = np.frombuffer(record(), dtype=endian+"f4")
                    if len(row) != nx:
                        raise ValueError("Incorrect grid row length")
                    grid[spin, z, y] = row
        if stream.read(1):
            raise ValueError("Unexpected trailing grid data")
    return cell.copy(), grid

def read_geometry(path):
    """Native XV contains lattice/positions in Bohr, velocities in atomic units."""
    lines = Path(path).read_text().splitlines()
    cell = [floats(row)[:3] for row in lines[:3]]
    count = int(lines[3])
    atoms = [floats(row) for row in lines[4:]]
    if len(atoms) != count or any(len(row) != 8 for row in atoms):
        raise ValueError("Invalid XV geometry")
    return dict(cell_bohr=cell, atomic_numbers=[int(row[1]) for row in atoms],
                positions_bohr=[row[2:5] for row in atoms])
