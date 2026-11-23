"""Small GPAC geometry handoff and TWINS notebook adapter, no new registry."""
import hashlib
import json
from pathlib import Path
import numpy as np


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def export_authored_structure(program, output, *, parent=None, changes=None):
    """Execute a validated GPAC program and retain exact source/site identity.

    The evaluated Cartesian frame is the lower Cholesky factor of GPAC's
    rational Gram matrix, interpreted in Angstrom. It is a numerical export,
    not a replacement for the exact GPAC word. Geometry construction is not an
    electronic CPU solver. Optional edits use GPAC's own correspondence.
    """
    from gpac import ir,parametric
    p=ir.load_program(program.bytes if hasattr(program,'bytes') else program)
    execution=ir.execute(p)
    cell=np.linalg.cholesky(np.array(execution.gram,dtype=float))
    sites=[]
    for orbit in ir.orbit_members(p):
        for atom in orbit['atoms']:
            exact_site=execution.sites[atom['site_index']]
            position=np.asarray(exact_site.x,dtype=float)@cell
            sites.append(dict(site_id=atom['id'],atomic_number=exact_site.z,
                              fractional=[str(v) for v in exact_site.x],position_angstrom=position.tolist()))
    geometry=dict(cell_angstrom=cell.tolist(),positions_angstrom=[s['position_angstrom'] for s in sites],
                  atomic_numbers=[s['atomic_number'] for s in sites],site_ids=[s['site_id'] for s in sites],length_unit='angstrom')
    record=dict(gpac_source_hash=p.sha256,evaluated_geometry_hash=_digest(geometry),
        source=p.to_dict(),geometry=geometry,sites=sites,parent_source_hash=None,intervention=None,
        numerical_frame='lower Cholesky of exact Gram in Angstrom',relaxed=False)
    if parent is not None:
        before=ir.load_program(parent.bytes if hasattr(parent,'bytes') else parent)
        record.update(parent_source_hash=before.sha256,intervention=changes,
                      correspondence=parametric.correspondence(before,p,changes or {}))
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    (root/'program.xcl').write_bytes(p.bytes)
    (root/'structure.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def state_metadata(structure,hsx,*,calculation_id,settings,evidence):
    """Bind evaluated GPAC geometry and actual SIESTA PAO quantum numbers."""
    from .operators import BOHR_ANGSTROM
    geometry=structure['geometry']
    if (not np.allclose(hsx['cell'].reshape(3,3)*BOHR_ANGSTROM,geometry['cell_angstrom'],rtol=0,atol=2.e-8)
            or not np.allclose(hsx['positions'].reshape(-1,3)*BOHR_ANGSTROM,geometry['positions_angstrom'],rtol=0,atol=2.e-8)):
        raise ValueError('Electronic geometry differs from its GPAC handoff')
    orbitals=[];sites=[]
    for i,site in enumerate(structure['sites']):
        species=int(hsx['labels'][i]);raw=np.frombuffer(hsx['basis'][species],dtype='<i4').reshape(-1,3)
        # HSX lists every real spherical PAO in upstream order. Repeated
        # (n,l,zeta) rows progress m=-l..l within that radial shell.
        seen={}
        for n,l,zeta in raw:
            key=(int(n),int(l),int(zeta));m=seen.get(key,0)-int(l);seen[key]=seen.get(key,0)+1
            orbitals.append(dict(site_id=site['site_id'],n=int(n),l=int(l),m=m,zeta=int(zeta)))
        species_name={26:'Fe',78:'Pt'}.get(site['atomic_number'],str(site['atomic_number']))
        sites.append(dict(site_id=site['site_id'],species=species_name,atomic_number=site['atomic_number'],
                          position_angstrom=site['position_angstrom'],magnetic_role={26:'local_moment',78:'induced'}.get(site['atomic_number'],'unassigned')))
    return dict(calculation_id=calculation_id,reference_direction=[0,0,1],
        magnetic_terms=dict(xc='rigid_spin_rotation',hubbard='absent',constraint='absent',external_field='absent'),
        gpac_source_hash=structure['gpac_source_hash'],evaluated_geometry_hash=structure['evaluated_geometry_hash'],
        parent_source_hash=structure['parent_source_hash'],structural_intervention=structure['intervention'],
        physical_site_mapping=structure.get('physical_site_mapping'),
        geometry={k:geometry[k] for k in ('cell_angstrom','positions_angstrom')},sites=sites,orbitals=orbitals,
        settings=settings,evidence=evidence)


def record_screen(notebook,experiment_id,screen,report_path):
    """Append measured screening observables through TWINS' existing Notebook."""
    digest=hashlib.sha256(Path(report_path).read_bytes()).hexdigest();events=[]
    if screen.get('mft'):
        for p in screen['mft']['points']:
            if np.allclose(p['direction'],[1,0,0]):
                events.append(notebook.record_outcome(experiment_id,observable='frozen_Fx_minus_Fz',value=p['delta_free_ev'],
                    unit='eV/cell',report_sha256=digest,status='measured',evidence=dict(identity=screen['identity'],
                    approximation='frozen-potential screening',report=str(Path(report_path).resolve()))))
    events.append(notebook.record_analysis(experiment_id,conclusion=json.dumps(screen['summary'],allow_nan=False),
                                          report_sha256=digest,status='inconclusive'))
    return events


def prepare_authored_input(structure,template,output):
    """Reuse a same-composition Cartesian-Angstrom template for geometry edits."""
    import re,shutil
    if _digest(structure['geometry'])!=structure['evaluated_geometry_hash']:
        raise ValueError('Evaluated geometry hash changed')
    geometry=structure['geometry'];text=(Path(template)/'input.fdf').read_text()
    if not re.search(r'^LatticeConstant\s+1(?:\.0*)?\s+Ang\s*$',text,re.M|re.I) or not re.search(
            r'^AtomicCoordinatesFormat\s+NotScaledCartesianAng\s*$',text,re.M|re.I):
        raise ValueError('Template requires LatticeConstant 1 Ang and NotScaledCartesianAng coordinates')
    species_block=re.search(r'%block ChemicalSpeciesLabel\n(.*?)%endblock ChemicalSpeciesLabel',text,re.S)
    coordinates_block=re.search(r'%block AtomicCoordinatesAndAtomicSpecies\n(.*?)%endblock AtomicCoordinatesAndAtomicSpecies',text,re.S)
    if species_block is None or coordinates_block is None:raise ValueError('Explicit species and coordinate blocks required')
    species={int(row.split()[1]):int(row.split()[0]) for row in species_block[1].splitlines() if row.strip()}
    labels={v:k for k,v in species.items()}
    old_species=[labels[int(row.split()[-1])] for row in coordinates_block[1].splitlines() if row.strip()]
    if old_species!=geometry['atomic_numbers']:
        raise ValueError('Template must preserve the authored atom order and composition, including initial spins')
    root=Path(output);shutil.copytree(template,root)
    # FDF distinguishes integer tokens (species) from real coordinate tokens.
    # Scientific notation keeps exact zeros from being read as species indices.
    lattice='\n'.join(' '.join(format(v,'.16e') for v in row) for row in geometry['cell_angstrom'])
    coordinates='\n'.join(' '.join(format(v,'.16e') for v in row)+f' {species[z]}'
        for row,z in zip(geometry['positions_angstrom'],geometry['atomic_numbers'],strict=True))
    for block,content in [('LatticeVectors',lattice),('AtomicCoordinatesAndAtomicSpecies',coordinates)]:
        text=re.sub(r'(%block '+block+r'\n).*?(%endblock '+block+r')',lambda m:m[1]+content+'\n'+m[2],text,flags=re.S)
    (root/'input.fdf').write_text(text)
    manifest=json.loads((root/'manifest.json').read_text())
    manifest.update(input_sha256=hashlib.sha256(text.encode()).hexdigest(),gpac_source_hash=structure['gpac_source_hash'],
        evaluated_geometry_hash=structure['evaluated_geometry_hash'],parent_source_hash=structure['parent_source_hash'],
        structural_intervention=structure['intervention'],fixed_cell_volume_ang3=float(abs(np.linalg.det(geometry['cell_angstrom']))),
        orientation_family_sha256=None,purpose='authored-magnetic-screening')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return root
