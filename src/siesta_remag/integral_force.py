"""Audit shared terminal force publication; upstream numerical agreement is external."""
import math


def record_errors(records, grids, tables, pairs, nonlocal_geometry, epochs, atoms, soc, gpu_history=False, retained_h0=False, compact_queries=False):
    errors = []
    pair_rows = pairs.get('cycles', [])
    nonlocal_rows = nonlocal_geometry.get('cycles', [])
    for name, rows in (('publication', records), ('grid', grids), ('pair geometry', pair_rows), ('nonlocal geometry', nonlocal_rows)):
        if not rows or [r.get('epoch') for r in rows] != epochs:
            errors.append('Integral force '+name+' does not cover every electronic geometry')
    if errors:
        return errors
    if pairs.get('complete') is not True or nonlocal_geometry.get('complete') is not True:
        errors.append('Integral geometry lease is incomplete')
    table_rows = tables.get('tables', [])
    if [r.get('kind') for r in table_rows] != [1, 2] or [r.get('uses') for r in table_rows] != [2*len(epochs), len(epochs)]:
        errors.append('Integral two-center table usage differs')
    if tables.get('comparison_performed') is not False or tables.get('cycles') != []:
        errors.append('Production integral tables claim shadow comparisons')
    for t in table_rows:
        shape=t.get('shape',[])
        if len(shape)!=11 or any(type(v) is not int or v<0 for v in shape):
            errors.append('Invalid integral table shape');continue
        nf,nr,nt,na,nb,_,_,lo1,hi1,lo2,hi2=shape
        if not (0<nf<=1000000 and nr<=1000000 and nt<=100000000 and na>0 and nb>0 and na*nb<=100000000 and
                1<=lo1<=hi1<=nf and 1<=lo2<=hi2<=nf):
            errors.append('Integral table dimensions or registry bounds invalid')
        staged=4*(nf+1+na*nb+1+2*nt)+8*(nf+2050*nr+nt)
        allocated=staged+(16 if nt==0 else 0)+(8 if nr==0 else 0)
        if staged>4*1024**3 or t.get('setup_bytes')!=staged or t.get('device_allocation_bytes')!=allocated:
            errors.append('Integral table staging/allocation differs')
        if (any(type(t.get(k)) not in (float,int) or not math.isfinite(t[k]) for k in ('delta','pi')) or
                t.get('delta',0)<=0 or not 3.14<t.get('pi',0)<3.15):
            errors.append('Invalid integral table constants')
    for r, g, p, n in zip(records, grids, pair_rows, nonlocal_rows):
        shape = g.get('grid_shape', [])
        nh, nq = g.get('spins'), p.get('queries')
        if len(shape) != 6 or any(type(v) is not int or v <= 0 for v in shape) or nh not in (1, 2, 4, 8) or type(nq) is not int or nq < 0:
            errors.append('Invalid integral force geometry');continue
        no, np, nsp, nuo, nnz, _ = shape
        if (nh == 8) != soc:
            errors.append('Integral spin/SOC selection differs')
        for name, count in (('energy', 10), ('force', 3*atoms), ('stress', 9)):
            v = r.get(name, [])
            if len(v) != count or any(type(x) not in (float, int) or not math.isfinite(x) for x in v):
                errors.append('Missing/nonfinite integral '+name)
        if any(r.get(k) != v for k, v in dict(atoms=atoms, status=0, export_bytes=4+8*(19+3*atoms), intermediate_export_bytes=0, derivative_reference_upload_bytes=0).items()):
            errors.append('Integral terminal status/export differs')
        if (g.get('atoms') != atoms or g.get('status') != 0 or g.get('export_bytes') != 0 or g.get('graph_nodes') != 2 or g.get('derivative_upload_bytes') != 0 or
                any(g.get(k) is not v for k, v in dict(orbital_basis_shadow=False, orbital_basis_device=True, atomic_basis_shadow=False, atomic_basis_device=True).items())):
            errors.append('Deferred grid execution or derivative mode differs')
        core = g.get('final_partial_core')
        atomic = g.get('atomic_terms', [])
        if core not in (0, 1) or [t.get('kind') for t in atomic] != ([1, 2] if core else [2]):
            errors.append('Integral atomic consumer coverage differs')
        for t in atomic:
            if (type(t.get('entries')) is not int or t['entries'] < 0 or t.get('points') != np*nsp or
                    t.get('diagonal_spins') != (min(2, nh) if t.get('kind') == 1 else 1) or
                    any(t.get(k) != v for k, v in dict(graph_nodes=2, setup_bytes=32, derivative_upload_bytes=0, host_derivative_storage_bytes=0).items())):
                errors.append('Integral atomic geometry/traffic differs')
        staged = 4*no+8*nuo+24+8*(19+3*atoms)+4+32*len(atomic)
        retained = 8*(nnz*(nh+min(nh, 4))+np*nsp*(min(nh, 4)+2)+9)+4*bool(gpu_history)
        if g.get('setup_bytes') != staged or g.get('retained_electronic_bytes') != retained:
            errors.append('Integral grid staging or retained fields differ')
        base = dict(atoms=atoms, rows=nuo, columns=no, capacity=nnz, consumer_mask=5, complete=True,
                    setup_bytes=48*nq+8*nuo+4*nnz, device_allocation_bytes=48*nq+8*nuo+4*nnz+(24 if nq == 0 else 0))
        if any(p.get(k) != v for k, v in base.items()):
            errors.append('Integral pair geometry/traffic differs')
        dims = [n.get(k) for k in ('centers', 'neighbors', 'blocks', 'queries')]
        if any(type(v) is not int or v < 0 for v in dims):
            errors.append('Invalid integral nonlocal dimensions');continue
        centers, neighbors, blocks, queries = dims
        geometry_bytes = 28*centers+16*neighbors+24*blocks+8*nuo+4*nnz+32*queries
        allocation = geometry_bytes+(28 if centers == 0 else 0)+(16 if neighbors == 0 else 0)+(24 if blocks == 0 else 0)+(12 if queries == 0 else 0)
        if n.get('compact_queries',False) is not compact_queries:
            errors.append('Nonlocal query capture mode differs')
        if compact_queries:
            projectors=n.get('projector_entries')
            if not retained_h0 or type(projectors) is not int or projectors<centers:
                errors.append('Compact query projector coverage missing');continue
            metadata=4*(centers+1+projectors)+28*neighbors
            if n.get('query_metadata_bytes')!=metadata or n.get('query_output_bytes')!=32*queries:
                errors.append('Compact query metadata/output inventory differs')
            geometry_bytes+=metadata-32*queries
            allocation+=metadata+(4 if projectors==0 else 0)+(12 if neighbors==0 else 0)
        if any(n.get(k) != v for k, v in dict(atoms=atoms, rows=nuo, columns=no, capacity=nnz, components=nh, consumer=3 if soc else 2,
                geometry_bytes=geometry_bytes, device_allocation_bytes=allocation, setup_bytes=0, export_bytes=0, electronic_upload_bytes=0,
                status=0, complete=True, passed=True, initial=[], reference=[], actual=[]).items()):
            errors.append('Integral nonlocal geometry, status or traffic differs')
        volume = p.get('volume')
        if type(volume) not in (float, int) or not math.isfinite(volume) or volume <= 0 or n.get('volume') != volume:
            errors.append('Integral geometry volume differs')
        consumers = r.get('consumers', [])
        expected = [dict(consumer=k, queries=q, graph_nodes=(2 if q else 1) if k in (1, 4) else (2 if retained_h0 else (3 if q else 2)),
                         setup_bytes=16 if k in (1, 4) else 0,
                         workspace_bytes=(0 if retained_h0 and k not in (1,4) else 8*max(1, 4*q))+(16 if k in (1, 4) else 8*nnz))
                    for k, q in ((1, nq), (3 if soc else 2, queries), (4, nq))]
        if retained_h0:expected[1]['retained_integrals']=True
        if consumers != expected:
            errors.append('Integral execution sequence, query count or workspace differs')
    return errors
