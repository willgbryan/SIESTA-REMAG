"""Paired measured responses, without inferring causation or comparability."""
import numpy as np
from .descriptors import SCREEN_SCHEMA


def compare_magnetic_responses(parent, child):
    if any(s.get('schema') != SCREEN_SCHEMA for s in (parent, child)):
        raise ValueError('Two REMAG-MAGNETIC-SCREEN/1 results required')
    reasons = []
    settings = parent.get('calculation_settings')
    if not settings or settings != child.get('calculation_settings'):
        reasons.append('Electronic calculation settings are missing or differ')
    for key in ('gpac_source_hash', 'evaluated_geometry_hash'):
        if not all(s.get('identity', {}).get(key) for s in (parent, child)):
            reasons.append(f'Exact {key} is unavailable')
    a, b = parent.get('site_identity'), child.get('site_identity')
    changes = None
    if not a or not b:
        reasons.append('Exact site-identity mapping is unavailable')
    else:
        before = {s['site_id']: s for s in a};after = {s['site_id']: s for s in b}
        if len(before) != len(a) or len(after) != len(b):
            raise ValueError('Duplicate site identities')
        changes = [dict(site_id=k, before=before.get(k), after=after.get(k))
                   for k in sorted(before.keys()|after.keys()) if before.get(k) != after.get(k)]
        if before.keys() != after.keys() or any(before[k].get('species') != after[k].get('species') for k in before.keys()&after.keys()):
            reasons.append('Composition differs; per-cell observables have different normalization')
    angular = None
    ma, mb = parent.get('mft'), child.get('mft')
    if ma and mb:
        if ma.get('approximation') != mb.get('approximation') or ma.get('energy_normalization') != mb.get('energy_normalization'):
            reasons.append('Angular methods or normalization differ')
        common = []
        for pa in ma.get('points', []):
            match = [pb for pb in mb.get('points', []) if np.linalg.norm(np.asarray(pa['direction'])-pb['direction']) < 1.e-10]
            if match:
                common.append((pa, match[0]))
        if len(common) >= 2:
            a0,b0=common[0]
            angular = [dict(direction=pa['direction'],
                parent_delta_free_ev=pa['delta_free_ev']-a0['delta_free_ev'],
                child_delta_free_ev=pb['delta_free_ev']-b0['delta_free_ev'],
                change_ev=(pb['delta_free_ev']-b0['delta_free_ev'])-(pa['delta_free_ev']-a0['delta_free_ev']))
                for pa,pb in common]
        else:
            reasons.append('Fewer than two common magnetization directions')
    else:
        reasons.append('Paired MFT response is unavailable')
    def change(key):
        x,y=parent['summary'].get(key),child['summary'].get(key)
        return None if x is None or y is None else y-x
    exchange_changes=None
    if parent.get('exchange') and child.get('exchange'):
        def pair_map(screen):
            return {(p['site_i'],p['site_j'],tuple(p['translation'])):p for p in screen['exchange'].get('pairs',[])}
        pa,pb=pair_map(parent),pair_map(child);exchange_changes=[]
        if parent['exchange'].get('convention')!=child['exchange'].get('convention'):
            reasons.append('Exchange sign or pair-counting conventions differ')
        for key in sorted(pa.keys()|pb.keys()):
            a,b=pa.get(key),pb.get(key)
            exchange_changes.append(dict(site_i=key[0],site_j=key[1],translation=list(key[2]),
                parent_isotropic_ev=None if a is None else a['isotropic_ev'],
                child_isotropic_ev=None if b is None else b['isotropic_ev'],
                isotropic_change_ev=None if a is None or b is None else b['isotropic_ev']-a['isotropic_ev'],
                tensor_change_ev=None if a is None or b is None else (np.asarray(b['exchange_tensor_ev'])-a['exchange_tensor_ev']).tolist(),
                parent_distance_angstrom=None if a is None else a['distance_angstrom'],
                child_distance_angstrom=None if b is None else b['distance_angstrom']))
    rare_changes=None
    if parent.get('rare_earth') and child.get('rare_earth'):
        ra,rb=parent['rare_earth'].get('summary'),child['rare_earth'].get('summary')
        if isinstance(ra,dict) and isinstance(rb,dict):
            rare_changes=[]
            for site in sorted(ra.keys()&rb.keys()):
                if isinstance(ra[site],dict) and isinstance(rb[site],dict):
                    x,y=ra[site].get('local_anisotropy_ev'),rb[site].get('local_anisotropy_ev')
                    rare_changes.append(dict(site_id=site,local_anisotropy_change_ev=None if x is None or y is None else y-x))
    orbital_changes=None
    if ma and mb:
        da=(ma.get('electronic_descriptors') or [{}])[0].get('orbital_response')
        db=(mb.get('electronic_descriptors') or [{}])[0].get('orbital_response')
        if da and db:
            aa={p['group']:p for p in da['projected_dos']};bb={p['group']:p for p in db['projected_dos']}
            co_a={tuple(p['groups']):p['near_fermi_co_weight'] for p in da['intersite_band_mixing']}
            co_b={tuple(p['groups']):p['near_fermi_co_weight'] for p in db['intersite_band_mixing']}
            orbital_changes=dict(projected_dos=[dict(group=k,change_states_per_ev_cell=bb[k]['fermi_projected_dos_states_per_ev_cell']-aa[k]['fermi_projected_dos_states_per_ev_cell']) for k in sorted(aa.keys()&bb.keys())],
                band_mixing=[dict(groups=list(k),co_weight_change=co_b[k]-co_a[k]) for k in sorted(co_a.keys()&co_b.keys())],
                interpretation='Measured PAO diagnostics; not an established causal mechanism')
    return dict(schema='REMAG-MAGNETIC-COMPARISON/1', parent=parent['identity'], child=child['identity'],
        structural_intervention=changes,
        geometry_change=dict(before=parent.get('geometry'), after=child.get('geometry')),
        magnetization_change_muB_cell=change('magnetization_muB_cell'),
        sampled_anisotropy_change_ev_cell=change('screening_anisotropy_ev_cell'),
        angular_response_change=angular, numerically_comparable=not reasons, comparability_limitations=reasons,
        changed_exchange_pathways=exchange_changes, changed_orbital_hybridization=orbital_changes, changed_soc_response=angular,
        changed_rare_earth_response=rare_changes,
        interpretation='Measured differences support hypotheses; they do not establish a causal mechanism.')
