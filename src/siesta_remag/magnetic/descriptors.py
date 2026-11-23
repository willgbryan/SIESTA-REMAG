"""Small model-facing summaries plus references to detailed scientific evidence."""
import numpy as np

SCREEN_SCHEMA = 'REMAG-MAGNETIC-SCREEN/1'


def summarize_magnetic_screen(state, *, mae=None, exchange=None, reduced=None, rare_earth=None):
    m = state.metadata if hasattr(state, 'metadata') else state
    if hasattr(reduced,'report'):
        reduced=reduced.report
    methods = dict(mft=mae, exchange=exchange, downfolding=reduced, rare_earth=rare_earth)
    missing = [name for name, value in methods.items() if value is None]
    for name, value in methods.items():
        if value is not None and value.get('status') not in ('screening', 'refined', 'qualified', 'incomplete'):
            raise ValueError(f'{name} is not a usable or explicitly incomplete method result')
    first = (mae.get('points') or [{}])[0] if mae else {}
    spin = first.get('spin_muB')
    electronic=None if mae is None else mae.get('electronic_descriptors')
    compact=None if electronic is None else [dict(direction_index=d['direction_index'],
        orbital_response=d.get('orbital_response'),
        k_anisotropy_hotspots=d.get('k_anisotropy_hotspots',[])[:3],
        near_fermi_states=[{k:v for k,v in s.items() if k!='orbital_weights'} for s in d.get('near_fermi_states',[])[:3]])
        for d in electronic]
    return dict(schema=SCREEN_SCHEMA, status='screening' if any(v and v.get('status') in ('screening','refined','qualified') for v in methods.values()) else 'incomplete',
        identity={key: m.get(key) for key in ('gpac_source_hash', 'evaluated_geometry_hash', 'calculation_id')},
        site_identity=m.get('sites'), geometry=m.get('geometry'), calculation_settings=m.get('settings'),
        summary=dict(magnetization_muB_cell=None if spin is None else float(np.linalg.norm(spin)),
            screening_anisotropy_ev_cell=None if mae is None else mae.get('sampled_barrier_ev'),
            preferred_sampled_direction=None if mae is None else mae.get('lowest_sampled_direction'),
            exchange_network=None if exchange is None else exchange.get('network'),
            principal_soc_orbital_contributions=compact,
            rare_earth_response=None if rare_earth is None else rare_earth.get('summary')),
        **methods, approximation_labels=[v.get('approximation') for v in methods.values() if v and v.get('approximation')],
        numerical_sensitivity={k: None if v is None else v.get('numerical_sensitivity') for k,v in methods.items()},
        missing_observables=missing,
        limitations=[item for value in methods.values() if value for item in value.get('limitations', [])],
        raw_evidence=list(m.get('evidence', []))+[item for value in methods.values() if value
                                               for item in value.get('raw_evidence', [])])
