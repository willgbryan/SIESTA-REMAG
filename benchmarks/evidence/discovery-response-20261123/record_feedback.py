"""Record this measured discovery example using GPAC and TWINS' public APIs.

Run with GPAC/TWINS installed alongside REMAG. No GPU work is launched here.
The existing prospective parent/child predictions are in research.sqlite.
"""
from pathlib import Path
import hashlib
import json
from gpac import ir, parametric
from twins.memory import Notebook
from siesta_remag.magnetic.authoring import export_authored_structure, record_screen

ROOT = Path(__file__).resolve().parents[3]
AUTHORING = ROOT / 'results/discovery-authoring-20261123'
CHILD = ROOT / 'results/discovery-child-20261123T180908Z'


def main():
    receipt = AUTHORING / 'feedback-receipt.json'
    if receipt.exists():
        raise RuntimeError('This immutable outcome has already been recorded')
    links = json.loads((AUTHORING / 'notebook-links.json').read_text())
    screen_path = CHILD / 'screen-k16.json'
    screen = json.loads(screen_path.read_text())
    digest = hashlib.sha256(screen_path.read_bytes()).hexdigest()
    notebook = Notebook(AUTHORING / 'research.sqlite')
    events = record_screen(notebook, links['child_experiment'], screen, screen_path)
    orbital = screen['mft']['electronic_descriptors'][0]['orbital_response']
    mixing = next(p['near_fermi_co_weight'] for p in orbital['intersite_band_mixing']
                  if p['groups'] == ['orbit.0000.atom.0000:Fe:d', 'orbit.0001.atom.0000:Pt:d'])
    events.append(notebook.record_outcome(links['child_experiment'],
        observable='near_fermi_Fe_d_Pt_d_co_weight', value=mixing, unit='dimensionless',
        report_sha256=digest, status='measured', evidence=dict(report=str(screen_path))))
    conclusion = (
        'The c+2% fixed-volume child increased frozen Fx-Fz from 2.737267 to '
        '3.322907 meV/cell and Fe-d/Pt-d co-weight from 0.401946 to 0.431753. '
        'Both directional predictions were contradicted; neither observation '
        'establishes a causal mechanism. Fe d Fermi DOS decreased while Pt d DOS '
        'increased. Large angle-sensitive band-index shifts also contain substantial '
        'Fe/Pt p weight. At 8^3 the nearest Fe-Fe response rose from 4.150 to '
        '31.424 meV, but the parent changes from 28.927 to 4.150 meV between '
        '4^3 and 8^3, so that exchange trend is not mesh-converged. The '
        'simultaneous in-plane compression and c stretch should be separated next.'
    )
    events.append(notebook.record_analysis(links['child_experiment'],
        conclusion=conclusion, report_sha256=digest, status='evaluated'))
    child = ir.load_program((AUTHORING / 'child/program.xcl').read_bytes())
    changes = {'metric.g11': '74529/10000', 'metric.g22': '74529/10000'}
    following = parametric.instantiate(child, changes, expected_sha256=child.sha256)
    authored = export_authored_structure(following, AUTHORING / 'followup', parent=child, changes=changes)
    hypothesis = notebook.record_hypothesis(
        'Removing in-plane compression while retaining expanded c may reduce '
        'the enhanced Fe/Pt near-Fermi mixing and angular anisotropy of the child.',
        predictions=['Co-weight decreases relative to the measured child',
                     'Frozen Fx-Fz decreases relative to the measured child'],
        alternatives=['The c stretch dominates and both observables remain elevated',
                      'Band crossings or p/d redistribution reverse either trend'])
    experiment = notebook.propose_experiment(hypothesis,
        structure_sha256=authored['gpac_source_hash'],
        action='Restore a=b=2.73 Angstrom; keep c=3.8658 Angstrom, with identical electronic settings.',
        rationale='Separate the two strains in the rejected fixed-volume hypothesis; '
                  'retain p- and d-resolved descriptors rather than assuming a d-only mechanism.')
    notebook.record_prediction(experiment, observable='frozen_Fx_minus_Fz',
        value=.0030, unit='eV/cell', uncertainty=.0006)
    notebook.record_prediction(experiment, observable='near_fermi_Fe_d_Pt_d_co_weight',
        value=.410, unit='dimensionless', uncertainty=.040)
    result = dict(child_experiment=links['child_experiment'], outcome_events=events,
        prediction_evaluation=notebook.evaluate(links['child_experiment']),
        interpretation=conclusion, followup_hypothesis=hypothesis,
        followup_experiment=experiment, followup_source_hash=authored['gpac_source_hash'],
        followup_evaluated=False, mechanism_verified=False)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    notebook.close()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
