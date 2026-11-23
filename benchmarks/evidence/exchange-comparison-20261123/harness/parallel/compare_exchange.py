"""One actual FePt exchange comparison; no independent numerical solver."""
from pathlib import Path
import json,time,sys,hashlib,importlib.metadata
import numpy as np
from ase import Atoms
from HamiltonIO.lcao_hamiltonian import LCAOHamiltonian
import TB2J.exchangeCL2 as tb
from TB2J.green import TBGreen
from siesta_remag.magnetic import load_magnetic_state,load_realspace,calculate_exchange
from siesta_remag.magnetic.operators import RY_EV
from siesta_remag.magnetic.realspace import uniform_mesh
from types import SimpleNamespace

root=Path(sys.argv[1]);source=Path('/results/discovery-response-20261123T175305Z')
def save(name,value):
    (root/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

training=load_magnetic_state(source/'state');model=load_realspace(source/'reference',training.metadata)
previous=Path('/results/exchange-comparison-20261123T194205Z')
state=SimpleNamespace(metadata=training.metadata,kpoints=uniform_mesh([9,9,9]))
interpolation=None
gpu=json.loads((previous/'remag-first.json').read_text())
warm=json.loads((previous/'remag-warm.json').read_text())
mu=gpu['pole_approximation'][0]['chemical_potential_ev'];temperature=gpu['pole_approximation'][0]['temperature_ev']
positive_poles=gpu['pole_approximation'][0]['poles']
m=state.metadata
atoms=Atoms(numbers=[s['atomic_number'] for s in m['sites']],positions=m['geometry']['positions_angstrom'],
            cell=m['geometry']['cell_angstrom'],pbc=True)
labels=[]
sites={s['site_id']:i for i,s in enumerate(m['sites'])}
for o in m['orbitals']:
    sym=m['sites'][sites[o['site_id']]]['species']
    labels.append(f"{sym}1|{o['n']}{'spdf'[o['l']]}m{o['m']}Z{o['zeta']}")
def canonical(coeff):
    # Preserve SIESTA/REMAG's lower-triangle convention in the transported data.
    inverse={tuple(r):i for i,r in enumerate(model.translations)}
    opposite=coeff[[inverse[tuple(-r)] for r in model.translations]]
    lower=np.tril(coeff,-1)
    result=lower+np.tril(opposite,-1).swapaxes(-1,-2).conj()
    ii=np.arange(coeff.shape[-1])
    result[:,ii,ii]=.5*(coeff[:,ii,ii]+opposite[:,ii,ii].conj())
    return result
h=canonical(model.coefficients['h_base']+model.coefficients['h_mag']);s=canonical(model.coefficients['overlap'])
models=[LCAOHamiltonian(HR=h[:,spin::2,spin::2]*RY_EV,SR=s[:,spin::2,spin::2],Rlist=model.translations,
    nbasis=48,orbs=labels,atoms=atoms,nspin=1,nel=m['electrons']/2) for spin in (0,1)]
checks=json.loads((previous/'input-matrix-checks.json').read_text())

class FullBasisGreen(TBGreen):
    def _reduce_eigens(self,evals,evecs,emin,emax):
        # TB2J otherwise silently removes bands above EF+5.1 eV. Retaining the
        # supplied full basis is the sole numerical-setting override.
        return evals,evecs

results=[]
for label,green in [('matched_full_basis_8_workers',FullBasisGreen)]:
    tb.TBGreen=green
    start=time.perf_counter()
    exchange=tb.ExchangeCL2(models,atoms,efermi=mu,smearing=temperature,basis=labels,
        magnetic_elements=['Fe','Pt'],kmesh=[9,9,9],nz=2*positive_poles,Rcut=2.8,nproc=8,
        integration_method='CFR',use_cache=False,output_path=str(root/label))
    setup=time.perf_counter()-start
    if np.max(abs(exchange.Gup.kpts-state.kpoints))>1.e-12:raise RuntimeError('k meshes differ')
    before=time.perf_counter();exchange.calculate_all();calculation=time.perf_counter()-before
    rows=[]
    for pair in gpu['pairs']:
        key=(tuple(pair['translation']),pair['i'],pair['j'])
        actual=float(exchange.exchange_Jdict[key])
        # TB2J's own io_espins.py specifies x2 for a sum over unique pairs.
        rows.append(dict(site_i=pair['site_i'],site_j=pair['site_j'],translation=pair['translation'],
            remag_ev=pair['isotropic_ev'],tb2j_ordered_pair_ev=actual,tb2j_unique_pair_ev=2*actual,
            difference_ev=2*actual-pair['isotropic_ev'],independent_spin_pair=pair['independent_spin_pair']))
    result=dict(label=label,setup_seconds=setup,calculation_seconds=calculation,total_seconds=setup+calculation,
        retained_bands=[exchange.Gup.evals.shape[1],exchange.Gdn.evals.shape[1]],pairs=rows,
        max_abs_error_ev=max(abs(p['difference_ev']) for p in rows),
        rms_error_ev=float(np.sqrt(np.mean([p['difference_ev']**2 for p in rows]))),
        atom_spin_muB=exchange.spinat.tolist(),directed_pair_count=len(exchange.exchange_Jdict),
        kpoints=len(state.kpoints),contour_points=len(exchange.contour.path),nproc=8,blas_threads=1)
    results.append(result);save(label+'.json',result)
    if setup+calculation>240:break

save('comparison.json',dict(status='complete',source_run=source.name,mesh=[9,9,9],radius_angstrom=2.8,
    chemical_potential_ev=mu,temperature_ev=temperature,positive_poles=positive_poles,
    input_matrix_checks=checks,remag_interpolation_seconds=interpolation,
    remag_first_seconds=gpu['performance']['total_wall_seconds'],remag_warm_seconds=warm['performance']['total_wall_seconds'],
    tb2j=results,package_versions={n:importlib.metadata.version(n) for n in ('TB2J','HamiltonIO','numpy','scipy','cupy-cuda12x')},
    tb2j_commit='4d7a65bd7a2b582f7ede53e9fec88efe8d6895ed',
    convention='E=-sum_unique_pairs e_i.J_ij.e_j; TB2J ordered-pair values multiplied by two',
    scope='Same frozen SOC-free FePt native generator, nonorthogonal overlap, onsite fields and sampling; Pt channels are induced responses.',
    accuracy_pass=max(abs(p['difference_ev']) for p in results[0]['pairs'])<1.e-7))
