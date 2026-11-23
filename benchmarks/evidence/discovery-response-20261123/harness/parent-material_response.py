from pathlib import Path
import json,shutil,re,time
import numpy as np
from siesta_remag.runner import run_case
from siesta_remag.hsx import read_hsx
from siesta_remag.magnetic import calculate_mft,calculate_exchange,summarize_magnetic_screen
from siesta_remag.magnetic.state import import_native_state
from siesta_remag.magnetic.authoring import state_metadata
from siesta_remag.magnetic.realspace import load_realspace
from siesta_remag.magnetic.gpu import cuda,cell_gauge


def run(root,volume):
    stage=Path('/opt/discovery-stage');plan=json.loads((stage/'plan.json').read_text());structure=json.loads((stage/'structure.json').read_text())
    def save(name,value):
        (root/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');volume.commit()
    inputs=root/'inputs';shutil.copytree(Path('/history')/plan['source_run']/'inputs'/plan['case'],inputs)
    content=(inputs/'input.fdf').read_text();geometry=structure['geometry']
    lattice='\n'.join(' '.join(format(v,'.16e') for v in row) for row in geometry['cell_angstrom'])
    coordinates='\n'.join(' '.join(format(v,'.16e') for v in row)+f' {i+1}' for i,row in enumerate(geometry['positions_angstrom']))
    content=re.sub(r'(%block LatticeVectors\n).*?(%endblock LatticeVectors)',lambda m:m[1]+lattice+'\n'+m[2],content,flags=re.S)
    content=re.sub(r'(%block AtomicCoordinatesAndAtomicSpecies\n).*?(%endblock AtomicCoordinatesAndAtomicSpecies)',lambda m:m[1]+coordinates+'\n'+m[2],content,flags=re.S)
    (inputs/'input.fdf').write_text(content)
    import hashlib
    manifest=json.loads((inputs/'manifest.json').read_text());manifest['input_sha256']=hashlib.sha256(content.encode()).hexdigest()
    (inputs/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'rotations.txt').write_text('.7071067811865476 0 .7071067811865476 0\n')
    env={**plan['settings'],'REMAG_INITIAL_MESH':'0','REMAG_FROZEN_ROTATIONS':str(root/'rotations.txt'),'REMAG_FROZEN_SOC_FREE':'1','REMAG_EXPORT_MAGNETIC_STATE':'1'}
    result=run_case(inputs,Path('/opt/magnetic-candidate-build/Src/siesta'),root/'reference',threads=1,timeout=180,extra_env=env)
    save('ledger.json',[dict(role='reference_scf',valid=result['valid'],seconds=result['wall_seconds'])])
    if not result['valid']:raise RuntimeError('Reference SCF failed: '+str(result.get('parse_errors')))
    metadata=state_metadata(structure,read_hsx(root/'reference/remag.HSX'),calculation_id=root.name+'/reference',
        settings=dict(material='FePt',basis='TZDP',mesh_cutoff_ry=400,kmesh=[4,4,4],temperature_k=300,reference='SOC-free'),evidence=[dict(path=str(root/'reference/result.json'))])
    state=import_native_state(root/'reference',metadata);state.save(root/'state');model=load_realspace(root/'reference',state.metadata)
    cp,_=cuda();original,k=cell_gauge(state);rebuilt=next(model.interpolate(cp.asnumpy(k),batch_size=64))[1]
    error=max(float(cp.max(cp.abs(a-rebuilt[name])).get()) for name,a in zip(('h_base','h_mag','h_soc','overlap'),original,strict=True))
    save('realspace-reconstruction.json',dict(max_abs_ry=error,translations=len(model.translations)))
    if error>1.e-8:raise RuntimeError('Real-space reconstruction differs from native sampled generator')
    del original,rebuilt
    progression=[];last=None
    for mesh in [4,8,12,16]:
        start=time.perf_counter();sample=model.sample([mesh]*3)
        axes=[[0,0,1],[1,0,0],[0,1,0]] if mesh!=16 else [[0,0,1],[.382683432365,0,.923879532511],[1,0,1],[.923879532511,0,.382683432365],[1,0,0],[0,1,0]]
        mft=calculate_mft(sample,axes,check_soc_free=mesh==4,memory_limit_bytes=8*1024**3,
                          angular_model='axial4' if mesh==16 else None,output=root/f'mft-k{mesh}')
        row=dict(mesh=mesh,seconds=time.perf_counter()-start,mft_seconds=mft['performance']['total_wall_seconds'],
                 delta_xz_ev=next(p['delta_free_ev'] for p in mft['points'] if np.allclose(p['direction'],[1,0,0])),
                 spin_muB=mft['points'][0]['spin_muB'])
        progression.append(row);save('kmesh-progression.json',progression)
        save(f'screen-k{mesh}.json',summarize_magnetic_screen(sample,mae=mft))
        if mesh in [4,8]:
            exchange=calculate_exchange(sample,radius_angstrom=2.8)
            save(f'exchange-k{mesh}.json',exchange)
            if mesh==4:
                old=json.loads(Path('/results/magnetic-material-response-20261123T172802Z/fept-exchange.json').read_text())
                # Same geometry/reference; small residual SCF differences are retained.
                differences=[abs(a['isotropic_ev']-b['isotropic_ev']) for a,b in zip(exchange['pairs'],old['pairs'],strict=True)]
                save('exchange-previous-comparison.json',dict(max_abs_ev=max(differences),old_seconds=old['performance']['total_wall_seconds'],new_seconds=exchange['performance']['total_wall_seconds']))
        last=mft;del sample
    save('structure.json',structure)
    return dict(progression=progression,physical_accuracy_qualified=False)
