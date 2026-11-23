from pathlib import Path
import json,time
import numpy as np
from siesta_remag.runner import run_case
from siesta_remag.hsx import read_hsx
from siesta_remag.magnetic import calculate_mft,calculate_exchange,summarize_magnetic_screen,load_magnetic_state,contract_radial_manifolds,compare_magnetic_responses
from siesta_remag.magnetic.state import import_native_state
from siesta_remag.magnetic.authoring import state_metadata,prepare_authored_input
from siesta_remag.magnetic.realspace import load_realspace


def run(root,volume):
    stage=Path('/opt/discovery-stage');plan=json.loads((stage/'plan.json').read_text());structure=json.loads((stage/'structure.json').read_text())
    def save(name,value):
        (root/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');volume.commit()
    save('structure.json',structure)
    inputs=prepare_authored_input(structure,Path('/history')/plan['source_run']/'inputs'/plan['case'],root/'inputs')
    (root/'rotations.txt').write_text('.7071067811865476 0 .7071067811865476 0\n')
    env={**plan['settings'],'CUDA_MODULE_LOADING':'EAGER','REMAG_INITIAL_MESH':'0','REMAG_FROZEN_ROTATIONS':str(root/'rotations.txt'),'REMAG_FROZEN_SOC_FREE':'1','REMAG_EXPORT_MAGNETIC_STATE':'1'}
    result=run_case(inputs,Path('/opt/magnetic-candidate-build/Src/siesta'),root/'reference',threads=1,timeout=180,extra_env=env)
    save('ledger.json',[dict(role='child_reference_scf',valid=result['valid'],seconds=result['wall_seconds'])])
    if not result['valid']:
        import subprocess,os
        with (root/'reference/input.fdf').open('rb') as stdin, (root/'startup-debug.log').open('w') as stdout:
            subprocess.run(['gdb','--batch','-ex','run','-ex','thread apply all bt','-ex','info sharedlibrary','--args','/opt/magnetic-candidate-build/Src/siesta'],cwd=root/'reference',env={**os.environ,**env},stdin=stdin,stdout=stdout,stderr=subprocess.STDOUT,timeout=60)
        volume.commit()
        raise RuntimeError('Child reference SCF failed; see startup-debug.log')
    metadata=state_metadata(structure,read_hsx(root/'reference/remag.HSX'),calculation_id=root.name+'/reference',
        settings=dict(material='FePt',basis='TZDP',mesh_cutoff_ry=400,kmesh=[4,4,4],temperature_k=300,reference='SOC-free'),evidence=[dict(path=str(root/'reference/result.json'))])
    state=import_native_state(root/'reference',metadata);state.save(root/'state');model=load_realspace(root/'reference',state.metadata)
    parent=Path('/results/discovery-response-20261123T175305Z');progress=[]
    for mesh in [8,16]:
        start=time.perf_counter();sample=model.sample([mesh]*3)
        axes=[[0,0,1],[1,0,0],[0,1,0]] if mesh==8 else [[0,0,1],[.382683432365,0,.923879532511],[1,0,1],[.923879532511,0,.382683432365],[1,0,0],[0,1,0]]
        mft=calculate_mft(sample,axes,check_soc_free=False,memory_limit_bytes=8*1024**3,
            angular_model='axial4' if mesh==16 else None,output=root/f'mft-k{mesh}')
        exchange=calculate_exchange(sample,radius_angstrom=2.8) if mesh==8 else None
        if exchange:save('exchange-k8.json',exchange)
        screen=summarize_magnetic_screen(sample,mae=mft,exchange=exchange);save(f'screen-k{mesh}.json',screen)
        before=json.loads((parent/f'screen-k{mesh}.json').read_text())
        if mesh==8:before['exchange']=json.loads((parent/'exchange-k8.json').read_text())
        comparison=compare_magnetic_responses(before,screen);save(f'parent-child-k{mesh}.json',comparison)
        progress.append(dict(mesh=mesh,seconds=time.perf_counter()-start,mft_seconds=mft['performance']['total_wall_seconds'],
            delta_xz_ev=next(p['delta_free_ev'] for p in mft['points'] if np.allclose(p['direction'],[1,0,0]))))
        save('kmesh-progression.json',progress);del sample
    # A genuine local radial contraction, trained once on the parent's 4^3 state.
    training=load_magnetic_state(parent/'state');original=load_realspace(parent/'reference',training.metadata)
    reduced,report=contract_radial_manifolds(original,training,radial_rank=2)
    np.savez(root/'contracted-realspace.npz',translations=reduced.translations,**reduced.coefficients)
    save('contracted-metadata.json',reduced.metadata)
    sub=reduced.sample([8]*3)
    mft=calculate_mft(sub,[[0,0,1],[1,0,0],[0,1,0]],check_soc_free=False,output=root/'contracted-mft')
    exchange=calculate_exchange(sub,radius_angstrom=2.8)
    before=json.loads((parent/'mft-k8/mft.json').read_text());old_exchange=json.loads((parent/'exchange-k8.json').read_text())
    report['response_error']=dict(mft_max_abs_ev=max(abs(a['delta_free_ev']-b['delta_free_ev']) for a,b in zip(mft['points'],before['points'],strict=True)),
        exchange_max_abs_ev=max(abs(a['isotropic_ev']-b['isotropic_ev']) for a,b in zip(exchange['pairs'],old_exchange['pairs'],strict=True)))
    report['mft_performance']=mft['performance'];report['exchange_performance']=exchange['performance']
    report['screening_use']='retain_full_pao' if report['response_error']['mft_max_abs_ev']>1.e-4 else 'provisional_candidate'
    save('contraction.json',report);save('contracted-exchange.json',exchange)
    # Warm call separates reusable-engine throughput from first CuPy compilation.
    warm=calculate_exchange(training,radius_angstrom=2.8);save('parent-exchange-warm-k4.json',warm)
    return dict(progression=progress,contraction=report,physical_accuracy_qualified=False)
