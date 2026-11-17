"""Independent known-grid controls; no solver execution or native-data edits."""
import itertools
from pathlib import Path

import numpy as np
import pytest

from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load,save
from siesta_remag.research.sampling import native_mesh_identity,compare_sampling


def native_fixture(root,cell=None,*,bohr=.529177210903,mesh=(3,3,3)):
    root.mkdir();stage=root/'soc-z';stage.mkdir()
    cell=np.asarray(cell if cell is not None else [[2.73,.17,.09],[0,2.88,.21],[0,0,3.79]],float)
    indices=np.array(list(itertools.product(*[range(-(n//2),n-n//2) for n in mesh])))
    fractional=indices/np.array(mesh)
    cart=2*np.pi*fractional@np.linalg.inv((cell/bohr).T)
    # Six significant decimal digits, independently written and reread like KP.
    rows=[[str(i+1),*[f'{x:.5E}' for x in k],f'{1/len(cart):.5E}'] for i,k in enumerate(cart)]
    (stage/'remag.KP').write_text(str(len(rows))+'\n'+'\n'.join(' '.join(r) for r in rows)+'\n')
    (stage/'input.fdf').write_text('LatticeConstant 1 Ang\n%block LatticeVectors\n'+
        '\n'.join(' '.join(f'{x:.16f}' for x in row) for row in cell)+'\n%endblock LatticeVectors\n%block kgrid_Monkhorst_Pack\n'+
        '\n'.join(' '.join(str(mesh[i] if i==j else 0) for j in range(3))+' 0.0' for i in range(3))+'\n%endblock kgrid_Monkhorst_Pack\n')
    report=dict(request=dict(fixed_geometry=True,kmesh=list(mesh)),structure=dict(geometry=dict(cell=cell.tolist())),
        artifacts={'soc-z/'+n:sha256(stage/n) for n in ('remag.KP','input.fdf')})
    context=dict(report_path=root/'report.json',report=report,metadata={'kp_sha256':sha256(stage/'remag.KP')},stage='soc-z')
    arrays=read_arrays(context,bohr)
    return arrays,context,indices%np.array(mesh)


def read_arrays(context,bohr=.529177210903):
    p=Path(context['report_path']).parent/'soc-z/remag.KP'
    values=np.loadtxt(p,skiprows=1);cell=np.asarray(context['report']['structure']['geometry']['cell'])
    weights=values[:,4]/values[:,4].sum()
    return dict(k_fractional=values[:,1:4]@(cell/bohr).T/(2*np.pi),k_weights=weights)


def replace_rows(context,rows):
    p=Path(context['report_path']).parent/'soc-z/remag.KP'
    p.write_text(str(len(rows))+'\n'+'\n'.join(' '.join(row) for row in rows)+'\n')
    context['report']['artifacts']['soc-z/remag.KP']=context['metadata']['kp_sha256']=sha256(p)


@pytest.mark.parametrize('bohr,convention',[(.529177210903,'codata2018'),(.529177,'legacy')])
def test_skew_cell_mesh_corresponds_with_literal_rounding_proof(tmp_path,bohr,convention):
    a,ca,expected=native_fixture(tmp_path/'a',bohr=bohr)
    cell=np.asarray(ca['report']['structure']['geometry']['cell']);cell[2]*=1.005
    b,cb,_=native_fixture(tmp_path/'b',cell,bohr=bohr)
    original={key:arr.copy() for key,arr in b.items()}
    assert np.max(abs(a['k_fractional']-b['k_fractional']))>1e-12
    identity,proof=native_mesh_identity(a,ca)
    assert identity['ordered_grid_indices']==expected.tolist()
    assert identity['multiplicities']==[1]*27
    assert [v['length_convention'] for v in proof['admitted_length_conventions']]==[convention]
    r=compare_sampling(a,b,ca,cb)
    assert r['mode']=='native_KP_printing_intervals' and r['common_length_conventions']==[convention]
    assert r['before']['identity_sha256']==r['after']['identity_sha256']
    assert r['energy_error_bound'] is None and not r['coordinates_modified'] and not r['states_interpolated']
    assert all(np.array_equal(b[k],original[k]) for k in b)


def test_reduced_mesh_keeps_native_multiplicities(tmp_path):
    a,c,expected=native_fixture(tmp_path/'a',mesh=(3,1,1))
    # Native representation of a +/- pair with twice the Gamma weight.
    rows=(tmp_path/'a/soc-z/remag.KP').read_text().splitlines()[1:]
    rows=[row.split() for row in rows[:2]];rows[0][4]='6.66667E-1';rows[1][4]='3.33333E-1'
    replace_rows(c,rows);a=read_arrays(c)
    identity,proof=native_mesh_identity(a,c)
    assert identity['multiplicities']==[2,1] and proof['native_sample_count']==2 and proof['full_mesh_size']==3


@pytest.mark.parametrize('stage',['scalar','soc-z'])
def test_relaxed_mesh_uses_final_cell_not_authored_or_initial_cell(tmp_path,stage):
    a,c,expected=native_fixture(tmp_path/'relaxed')
    final=np.array(c['report']['structure']['geometry']['cell'])
    folder=tmp_path/'relaxed/soc-z'
    if stage=='scalar':
        folder.rename(folder.with_name('scalar'));folder=folder.with_name('scalar')
        c['report']['artifacts']={k.replace('soc-z/','scalar/'):v for k,v in c['report']['artifacts'].items()}
        p=folder/'input.fdf'
        p.write_text(p.read_text().replace('2.7300000000000000','3.0000000000000000'))
        c['report']['artifacts']['scalar/input.fdf']=sha256(p)
    c['stage']=stage;c['report']['format']='REMAG-DISCOVERY-OBSERVATION/1'
    c['report']['request']['fixed_geometry']=False
    c['report']['evaluated_geometry']=dict(cell=final.tolist(),atomic_numbers=[28])
    c['report']['structure']['geometry']['cell']=(final*1.1).tolist()
    struct=folder/'remag.STRUCT_OUT'
    struct.write_text('\n'.join(' '.join(map(str,row)) for row in final)+'\n1\n1 28 0 0 0\n')
    c['report']['artifacts'][stage+'/remag.STRUCT_OUT']=sha256(struct)
    identity,proof=native_mesh_identity(a,c)
    assert identity['ordered_grid_indices']==expected.tolist()
    assert proof['cell_origin']=='accepted native relaxation'
    c['report']['evaluated_geometry']['cell'][0][0]+=.1
    with pytest.raises(ValueError,match='relaxed geometry'):native_mesh_identity(a,c)


@pytest.mark.parametrize('mutation',[
    'off_grid','duplicate','coarse','bad_weight','missing_array','changed_array','changed_array_weight',
    'file_hash','metadata_hash','source_cell','shifted_input','input_mesh','native_count','unknown_units',
])
def test_native_proof_rejects_unresolved_or_unbound_sampling(tmp_path,mutation):
    a,c,_=native_fixture(tmp_path/'a',bohr=.6 if mutation=='unknown_units' else .529177210903)
    root=tmp_path/'a/soc-z';rows=[r.split() for r in (root/'remag.KP').read_text().splitlines()[1:]]
    if mutation=='off_grid':rows[0][1]=f'{float(rows[0][1])+.0002:.5E}'
    if mutation=='duplicate':rows[1][1:4]=rows[0][1:4]
    if mutation=='coarse':
        for row in rows:row[1:4]=['0E+0']*3
    if mutation=='bad_weight':
        rows[0][4]=f'{float(rows[0][4])+.001:.5E}';rows[1][4]=f'{float(rows[1][4])-.001:.5E}'
    if mutation in ('off_grid','duplicate','coarse','bad_weight'):
        replace_rows(c,rows);a=read_arrays(c)
    if mutation=='missing_array':a={k:v[:-1] for k,v in a.items()}
    if mutation=='changed_array':a['k_fractional'][0,0]+=.0001
    if mutation=='changed_array_weight':a['k_weights'][0]+=.00001
    if mutation=='file_hash':(root/'remag.KP').write_text('changed')
    if mutation=='metadata_hash':c['metadata']['kp_sha256']='f'*64
    if mutation=='source_cell':c['report']['structure']['geometry']['cell'][0][0]+=.1
    if mutation in ('shifted_input','input_mesh'):
        p=root/'input.fdf';p.write_text(p.read_text().replace('3 0 0 0.0','3 0 0 0.5' if mutation=='shifted_input' else '4 0 0 0.0'))
        c['report']['artifacts']['soc-z/input.fdf']=sha256(p)
    if mutation=='native_count':
        p=root/'remag.KP';p.write_text(p.read_text().replace('27\n','26\n',1));c['report']['artifacts']['soc-z/remag.KP']=c['metadata']['kp_sha256']=sha256(p)
    with pytest.raises(ValueError):native_mesh_identity(a,c)


def test_ordered_indices_and_units_cannot_change_between_endpoints(tmp_path):
    a,ca,_=native_fixture(tmp_path/'a');b,cb,_=native_fixture(tmp_path/'b',np.diag([2.73,2.88,3.81]))
    rows=[r.split() for r in (tmp_path/'b/soc-z/remag.KP').read_text().splitlines()[1:]][::-1]
    for i,row in enumerate(rows):row[0]=str(i+1)
    replace_rows(cb,rows);b=read_arrays(cb)
    with pytest.raises(ValueError,match='ordered mesh points'):compare_sampling(a,b,ca,cb)
    b,cb,_=native_fixture(tmp_path/'legacy',bohr=.529177)
    with pytest.raises(ValueError,match='length conventions'):compare_sampling(a,b,ca,cb)
    assert compare_sampling(a,a)['mode']=='literal_ordered_arrays'
    with pytest.raises(ValueError,match='context unavailable'):compare_sampling(a,b)


def test_series_and_prediction_outcomes_share_native_mesh_correspondence(tmp_path):
    # Reuse independently validated synthetic native acceptance and exact GPAC
    # family fixtures; replace only their explicitly synthetic spectra.
    pytest.importorskip('gpac')
    from test_research import experiment
    from test_research_series import measured
    from test_research_inspection import spectrum_fixture
    from siesta_remag.research.series import analyze_series
    from siesta_remag.research.evaluation import response_observation
    bundle=experiment.__wrapped__(tmp_path)
    series,evidence=measured(bundle,tmp_path)
    for point in series['points']:
        endpoint=evidence[point['id']];rp=endpoint['report'];report=load(rp)
        arrays,context,_=native_fixture(tmp_path/(point['id']+'-native'),report['structure']['geometry']['cell'])
        source=Path(context['report_path']).parent/'soc-z/remag.KP';dest=rp.parent/'soc-z/remag.KP'
        dest.write_bytes(source.read_bytes());report['artifacts']['soc-z/remag.KP']=sha256(dest)
        for suffix in ('HSX','EIG'):
            p=rp.parent/'soc-z'/('remag.'+suffix);p.write_text('synthetic '+suffix);report['artifacts']['soc-z/'+p.name]=sha256(p)
        save(rp,report)
        meta=spectrum_fixture(tmp_path/(point['id']+'-spectrum'));m=load(meta)
        n=len(arrays['k_weights']);q=float(point['q_exact'].split('/')[0])/float(point['q_exact'].split('/')[1]) if '/' in point['q_exact'] else 0
        np.savez(meta.parent/'spectrum.npz',**arrays,eigenvalues_ev=np.tile([-.1+q,.2+q],(n,1)),occupations=np.tile([1,0],(n,1)),
            orbital_mulliken=np.tile([[1.1,.2],[-.1,.8]],(n,1,1)),spin_pauli=np.zeros((n,2,3)))
        m.update(artifact_sha256=sha256(meta.parent/'spectrum.npz'),**{key:report['artifacts']['soc-z/remag.'+suffix] for key,suffix in
            [('hsx_sha256','HSX'),('kp_sha256','KP'),('eig_sha256','EIG')]});save(meta,m)
        endpoint['spectra']={'response':meta}
    definition=dict(kind='spectral',stage='soc-z',spin=0,metric='projected_energy_moment',selection={'atom_indices':[1],'energy_window_ev':[-.2,.4]})
    r=analyze_series(tmp_path/'series',evidence,{'response':definition},tmp_path/'analysis')['observables']['response']
    assert r['derivatives'][0]['derivative']['first']==pytest.approx(.7)
    assert r['points'][1]['evidence']['sampling_correspondence']['mode']=='native_KP_printing_intervals'
    endpoints=[dict(report=e['report'],spectrum=e['spectra']['response']) for e in evidence.values()]
    target=definition|dict(kind='spectral_response',baseline_source_sha256=series['points'][0]['source_sha256'])
    result=response_observation(target,endpoints[0],endpoints[2],source_sha256=series['points'][2]['source_sha256'])
    assert result['value']==pytest.approx(.03*.7)
    assert result['evidence']['sampling_correspondence']['mode']=='native_KP_printing_intervals'
    from siesta_remag.research.inspection import compare_electronic
    native=[dict(report=e['report'],stage='soc-z') for e in endpoints]
    with pytest.raises(ValueError,match='context unavailable'):
        compare_electronic(endpoints[0]['spectrum'],endpoints[2]['spectrum'])
    with pytest.raises(ValueError,match='both native'):
        compare_electronic(endpoints[0]['spectrum'],endpoints[2]['spectrum'],before_native=native[0])
    compared=compare_electronic(endpoints[0]['spectrum'],endpoints[2]['spectrum'],before_native=native[0],after_native=native[2],points=5)
    assert compared['sampling_correspondence']['mode']=='native_KP_printing_intervals'
    assert compared['native_contexts']['before']['source_sha256']==series['points'][0]['source_sha256']
    assert len(compared['delta_dos'])==5 and not compared['mechanism_verified']
    from siesta_remag.research.tools import ResearchTools
    args=dict(before=str(endpoints[0]['spectrum'].relative_to(tmp_path)),after=str(endpoints[2]['spectrum'].relative_to(tmp_path)),points=5,
        before_native=dict(report=str(endpoints[0]['report'].relative_to(tmp_path)),stage='soc-z'),
        after_native=dict(report=str(endpoints[2]['report'].relative_to(tmp_path)),stage='soc-z'))
    assert ResearchTools(tmp_path).call('compare_electronic_states',args)==compared
    with pytest.raises(ValueError,match='held-out evaluator'):
        ResearchTools(tmp_path,withheld_sources=[series['points'][2]['source_sha256']]).call('compare_electronic_states',args)
    with pytest.raises(ValueError,match='escapes'):
        ResearchTools(tmp_path).call('compare_electronic_states',args|{'before_native':dict(report='../escape.json',stage='soc-z')})
    with pytest.raises(ValueError,match='native context stage'):
        compare_electronic(endpoints[0]['spectrum'],endpoints[2]['spectrum'],before_native=native[0]|{'stage':'soc-x'},after_native=native[2])
