"""Independent raw-output checks; implementation gates are separate from method bias."""
import json,math
from pathlib import Path
import numpy as np
from siesta_remag.frozen_reference import direction_checks
RY_EV=13.605693122994

def probe(path,scf_result,nk,spin,directions):
    path=Path(path);report=json.loads((path/'remag-frozen-potential.json').read_text())
    soc_free=scf_result.get('runtime_settings',{}).get('REMAG_FROZEN_SOC_FREE')=='1'
    n=report['bands'];points=report['points'];errors=[];checks=[];raw=[]
    def check(name,ok,value=None):
        checks.append(dict(name=name,passed=bool(ok),value=value))
        if not ok:errors.append(name)
    check('completed_native_probe',report['status']=='complete' and bool(points))
    check('actual_mesh',report['kpoints']==nk)
    check('spin_mode',report['spin_expectations']==spin)
    check('generator_spectrum',report['generator_spectrum_max_abs_error_ry'] is not None and report['generator_spectrum_max_abs_error_ry']*RY_EV<=1e-7,
          report['generator_spectrum_max_abs_error_ry'])
    check('reference_soc_scale',report.get('reference_soc_scale',1)==int(not soc_free))
    check('target_soc_scale',report.get('target_soc_scale',1)==1)
    alignment=direction_checks(report,directions)
    check('intended_target_directions',alignment['passed'])
    control=None
    if soc_free:
        c=report.get('reference_control') or {}
        check('reference_control_native',c.get('completed') is True and c.get('status')==0 and c.get('soc_scale')==0 and c.get('eigensystem_jobs')==nk)
        control=np.genfromtxt(path/c['raw_spectrum'],delimiter=',',names=True)
        check('reference_control_shape',control.shape==(nk*n,))
        if control.shape!=(nk*n,):return dict(passed=False,checks=checks,errors=errors,report=report,directions=alignment)
        check('reference_control_layout',np.array_equal(control['k'],np.repeat(np.arange(nk),n)) and np.array_equal(control['band'],np.tile(np.arange(n),nk)))
        error=float(np.max(abs(control['energy_ry']-control['generator_scf_energy_ry'])))
        check('independent_reference_identity',np.isfinite(error) and error*RY_EV<=1e-7,error*RY_EV)
        check('reference_identity_receipt',abs(error-report['generator_spectrum_max_abs_error_ry'])<=1e-14)
    for p in points:
        check(f"point_{p['index']}_native",p['completed'] and p['status']==0 and p['electron_count_passed'] and p['eigensystem_jobs']==nk)
        a=np.genfromtxt(path/p['raw_spectrum'],delimiter=',',names=True)
        check(f"point_{p['index']}_shape",a.shape==(nk*n,))
        if a.shape!=(nk*n,):continue
        check('paired_layout',np.array_equal(a['k'],np.repeat(np.arange(nk),n)) and np.array_equal(a['band'],np.tile(np.arange(n),nk)))
        e,q,w=[a[key].astype(np.longdouble) for key in ('energy_ry','occupation','k_weight')]
        check('finite_occupied_states',np.isfinite(e).all() and np.isfinite(q).all() and np.isfinite(w).all() and np.all(w>0) and np.all(q>=0) and np.all(q<=w))
        check('normalized_weights',abs(float(w[::n].sum()-1))<=1e-12 and np.all(w.reshape(nk,n)==w[::n,None]))
        if control is not None:
            check('same_scf_generator_spectrum',np.array_equal(a['generator_scf_energy_ry'],control['generator_scf_energy_ry']))
        raw.append((e,q,w,a))
        ea,qa,wa,_=raw[0]
        check('common_weights',np.array_equal(w,wa))
        ea0=ea.astype(np.longdouble);eb0=e.astype(np.longdouble)
        de=np.sum(q*eb0-qa*ea0,dtype=np.longdouble)
        def entropy(q):
            probability=q/w
            occupied=np.maximum(probability,np.longdouble('1e-15'));empty=np.maximum(1-probability,np.longdouble('1e-15'))
            return -(occupied*np.log(occupied)+empty*np.log(empty))
        ds=np.sum(w*(entropy(q)-entropy(qa)),dtype=np.longdouble)
        df=de-np.longdouble(report['temperature_ry'])*ds
        na=qa.sum(dtype=np.longdouble);nb=q.sum(dtype=np.longdouble);target=np.longdouble(report['electrons'])
        canonical=df-np.longdouble(p['fermi_ry'])*(nb-target)+np.longdouble(points[0]['fermi_ry'])*(na-target)
        for name,value in [('band_difference_ry',de),('band_free_difference_ry',df),('canonical_free_difference_ry',canonical)]:
            error=float(abs(value-np.longdouble(p[name])))
            check('raw_'+name,error<=2e-11,error)
        check('independent_electron_count',abs(float(na-target))<=1e-10 and abs(float(nb-target))<=1e-10)
        if spin:
            vector=np.array([np.sum(q*a['spin_'+c].astype(np.longdouble)) for c in 'xyz'],dtype=float)
            check('raw_spin',np.max(abs(vector-p['spin_muB']))<=1e-10)
            initial=np.array(scf_result['observables']['total_moment_muB'],dtype=float)
            wq,x,y,z=np.array(p['quaternion'])/np.linalg.norm(p['quaternion'])
            rotation=np.array([[1-2*(y*y+z*z),2*(x*y-wq*z),2*(x*z+wq*y)],
                              [2*(x*y+wq*z),1-2*(x*x+z*z),2*(y*z-wq*x)],
                              [2*(x*z-wq*y),2*(y*z+wq*x),1-2*(x*x+y*y)]])
            expected=rotation@initial
            norm=float(np.linalg.norm(vector)*np.linalg.norm(expected))
            angle=None
            if np.isfinite(norm) and norm>0:
                cosine=float(np.dot(vector,expected)/norm)
                angle=math.degrees(math.acos(np.clip(cosine,-1,1)))
            check('final_direction_degrees',angle is not None and angle<=1.0,angle)
        if p['index']==0:
            check('identity_band_difference',abs(p['band_difference_ry']*RY_EV)<=5e-6)
            if spin and not soc_free:check('identity_scf_spin',np.max(abs(np.array(p['spin_muB'])-scf_result['observables']['total_moment_muB']))<=1e-4)
    return dict(passed=not errors,checks=checks,errors=errors,report=report,directions=alignment)

def compare_modes(vector_path,values_path):
    a=json.loads((Path(vector_path)/'remag-frozen-potential.json').read_text());b=json.loads((Path(values_path)/'remag-frozen-potential.json').read_text())
    checks=[]
    for pa,pb in zip(a['points'],b['points'],strict=True):
        ea=np.genfromtxt(Path(vector_path)/pa['raw_spectrum'],delimiter=',',names=True)['energy_ry']
        eb=np.genfromtxt(Path(values_path)/pb['raw_spectrum'],delimiter=',',names=True)['energy_ry']
        errors={key:abs(pa[key]-pb[key])*RY_EV for key in ['band_difference_ry','band_free_difference_ry','canonical_free_difference_ry']}
        errors['spectrum_ev']=float(np.max(abs(ea-eb))*RY_EV)
        checks.append(dict(index=pa['index'],errors=errors,passed=all(v<=(1e-4 if k=='spectrum_ev' else 5e-6) for k,v in errors.items())))
    return dict(passed=all(x['passed'] for x in checks),checks=checks)

def bias(report,anchors):
    reference=anchors[0]['observables'];rows=[]
    for p,anchor in zip(report['points'][1:],anchors[1:],strict=True):
        obs=anchor['observables']
        total=obs['total_energy_ev']-reference['total_energy_ev'];free=obs['free_energy_ev']-reference['free_energy_ev']
        band=p['band_difference_ry']*RY_EV;band_free=p['band_free_difference_ry']*RY_EV
        rows.append(dict(index=p['index'],full_scf_total_difference_ev=total,full_scf_free_difference_ev=free,
            frozen_band_difference_ev=band,frozen_band_free_difference_ev=band_free,
            total_bias_ev=band-total,free_bias_ev=band_free-free,
            within_5_microev_diagnostic_target=abs(band-total)<=5e-6 and abs(band_free-free)<=5e-6))
    return dict(scope='Method bias at this basis/grid/k/smearing only; no physical convergence',rows=rows)


def reference_sensitivity(specs,probes,material,axes):
    rows=[]
    for target in axes[1:]:
        values=[]
        for spec in specs:
            if spec['material']!=material or spec['role']!='soc_free_spin':continue
            points=probes[spec['case'],spec['role']]['points']
            by_axis=dict(zip([spec['case']]+spec['targets'],points,strict=True))
            values.append(dict(reference=spec['case'],
                total_difference_ev=(by_axis[target]['band_difference_ry']-by_axis[axes[0]]['band_difference_ry'])*RY_EV,
                free_difference_ev=(by_axis[target]['band_free_difference_ry']-by_axis[axes[0]]['band_free_difference_ry'])*RY_EV))
        rows.append(dict(target=target,relative_to=axes[0],references=values,
            total_spread_ev=max(v['total_difference_ev'] for v in values)-min(v['total_difference_ev'] for v in values),
            free_spread_ev=max(v['free_difference_ev'] for v in values)-min(v['free_difference_ev'] for v in values)))
    return rows


def input_directions(spec,inputs):
    """Get requested axes from actual target manifests, never probe quaternions."""
    directions=[]
    for name in [spec['case']]+spec['targets']:
        manifest=json.loads((Path(inputs)/name/'manifest.json').read_text())
        vector=manifest.get('direction_vector')
        if vector is None:
            theta=math.radians(manifest['theta_deg']);phi=math.radians(manifest.get('phi_deg',0.))
            vector=[math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi),math.cos(theta)]
        directions.append(vector)
    return directions
