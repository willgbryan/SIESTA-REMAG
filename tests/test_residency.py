from siesta_remag.residency import summarize_residency

def test_phase_coverage_and_reconciliation_do_not_imply_residency():
    def interval(phase,iteration,h2d,d2h):
        return dict(phase=phase,iteration=iteration,boundaries={"solver_status":dict(
            host_to_device_bytes=h2d,device_to_host_bytes=d2h,copy_calls=1)})
    trace=dict(scope="explicit",phase_markers_seen=True,geometry_epochs=2,intervals=[
        interval("initialization",0,80,0),interval("scf",1,0,4),interval("scf",2,0,4),
        interval("analysis_export",0,0,100),interval("scf",1,0,4)])
    telemetry={"gpu_bridge":dict(host_to_device_bytes=80,device_to_host_bytes=112)}
    result=summarize_residency(trace,telemetry,3)
    assert result['accounting_errors']==[]
    assert result['steady_state_explicit']['device_to_host_bytes']==4
    assert result['phases']['scf']['device_to_host_bytes']==12
    assert not result['strict_residency_verified']
    assert len(result['unmeasured'])==3
    assert len(summarize_residency(trace,{},4)['accounting_errors'])==3

def test_whole_cycle_counts_regions_and_includes_first_iteration_traffic():
    traffic=dict(host_to_device_bytes=0,device_to_host_bytes=4,copy_calls=1)
    trace=dict(scope="explicit",execution_mode="device_cycle",phase_markers_seen=True,geometry_epochs=1,
               intervals=[dict(phase="scf",iteration=1,boundaries={"bad_status":traffic})])
    telemetry={"gpu_device_cycle":dict(cycles=[dict(iterations=9)],host_to_device_bytes=0,device_to_host_bytes=4)}
    result=summarize_residency(trace,telemetry,9)
    assert result['accounting_errors']==[]
    assert result['scf_intervals']==1 and result['device_iterations']==9
    assert result['steady_state_explicit']['device_to_host_bytes']==4
    assert not result['strict_residency_verified']

def test_terminal_integral_publication_reconciles_without_counting_as_scf_traffic():
    setup=dict(host_to_device_bytes=180,device_to_host_bytes=0,copy_calls=2)
    output=dict(host_to_device_bytes=0,device_to_host_bytes=180,copy_calls=2)
    trace=dict(scope="explicit",execution_mode="device_cycle",phase_markers_seen=True,geometry_epochs=1,
               intervals=[dict(phase="scf",iteration=1,boundaries={}),
                          dict(phase="post_scf",iteration=0,boundaries={"grid_flow_geometry":setup,"final_force_output":output})])
    telemetry={"gpu_device_cycle":dict(cycles=[dict(iterations=9)]),
               "gpu_integral_force":dict(host_to_device_bytes=180,device_to_host_bytes=180)}
    result=summarize_residency(trace,telemetry,9)
    assert result['accounting_errors']==[]
    assert result['steady_state_explicit']==dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0)
    assert result['total_explicit']==dict(host_to_device_bytes=180,device_to_host_bytes=180,copy_calls=4)
    telemetry['gpu_integral_force']['device_to_host_bytes']=0
    assert summarize_residency(trace,telemetry,9)['accounting_errors']
