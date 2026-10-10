from siesta_remag.io_trace import summarize_io

def test_unfinished_calls_and_conservative_phase_boundaries():
    trace={"intervals":[
        dict(phase="initialization",iteration=0,unix_begin_ns=1_000_000_000,unix_end_ns=2_000_000_000),
        dict(phase="scf",iteration=1,unix_begin_ns=2_000_000_000,unix_end_ns=3_000_000_000),
        dict(phase="scf",iteration=2,unix_begin_ns=3_000_000_000,unix_end_ns=4_000_000_000),
        dict(phase="post_scf",iteration=0,unix_begin_ns=4_000_000_000,unix_end_ns=5_000_000_000)]}
    lines=['7 1.500000 read(0</input>, "fdf", 3) = 3 <0.000010>',
           '7 2.500000 write(1</stdout>, "scf", 3) = 3 <0.000010>',
           '8 2.900000 read(9</cache>,  <unfinished ...>',
           '8 3.100000 <... read resumed>"value", 5) = 5 <0.200000>',
           '7 3.300000 openat(AT_FDCWD, "absent", O_RDONLY) = -1 ENOENT (No such file or directory) <0.000010>',
           '7 4.000000 write(1</stdout>, "done", 4) = 4 <0.000001>',
           '7 5.500000 +++ exited with 0 +++']
    r=summarize_io(lines,trace,process_succeeded=True)
    assert r['coverage_complete']
    assert r['parsed_calls']==5
    assert r['total']['read_bytes']==8
    assert r['steady_state']['read_bytes']==5
    assert r['steady_state']['written_bytes']==4
    assert r['phases']['post_scf']['written_bytes']==4
    assert r['steady_state']['failed_calls']==1
    assert not r['strict_no_io_verified']
    assert not summarize_io(lines+['unparsed'],trace,process_succeeded=True)['coverage_complete']
    assert not summarize_io(lines,trace,process_succeeded=False)['coverage_complete']
    assert not summarize_io([],{},process_succeeded=True)['coverage_complete']

def test_whole_device_cycle_does_not_exempt_the_first_launch():
    trace=dict(execution_mode="device_cycle",intervals=[
        dict(phase="scf",iteration=1,unix_begin_ns=1_000_000_000,unix_end_ns=2_000_000_000)])
    r=summarize_io(['7 1.500000 read(3</unexpected>, "x", 1) = 1 <0.000010>', '7 2.500000 +++ exited with 0 +++'],trace,process_succeeded=True)
    assert r['coverage_complete']
    assert r['steady_state']['read_bytes']==1
