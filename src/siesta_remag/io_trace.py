"""Conservative phase attribution of a strace log; unknown activity never implies zero I/O."""
from bisect import bisect_left
from collections import Counter, defaultdict
from decimal import Decimal
import re

# Deliberately exclude CUDA driver ioctls: they are device command submission,
# not filesystem reads/writes. Mappings are retained as a separate category.
SYSCALLS = ("%file,read,write,readv,writev,pread64,pwrite64,preadv,pwritev,"
            "close,lseek,fsync,fdatasync,ftruncate,mmap,munmap,msync")
READS = {"read", "readv", "pread64", "preadv"}
WRITES = {"write", "writev", "pwrite64", "pwritev"}
MAPS = {"mmap", "munmap", "msync"}
STAMP = re.compile(r"^(?:(\d+)\s+)?(\d+\.\d+)\s+(.*)$")
CALL = re.compile(r"^(\w+)\((.*)\)\s+=\s+(.+?)(?:\s+<(\d+\.\d+)>)?$")
RESUMED = re.compile(r"^<\.\.\. (\w+) resumed>(.*)$")

def nanoseconds(value):
    return int(Decimal(value) * 1_000_000_000)

def summarize_io(lines, trace, *, process_succeeded):
    intervals = trace.get("intervals", [])
    errors = []
    if not intervals or any("unix_begin_ns" not in x or "unix_end_ns" not in x for x in intervals):
        return dict(coverage_complete=False, errors=["Phase timestamps unavailable"], strict_no_io_verified=False)
    starts = [v["unix_begin_ns"] for v in intervals]
    ends = [v["unix_end_ns"] for v in intervals]
    if any(a > b for a, b in zip(starts, ends)) or any(a != b for a, b in zip(ends, starts[1:])):
        return dict(coverage_complete=False, errors=["Phase wall-clock discontinuity"], strict_no_io_verified=False)
    pending = {}
    phases = defaultdict(Counter)
    operations = defaultdict(Counter)
    total, steady = Counter(), Counter()
    unmatched, parsed, exited, ambiguous = 0, 0, False, 0
    paths = Counter()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        stamp = STAMP.match(line)
        if not stamp:
            unmatched += 1
            continue
        pid, timestamp, call = stamp.groups()
        pid = pid or "main"
        begin = nanoseconds(timestamp)
        if call.startswith("+++ exited with ") or call.startswith("+++ killed by "):
            exited = True
            continue
        if call.startswith("--- "):
            continue  # signal delivery, not a filesystem syscall
        if call.endswith("<unfinished ...>"):
            if pid in pending:
                unmatched += 1
            pending[pid] = (begin, call.removesuffix("<unfinished ...>"))
            continue
        resumed = RESUMED.match(call)
        if resumed:
            previous = pending.pop(pid, None)
            if previous is None or not previous[1].startswith(resumed[1]+"("):
                unmatched += 1
                continue
            begin, prefix = previous
            call = prefix + resumed[2]
        match = CALL.match(call)
        if not match:
            unmatched += 1
            continue
        name, arguments, returned, duration = match.groups()
        end = begin + nanoseconds(duration or "0")
        # -ttt timestamps have microsecond resolution. Attribute any operation
        # touching a boundary to both neighboring phases rather than hide it.
        first = bisect_left(ends, begin-1000)
        overlapping = []
        for index in range(first, len(intervals)):
            if starts[index] > end+1000:
                break
            overlapping.append(intervals[index])
        if len(overlapping) > 1:
            ambiguous += 1
        labels = {v["phase"] for v in overlapping} or {"outside_markers"}
        count = Counter(calls=1)
        result = re.match(r"(-?\d+)", returned)
        if returned.startswith("-1 "):
            count["failed_calls"] = 1
        if name in READS | WRITES:
            if result is None:
                unmatched += 1
            else:
                count["read_bytes" if name in READS else "written_bytes"] = max(0, int(result[1]))
        count["mapping_calls" if name in MAPS else "filesystem_calls"] = 1
        for phase in labels:
            phases[phase].update(count)
            operations[phase][name] += 1
        total.update(count)
        if any(v["phase"] == "scf" and (trace.get("execution_mode")=="device_cycle" or v["iteration"] > 1) for v in overlapping):
            steady.update(count)
        # Best-effort labels only; coverage and call/byte counts do not depend
        # on recovering a path from a descriptor annotation.
        if "scf" in labels:
            descriptor = re.search(r"\d+<([^>]+)>", arguments)
            if descriptor:
                paths[descriptor[1]] += 1
        parsed += 1
    if unmatched or pending:
        errors.append(f"Unparsed trace records: {unmatched}; unfinished calls: {len(pending)}")
    if not exited or not process_succeeded:
        errors.append("Successful traced-process completion was not established")
    if parsed == 0:
        errors.append("No syscall records captured")
    return dict(scope="selected filesystem syscalls and mappings from SIESTA and its child threads",
                coverage_complete=not errors, errors=errors, parsed_calls=parsed,
                boundary_overlaps=ambiguous, phases=dict(phases), phase_operations=dict(operations),
                total=dict(total), steady_state=dict(steady), scf_descriptor_paths=dict(paths),
                strict_no_io_verified=False,
                unmeasured=["memory-mapped page faults", "asynchronous kernel writeback"],
                notes="Calls overlapping phase boundaries count in each phase; phase totals need not add to the process total. "
                      "Mapping calls include anonymous and device mappings; descriptor labels are best effort. "
                      "Tracing perturbs timing; no speedup is qualified for traced runs.")
