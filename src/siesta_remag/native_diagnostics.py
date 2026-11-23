"""Bounded, best-effort crash context for the owned native child, never a retry.

Read mapping metadata, not process memory or environment. A sampled mapping is
not a core dump, an ELF image hash, or proof of which instruction caused a fault.
"""
from pathlib import Path
import re
import sys


_MAP = re.compile(r'^([0-9a-f]+)-([0-9a-f]+) ([r-][w-][x-][ps]) '
                  r'([0-9a-f]+) ([0-9a-f]+:[0-9a-f]+) ([0-9]+)(?:\s+(.*))?$')


def parse_maps(text):
    regions = []
    for line in text.splitlines():
        match = _MAP.fullmatch(line)
        if not match:
            raise ValueError('Unrecognized process mapping')
        start, end, permissions, offset, device, inode, path = match.groups()
        if int(end, 16) <= int(start, 16):
            raise ValueError('Invalid process mapping range')
        regions.append(dict(start='0x'+start, end='0x'+end, permissions=permissions,
                            file_offset='0x'+offset, device=device, inode=int(inode),
                            path=path or None))
    if not regions:
        raise ValueError('Empty process mapping')
    return regions


def _read_bounded(path, limit):
    with path.open('rb') as stream:
        data = stream.read(limit+1)
    if len(data) > limit:
        raise ValueError('Diagnostic file exceeds byte limit')
    return data.decode('utf-8', errors='replace')


class NativeProcessDiagnostics:
    def __init__(self, pid, *, traced=False, proc_root=Path('/proc'), platform=None):
        self.pid, self.root, self.next_sample = pid, Path(proc_root), 0.
        platform = sys.platform if platform is None else platform
        self.enabled = platform.startswith('linux') and not traced
        reason = 'tracer parent is not the native process' if traced else 'procfs requires Linux'
        self.record = dict(schema=1, pid=pid, status='pending' if self.enabled else 'unavailable',
                           reason=None if self.enabled else reason, sample_interval_seconds=.5,
                           snapshot=None, last_error=None, driver_version_text=None,
                           driver_version_error=None,
                           scope='Last successful live sample; not a core dump or byte identity of mapped images.')
        if self.enabled:
            try:
                self.record['driver_version_text'] = _read_bounded(self.root/'driver/nvidia/version', 8192)
            except (OSError, ValueError) as error:
                self.record['driver_version_error'] = type(error).__name__+': '+str(error)

    def sample(self, seconds):
        if not self.enabled or seconds < self.next_sample:
            return
        self.next_sample = seconds + .5
        try:
            maps = parse_maps(_read_bounded(self.root/str(self.pid)/'maps', 1024*1024))
            try:
                limits = _read_bounded(self.root/str(self.pid)/'limits', 16384)
                limits_error = None
            except (OSError, ValueError) as error:
                limits, limits_error = None, type(error).__name__+': '+str(error)
            self.record.update(status='sampled', last_error=None,
                               snapshot=dict(seconds=seconds, maps=maps, limits=limits, limits_error=limits_error))
        except (OSError, ValueError) as error:
            # Keep the last usable sample when a fast exit races the observer.
            self.record['last_error'] = dict(seconds=seconds, error=type(error).__name__+': '+str(error))
            if self.record['snapshot'] is None:
                self.record['status'] = 'unavailable'

    def finish(self, returncode, timed_out):
        self.record.update(returncode=returncode, timed_out=timed_out)
        if self.record['status'] == 'pending':
            self.record.update(status='unavailable', reason='native process exited before a sample')
        return self.record
