"""Compact complete reports, recoverable from a function call without new GPU work."""
import hashlib
import io
from pathlib import Path
import tarfile
import time

from ..corpus import sha256
from .common import load, save


def pack(root):
    root = Path(root)
    report = load(root/'report.json')
    if report['status'] == 'running': raise ValueError('Cannot package an unfinished pipeline')
    files = [p for p in sorted(root.rglob('*')) if p.is_file() and
             p.name != 'archive-manifest.json' and p.suffix in ('.json', '.log', '.csv', '.fdf', '.cif', '.extxyz', '.EIG')]
    manifest = dict(schema=1, invocation_sha256=report['invocation_sha256'], status=report['status'],
                    files={str(p.relative_to(root)): sha256(p) for p in files},
                    omitted_binary_artifacts='DM/HSX and pseudopotential bytes remain on the remote volume; hashes are in reviews.')
    save(root/'archive-manifest.json', manifest)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for path in files + [root/'archive-manifest.json']:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    return stream.getvalue()


def unpack(payload, output):
    output = Path(output).resolve()
    payload_hash = hashlib.sha256(payload).hexdigest()
    if output.exists():
        receipt = load(output/'collection.json')
        if receipt['archive_sha256'] != payload_hash: raise ValueError('Collection output already contains different evidence')
        verify_archive(output)
        return load(output/'report.json')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() or not (output/m.name).resolve().is_relative_to(output) for m in members):
            raise ValueError('Archive contains unsafe or duplicate paths')
        archive.extractall(output, filter='data')
    verify_archive(output)
    save(output/'collection.json', dict(archive_sha256=payload_hash, archive_bytes=len(payload),
                                      local_unpack_seconds=time.monotonic()-started))
    return load(output/'report.json')


def verify_archive(output):
    output = Path(output)
    manifest = load(output/'archive-manifest.json')
    report = load(output/'report.json')
    actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} - {'archive-manifest.json', 'collection.json'}
    if 'report.json' not in manifest['files'] or actual != set(manifest['files']):
        raise ValueError('Incomplete archive file manifest')
    if report['status'] == 'running' or report['status'] != manifest['status'] or report['invocation_sha256'] != manifest['invocation_sha256']:
        raise ValueError('Incomplete or mismatched collection')
    for name, h in manifest['files'].items():
        if not (output/name).resolve().is_relative_to(output.resolve()) or sha256(output/name) != h:
            raise ValueError('Archive checksum mismatch: '+name)


def collect(receipt_path, output, timeout=30):
    import modal
    receipt = load(receipt_path)
    if not receipt.get('function_call_id'): raise ValueError('No submitted function call in receipt')
    started = time.monotonic()
    # This retrieves an existing call only. It cannot launch or restart work.
    payload = modal.FunctionCall.from_id(receipt['function_call_id']).get(timeout=timeout)
    wait_and_download = time.monotonic()-started
    report = unpack(payload, output)
    receipt.update(status='collected', output=str(Path(output).resolve()), collection_call_seconds=wait_and_download,
                   collection_call_includes_wait_for_function=True, download_only_seconds=None,
                   archive_sha256=hashlib.sha256(payload).hexdigest())
    save(receipt_path, receipt)
    return report
