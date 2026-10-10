"""Preserve interrupted Modal attempts without merging scientific evidence."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import uuid

MARKER='run-attempt.json'

def _write(path,value):
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)

def invocation_digest(parameters,sources,binaries):
    value=dict(parameters=parameters,sources=sources,binaries=binaries)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def prepare_attempt(root,fingerprint):
    """Return (marker, completed). Existing evidence is never overwritten.

    Called only inside the single-container, serial Modal function. A platform
    reschedule may repeat an invocation despite application retries=0. Preserve
    the prior attempt in a sibling directory and start a fresh scientific batch.
    Completed attempts can be repackaged after a lost response without rerunning.
    """
    root=Path(root)
    if root.exists():
        marker=root/MARKER
        if not marker.is_file():raise ValueError('Existing run lacks an attempt identity; recover it explicitly')
        prior=json.loads(marker.read_text())
        if prior.get('invocation_sha256')!=fingerprint:raise ValueError('Existing run belongs to a different invocation')
        if prior.get('state')=='complete':return prior,True
        if prior.get('state')!='running':raise ValueError('Unknown run-attempt state')
        identifier=prior.get('attempt_id','')
        if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise ValueError('Invalid run-attempt identity')
        saved=root.with_name(root.name+'-interrupted-'+identifier)
        if saved.exists():raise ValueError('Interrupted-attempt destination already exists')
        root.rename(saved)
    # Reconstruct this list even if a previous worker disappeared between the
    # directory rename and creation of the next attempt. Only matching inputs
    # are associated with this invocation; their contents remain separate.
    previous=[]
    for path in sorted(root.parent.glob(root.name+'-interrupted-*')):
        marker=path/MARKER
        if marker.is_file() and json.loads(marker.read_text()).get('invocation_sha256')==fingerprint:
            previous.append(path.name)
    root.mkdir(exist_ok=False)
    record=dict(invocation_sha256=fingerprint,attempt_id=uuid.uuid4().hex,state='running',interrupted_attempts=previous,
                policy='Fresh batch per interrupted attempt; no cross-attempt material or timing merge')
    _write(root/MARKER,record)
    return record,False

def complete_attempt(root):
    root=Path(root);record=json.loads((root/MARKER).read_text())
    if record.get('state')!='running':raise ValueError('Cannot complete an inactive attempt')
    record['state']='complete';_write(root/MARKER,record)

def package_evidence(root):
    root=Path(root);buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and (path.suffix in ('.json','.log','.csv','.fdf','.EIG','.npz','.ptx','.F','.f90') or path.name=='implementation.tar.gz' or
                (path.name.startswith(('remag-failed-','remag-final-grid-shadow-','remag-final-density-shadow-','remag-nonlocal-h0-')) and path.suffix=='.bin')):
                archive.add(path,arcname=str(path.relative_to(root)))
    return buffer.getvalue()
