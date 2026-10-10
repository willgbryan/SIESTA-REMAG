#!/usr/bin/env python3
"""Recover compact evidence from a completed/interrupted run; allocates no compute."""
import argparse
import asyncio
import fnmatch
import json
from pathlib import Path
from modal.volume import (_Volume, _expected_block_lengths, _read_block_body,
                          _raise_on_block_response_error, _block_download_timeout, retry)
from modal._utils.http_utils import ClientSessionRegistry
from modal.exception import ExecutionError
from modal_proto import api_pb2

parser=argparse.ArgumentParser()
parser.add_argument("run_id")
parser.add_argument("--output",type=Path)
parser.add_argument("--density", action="store_true", help="Also download raw RHO/DRHO files for local scientific comparisons")
parser.add_argument("--matrices", action="store_true", help="Also download requested HSX/DM/TSDE/NetCDF matrix outputs and final XV geometry")
mode=parser.add_mutually_exclusive_group()
mode.add_argument("--sources-only",action="store_true",help="Fetch only source/provenance checkpoints while a run is active")
mode.add_argument("--outputs-only",action="store_true",help="Fetch requested density/matrix binaries omitted from the returned archive")
mode.add_argument("--materials-only",action="store_true",help="Fetch material records and scientific comparisons while the complete component archive is packaging")
parser.add_argument("--workers",type=int,default=4,help="Concurrent artifact reads only; allocates no compute")
parser.add_argument("--attempt-id",help="Require this attempt before and after a live checkpoint fetch; prevents cross-retry evidence reuse")
parser.add_argument("--include",action="append",default=[],help="Read only matching paths relative to the run (shell-style pattern); repeat for multiple paths")
args=parser.parse_args()
if not args.run_id.replace("-", "").isalnum():
    raise SystemExit("Invalid run ID")
if not 1<=args.workers<=8:raise SystemExit("Choose one to eight artifact readers")
if args.attempt_id and (len(args.attempt_id)!=32 or any(c not in '0123456789abcdef' for c in args.attempt_id)):
    parser.error("Attempt ID must be 32 lowercase hexadecimal characters")
if args.outputs_only and not (args.density or args.matrices):parser.error("--outputs-only requires --density or --matrices")
if args.sources_only and (args.density or args.matrices):parser.error("--sources-only does not include requested output binaries")
target=args.output or Path(__file__).resolve().parents[1]/"results"/args.run_id
volume=_Volume.from_name("siesta-remag-results")
def selected(entry):
    path=Path(entry.path)
    relative=path.relative_to(args.run_id)
    if args.include and not any(fnmatch.fnmatchcase(relative.as_posix(),pattern) for pattern in args.include):
        return None
    if args.materials_only and not ((len(relative.parts)>=3 and (relative.parts[1]=='reference' or relative.parts[1].startswith('candidate'))) or
            (len(relative.parts)==1 and (path.name in ('metadata.json','ledger.json','run-attempt.json') or path.name.startswith(('mae-','ablation-'))))):
        return None
    if args.sources_only and (len(relative.parts)!=1 or path.name not in ("metadata.json","implementation.tar.gz","run-attempt.json")):
        return None
    if args.outputs_only and not ((args.density and path.suffix in (".RHO",".DRHO")) or (args.matrices and path.suffix in (".HSX",".DM",".TSDE",".nc",".XV"))):
        return None
    if path.suffix not in (".json", ".log", ".csv", ".fdf", ".EIG", ".npz", ".ptx", ".F", ".f90") and path.name != "implementation.tar.gz" and not (args.density and path.suffix in (".RHO", ".DRHO")) and not (args.matrices and path.suffix in (".HSX", ".DM", ".TSDE", ".nc", ".XV")) and not (path.name.startswith(("remag-failed-","remag-final-grid-shadow-","remag-final-density-shadow-","remag-nonlocal-h0-")) and path.suffix == ".bin"):
        return None
    if ".." in relative.parts:
        raise ValueError("Unsafe artifact path")
    output=target/relative
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        return None
    return entry.path,output

async def fetch():
    await volume.hydrate()
    async def attempt():
        data=b''.join([b async for b in volume.read_file(args.run_id+'/run-attempt.json')])
        marker=json.loads(data)
        if marker.get('attempt_id')!=args.attempt_id:
            raise ValueError('Remote attempt changed; fetch into a new attempt-specific directory')
        return marker
    checkpoint=None
    if args.attempt_id:
        before=await attempt()
        target.mkdir(parents=True,exist_ok=True)
        checkpoint=target/'fetch-attempt.json'
        if checkpoint.exists():
            previous=json.loads(checkpoint.read_text())
            if (previous['before']['attempt_id']!=before['attempt_id'] or
                    previous['before']['invocation_sha256']!=before['invocation_sha256']):
                raise ValueError('Local checkpoint belongs to a different attempt')
            if previous.get('consistent') is False:
                raise ValueError('Local checkpoint crossed attempts; preserve it and use a fresh directory')
        checkpoint.write_text(json.dumps(dict(before=before,consistent=None),indent=2)+'\n')
    entries=await volume.listdir(args.run_id,recursive=True)
    # A live batch may be writing the next material's binary files. Only
    # recover outputs whose process has already published its result record.
    completed={Path(entry.path).parent for entry in entries if Path(entry.path).name=='result.json'}
    pending=[item for entry in entries
             if (not args.outputs_only or Path(entry.path).parent in completed)
             and (item:=selected(entry)) is not None]
    pending.sort(key=lambda item:(len(Path(item[0]).parts),item[0]))
    limit=asyncio.Semaphore(args.workers)
    @retry(n_attempts=5,base_delay=0.1,attempt_timeout=None)
    async def read_block(url,expected):
        async with ClientSessionRegistry.get_session().get(url,timeout=_block_download_timeout()) as response:
            await _raise_on_block_response_error(response)
            body=await _read_block_body(response,expected)
        if len(body)!=expected:
            raise ExecutionError(f'Incomplete server-padded block: {len(body)} != {expected}')
        return body
    async def download(path,output):
        async with limit:
            partial=output.with_name(output.name+".partial")
            # v176 had malformed HSX/DM copies after two client-padded reads.
            # Server-padded reads recovered them. Require complete responses
            # instead of synthesizing zero tails; retain SDK digest verification.
            response=await volume.client._stub.VolumeGetFile2(api_pb2.VolumeGetFile2Request(
                volume_id=volume.object_id,path=path,client_pads_blocks=False))
            if response.start!=0:raise ValueError('Unexpected partial-file response')
            lengths=_expected_block_lengths(response.start,response.len,len(response.get_urls))
            with partial.open("wb") as f:
                for url,expected in zip(response.get_urls,lengths):
                    f.write(await read_block(url,expected))
            if partial.stat().st_size!=response.len:raise ValueError('Unexpected downloaded file length')
            partial.replace(output)
    await asyncio.gather(*(download(path,output) for path,output in pending))
    if checkpoint:
        try:
            after=await attempt()
            if after['invocation_sha256']!=before['invocation_sha256']:
                raise ValueError('Remote invocation changed during evidence fetch')
        except ValueError as error:
            checkpoint.write_text(json.dumps(dict(before=before,consistent=False,error=str(error)),indent=2)+'\n')
            raise
        checkpoint.write_text(json.dumps(dict(before=before,after=after,consistent=True,
            downloaded_files=len(pending)),indent=2)+'\n')
asyncio.run(fetch())
print(target)
