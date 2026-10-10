#!/usr/bin/env python3
"""Serialize launches and refuse a second active REMAG Modal app."""
import fcntl
import asyncio
import json
import subprocess
import sys
from pathlib import Path


async def active_gpus(containers):
    # The CLI list omits resource types. Query allocation metadata only; never
    # execute commands in, or stop, an unrelated user's container.
    from modal.client import _Client
    from modal_proto import api_pb2
    client = await _Client.from_env()
    active = []
    for container in containers:
        response = await client._stub.TaskGetInfo(api_pb2.TaskGetInfoRequest(task_id=container["container_id"]))
        info = response.info
        if info.finished_at:
            continue
        gpu_type = info.gpu_type or info.gpu_config.gpu_type
        if gpu_type.upper() != "CPU" or info.gpu_config.count:
            active.append(dict(container=container["container_id"], gpu_type=gpu_type or "unknown",
                               gpu_count=info.gpu_config.count))
    return active

ROOT = Path(__file__).resolve().parents[1]
arguments = sys.argv[1:]
module = ROOT / "modal_app.py"
# Isolated component probes use the same lock and account-wide allocation
# checks as the material harness. This changes only the selected entry file.
if arguments[:1] == ["--module-file"]:
    if len(arguments) < 2:
        raise SystemExit("--module-file requires a workspace Python file")
    module = (ROOT / arguments[1]).resolve()
    if not module.is_relative_to(ROOT) or not module.is_file() or module.suffix != ".py":
        raise SystemExit("The Modal entry file must be an existing workspace Python file")
    arguments = arguments[2:]
pin = json.loads((ROOT / "upstream.lock.json").read_text())
source = ROOT / "upstream/siesta"
commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
if commit != pin["commit"] or dirty.strip():
    raise SystemExit("The reference must be a clean checkout of upstream.lock.json's commit")
with (ROOT / ".modal-run.lock").open("w") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A REMAG Modal launch is already active; refusing another GPU instance")
    command = [sys.executable, "-m", "modal"]
    apps = json.loads(subprocess.check_output(command + ["app", "list", "--json"], text=True))
    active = [a for a in apps if a.get("description") == "siesta-remag-single-gpu"
              and a.get("state", "").lower() != "stopped"]
    if active:
        raise SystemExit("An active REMAG app already exists; refusing another GPU instance")
    containers = json.loads(subprocess.check_output(command + ["container", "list", "--json"], text=True))
    allocated = asyncio.run(active_gpus(containers)) if containers else []
    if allocated:
        raise SystemExit("An active or unknown GPU allocation already exists; refusing another GPU instance: "
                         + json.dumps(allocated))
    # Keep bounded remote work alive if the local client loses connectivity.
    # A subsequent launch still refuses an active app/allocation above; detach
    # must never be used to overlap GPU jobs after a local process exits.
    result = subprocess.run(command + ["run", "--detach", str(module.relative_to(ROOT)), *arguments], cwd=ROOT)
    raise SystemExit(result.returncode)
