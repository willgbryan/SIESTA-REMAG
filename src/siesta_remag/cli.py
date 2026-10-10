import argparse
import json
from pathlib import Path
from .corpus import CASES, IO_POLICIES, prepare
from .runner import run_case
from .compare import compare, compare_mae

def main():
    parser = argparse.ArgumentParser(prog="remag")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("case", choices=CASES)
    p.add_argument("--upstream", type=Path, default=Path("upstream/siesta"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--io-policy", choices=IO_POLICIES, default="end-of-cycle")
    p = commands.add_parser("run")
    p.add_argument("case", type=Path)
    p.add_argument("--executable", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--timeout", type=float, default=1800)
    p = commands.add_parser("compare")
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("mae")
    for name in ("reference_x", "reference_z", "candidate_x", "candidate_z"):
        p.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.case, args.upstream, args.output, io_policy=args.io_policy))
        return
    if args.command == "run":
        result = run_case(args.case, args.executable, args.output, threads=args.threads, timeout=args.timeout)
        print(json.dumps({k: result[k] for k in ("valid", "wall_seconds", "observables")}, indent=2))
        raise SystemExit(0 if result["valid"] else 1)
    if args.command == "compare":
        result = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()),
                         reference_dir=args.reference.parent, candidate_dir=args.candidate.parent)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    else:
        paths = [getattr(args, name) for name in ("reference_x", "reference_z", "candidate_x", "candidate_z")]
        result = compare_mae(*(json.loads(path.read_text()) for path in paths), directories=[p.parent for p in paths])
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)

if __name__ == "__main__":
    main()
