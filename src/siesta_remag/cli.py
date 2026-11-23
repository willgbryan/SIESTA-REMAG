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
    p.add_argument("--electronic-only", action="store_true", help="Compare declared energy-only candidates, including free energy; omit forces/stress")
    p = commands.add_parser("mae")
    for name in ("reference_x", "reference_z", "candidate_x", "candidate_z"):
        p.add_argument(name, type=Path)
    p.add_argument("--electronic-only", action="store_true", help="Check total and free directional energies for declared energy-only candidates")
    p = commands.add_parser('magnetic-import', help='Import a complete native generator export')
    p.add_argument('calculation', type=Path)
    p.add_argument('--metadata', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p = commands.add_parser('mft', help='GPU frozen-potential magnetic screening')
    p.add_argument('state', type=Path)
    p.add_argument('--directions', required=True, type=Path, help='JSON array of Cartesian directions')
    p.add_argument('--angular-model', choices=('axial4', 'cubic6'))
    p.add_argument('--mesh', type=int, nargs=3, help='Resample the native finite-range generator')
    p.add_argument('--generator', type=Path, help='Directory with remag-magnetic-realspace export')
    p.add_argument('--memory-gib', type=float, default=8., help='Native device workspace limit')
    p.add_argument('--output', required=True, type=Path)
    p = commands.add_parser('exchange', help='GPU real-space Green response from a saved state')
    p.add_argument('state', type=Path)
    p.add_argument('--radius', type=float, default=3.)
    p.add_argument('--mesh', type=int, nargs=3)
    p.add_argument('--generator', type=Path)
    p.add_argument('--relativistic', action='store_true')
    p.add_argument('--output', required=True, type=Path)
    p = commands.add_parser('mft-evidence', help='Read an existing complete CUDA frozen sweep')
    p.add_argument('report', type=Path)
    p.add_argument('--reference-direction', type=float, nargs=3, required=True)
    p.add_argument('--volume-angstrom3', type=float)
    p.add_argument('--output', required=True, type=Path)
    p = commands.add_parser('magnetic-compare', help='Compare two magnetic screening packages')
    p.add_argument('parent', type=Path)
    p.add_argument('child', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command in ('magnetic-import', 'mft', 'exchange', 'mft-evidence', 'magnetic-compare'):
        from .magnetic import load_magnetic_state, calculate_mft, read_mft_evidence, compare_magnetic_responses, summarize_magnetic_screen
        if args.command == 'magnetic-import':
            from .magnetic.state import import_native_state
            state = import_native_state(args.calculation, json.loads(args.metadata.read_text()))
            print(state.save(args.output)); return
        if args.command in ('mft','exchange'):
            state = load_magnetic_state(args.state)
            if args.mesh:
                if not args.generator:parser.error('--mesh requires --generator with native finite-range data')
                from .magnetic import load_realspace
                state=load_realspace(args.generator,state.metadata).sample(args.mesh)
            if args.command=='exchange':
                from .magnetic import calculate_exchange
                result=calculate_exchange(state,radius_angstrom=args.radius,relativistic=args.relativistic)
                args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
                print(args.output);return
            result = calculate_mft(state, json.loads(args.directions.read_text()),
                                   angular_model=args.angular_model, output=args.output,
                                   memory_limit_bytes=int(args.memory_gib*1024**3))
            screen = summarize_magnetic_screen(state, mae=result)
            (args.output/'magnetic-screen.json').write_text(json.dumps(screen, indent=2, allow_nan=False)+'\n')
            print(json.dumps(screen['summary'], indent=2, allow_nan=False)); return
        if args.command == 'mft-evidence':
            result = read_mft_evidence(args.report, reference_direction=args.reference_direction,
                                       volume_angstrom3=args.volume_angstrom3)
        else:
            result = compare_magnetic_responses(json.loads(args.parent.read_text()), json.loads(args.child.read_text()))
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(args.output); return
    if args.command == "prepare":
        print(prepare(args.case, args.upstream, args.output, io_policy=args.io_policy))
        return
    if args.command == "run":
        result = run_case(args.case, args.executable, args.output, threads=args.threads, timeout=args.timeout)
        print(json.dumps({k: result[k] for k in ("valid", "wall_seconds", "observables")}, indent=2))
        raise SystemExit(0 if result["valid"] else 1)
    if args.command == "compare":
        result = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()),
                         reference_dir=args.reference.parent, candidate_dir=args.candidate.parent,
                         electronic_only=args.electronic_only)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    else:
        paths = [getattr(args, name) for name in ("reference_x", "reference_z", "candidate_x", "candidate_z")]
        result = compare_mae(*(json.loads(path.read_text()) for path in paths), directories=[p.parent for p in paths],
                             electronic_only=args.electronic_only)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)

if __name__ == "__main__":
    main()
