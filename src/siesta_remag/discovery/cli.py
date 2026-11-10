import argparse
import json
from pathlib import Path

from .common import load
from .config import Settings


def main(argv=None):
    parser = argparse.ArgumentParser(prog='remag-discover', description='Bounded Orb → REMAG relaxation → xyz SOC screening')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('defaults', help='Print the complete screening policy as JSON')
    for name in ('prepare', 'run'):
        p = commands.add_parser(name, help='Prepare an Orb-relaxed bundle' if name == 'prepare' else 'Run all stages on this machine')
        p.add_argument('structure', type=Path)
        p.add_argument('--pseudos', type=Path, required=True, help='Directory of fully relativistic PBE Element.psml files')
        p.add_argument('--orb-weights', type=Path, required=True, help='Cached orb-v3-conservative-inf-mpa weights; no automatic download')
        p.add_argument('--bundle', type=Path, required=True)
        p.add_argument('--settings', type=Path, help='JSON overrides of documented defaults')
        p.add_argument('--orb-device', choices=('cpu', 'cuda'), default='cpu')
        p.add_argument('--orb-threads', type=int, default=4)
        if name == 'run':
            p.add_argument('--runtime', type=Path, required=True, help='JSON with executable, executable_sha256, environment, threads')
            p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('dft', help='Continue a verified Orb bundle with REMAG on this machine')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('status', help='Read an existing report; starts no work')
    p.add_argument('report', type=Path)
    p = commands.add_parser('collect', help='Recover the existing Modal return; starts no work')
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args(argv)
    if args.command == 'defaults': result = Settings().to_dict()
    elif args.command == 'status': result = load(args.report)
    elif args.command == 'collect':
        from .collection import collect
        result = collect(args.receipt, args.output, args.timeout)
    else:
        from .pipeline import prepare_bundle, run_dft
        if args.command in ('prepare', 'run'):
            settings = Settings.from_dict(load(args.settings) if args.settings else {})
            result = prepare_bundle(args.structure, args.pseudos, args.orb_weights, args.bundle, settings,
                                    orb_device=args.orb_device, orb_threads=args.orb_threads)
            if result['status'] != 'prepared':
                print(json.dumps(result, indent=2)); return 2
        if args.command in ('run', 'dft'):
            runtime = load(args.runtime)
            result = run_dft(args.bundle, args.output, runtime['executable'], executable_sha256=runtime['executable_sha256'],
                             runtime_env=runtime.get('environment', {}), threads=runtime.get('threads', 1))
    print(json.dumps(result, indent=2))
    return 2 if str(result.get('status', '')).startswith('held') else 0


if __name__ == '__main__':
    raise SystemExit(main())
