import argparse
import json
from pathlib import Path

from ..discovery.common import load, save
from ..discovery.config import Settings
from .experiments import prepare_experiment, run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m siesta_remag.research')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fept', help='Author exact FePt strain family and prepare fixed-geometry bundles')
    p.add_argument('--pseudos', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--settings', type=Path)
    p.add_argument('--mesh', type=int, nargs=3, default=[6, 6, 6])
    p.add_argument('--basis', default='DZP')
    p = sub.add_parser('run', help='Run one fixed-geometry experiment with an explicit native runtime')
    p.add_argument('bundle', type=Path)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('serve', help='Open local scientific workspace')
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--port', type=int, default=8765)
    p = sub.add_parser('electronic', help='Export orbital-resolved spectra from HSX')
    p.add_argument('hsx', type=Path)
    p.add_argument('--sampling', type=Path, required=True, help='JSON kpoints (reciprocal fractional) and normalized weights')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('electronic-run', help='Reconstruct and check spectra against native KP/EIG outputs')
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--spin', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'serve':
        from .workspace import serve
        serve(args.workspace, args.port)
        return 0
    if args.command == 'fept':
        from gpac.physical import fept_reference, strain, export_geometry
        p = fept_reference()
        settings = Settings.from_dict(load(args.settings) if args.settings else {})
        args.output.mkdir(parents=True, exist_ok=False)
        rows = []
        for label, q in [('m200', '-1/50'), ('m100', '-1/100'), ('m050', '-1/200'), ('zero', '0'),
                         ('p050', '1/200'), ('p100', '1/100'), ('p200', '1/50')]:
            intervention = strain(p, 2, q, expected_sha256=p.sha256)
            export = export_geometry(intervention['child']['program'], length_unit='angstrom')
            prepare_experiment(export, args.pseudos, args.output/label, settings=settings, mesh=args.mesh,
                               basis_size=args.basis, parameter=dict(name='epsilon_c', exact=q, unit='dimensionless'))
            rows.append(dict(id=label, strain=q, parent_sha256=p.sha256, source_sha256=export['source']['sha256'],
                             heldout=label in ('m050', 'p050'), status='prepared'))
        result = dict(format='REMAG-CAMPAIGN/1', material='L10 FePt',
            strain_path='fixed a=b=2.73 angstrom; c=3.79*(1+epsilon_c); fixed fractional sites',
            reference_source_sha256=p.sha256, experiments=rows, physical_accuracy_qualified=False)
        save(args.output/'campaign.json', result)
    elif args.command == 'run':
        runtime = load(args.runtime)
        result = run_experiment(args.bundle, args.output, runtime['executable'], executable_sha256=runtime['executable_sha256'],
                                runtime_env=runtime.get('environment', {}), threads=runtime.get('threads', 1))
    elif args.command == 'electronic-run':
        from .electronic import spectrum_from_run
        result = spectrum_from_run(args.directory, args.output, collinear_spin=args.spin)
    else:
        from .electronic import spectrum
        sampling = load(args.sampling)
        result = spectrum(args.hsx, sampling['kpoints'], sampling['weights'], args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if result.get('status', '').startswith('held') else 0


if __name__ == '__main__':
    raise SystemExit(main())
