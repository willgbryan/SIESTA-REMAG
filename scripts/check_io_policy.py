#!/usr/bin/env python3
"""Validate the sole Write.DM policy change against unmodified upstream.

This is an output-policy experiment, not a qualified performance comparison.
Original result files and their distinct input hashes are never rewritten.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from siesta_remag.compare import compare, difference
from siesta_remag.corpus import prepare
from siesta_remag.runner import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("upstream/siesta"))
    parser.add_argument("--executable", type=Path, default=Path("build/upstream/Src/siesta"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records, runs = [], {}
    for case in ("fe2_forces", "fe2_relax", "fept2_soc_z", "fept2_soc_x"):
        pair, paths, contents = [], [], []
        for policy in ("checkpoint", "end-of-cycle"):
            prepared = prepare(case, args.upstream, args.output/"inputs"/case/policy, io_policy=policy)
            contents.append((prepared/"input.fdf").read_text())
            path = args.output/case/policy
            print(f"Running {case}: {policy}", flush=True)
            pair.append(run_case(prepared, args.executable, path, timeout=900))
            paths.append(path)
        # The scientific comparison keeps its strict input-provenance rejection.
        # This separate experiment allows exactly one known output-only change.
        exact_policy_change = (contents[0].count("Write.DM true\n") == 1 and
                               contents[0].replace("Write.DM true\n", "Write.DM false\n") == contents[1])
        comparison = compare(*pair, reference_dir=paths[0], candidate_dir=paths[1])
        dm_present = all((p/"remag.DM").is_file() and (p/"remag.DM").stat().st_size > 0 for p in paths)
        record = dict(benchmark=case, exact_write_dm_only_change=exact_policy_change,
                      input_sha256=[r["benchmark"]["input_sha256"] for r in pair],
                      valid_runs=[r["valid"] for r in pair],
                      final_density_matrix_files_present=dm_present,
                      checks=comparison["checks"],
                      strict_comparison_reasons=comparison["reasons"],
                      qualified_speedup=None)
        record["passed"] = (exact_policy_change and all(record["valid_runs"]) and dm_present and
                            comparison["reasons"] == ["Non-equivalent benchmark: input_sha256"] and
                            all(c["passed"] for c in comparison["checks"].values()))
        records.append(record)
        runs[case] = pair
        (args.output/"io-policy-check.json").write_text(json.dumps(records, indent=2)+"\n")
        print(f"{case}: output-policy regression passed={record['passed']}", flush=True)

    x, z = runs["fept2_soc_x"], runs["fept2_soc_z"]
    mae = [a["observables"]["total_energy_ev"]-b["observables"]["total_energy_ev"] for a,b in zip(x,z)]
    mae_check = difference(*mae, 5.e-6)
    axes = True
    for pair, axis in ((x,0),(z,2)):
        for run in pair:
            m = np.asarray(run["observables"]["total_moment_muB"])
            axes &= bool(np.isfinite(m).all() and np.linalg.norm(m) > 1.e-8 and
                         abs(m[axis])/np.linalg.norm(m) >= np.cos(np.deg2rad(1.)))
    mae_check.update(checkpoint_mae_ev=mae[0], end_of_cycle_mae_ev=mae[1], axes_passed=axes)
    mae_check["passed"] &= axes and all(r["passed"] for r in records[-2:])
    (args.output/"io-policy-mae.json").write_text(json.dumps(mae_check, indent=2)+"\n")
    if not all(r["passed"] for r in records) or not mae_check["passed"]:
        raise SystemExit("Output-policy regression failed; original evidence retained")
    print("All four output-policy comparisons and the direct MAE check passed.", flush=True)


if __name__ == "__main__":
    main()
