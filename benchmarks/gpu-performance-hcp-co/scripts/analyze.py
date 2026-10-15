#!/usr/bin/env python3
"""Figures and summary numbers for the REMAG benchmark campaign.

Usage: analyze.py DATA_DIR OUT_DIR
DATA_DIR holds records.jsonl and runs/<group>/<run>/ (time.json, remag-device-scf.json).
Writes vector PDF figures, summary.json and numbers.tex (LaTeX macros used by the report).
"""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "lines.linewidth": 1.1, "lines.markersize": 4, "legend.frameon": False, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
ONE, TWO = 3.37, 7.0  # single- and double-column widths, inches

# Okabe-Ito palette; one colour and marker per implementation throughout.
STYLE = {
    "pristine-serial": dict(color="#000000", marker="o", ls="-", label="SIESTA 5.4.2, 1 core"),
    "pristine-mpi-28": dict(color="#999999", marker="s", ls="-", label="SIESTA 5.4.2, 28 MPI processes"),
    "host-28": dict(color="#E69F00", marker="^", ls="-", label="REMAG CPU, 28 threads"),
    "gpu-bj-64": dict(color="#56B4E9", marker="D", ls="-", label="REMAG GPU, block-Jacobi solver"),
    "gpu-batched-256": dict(color="#0072B2", marker="v", ls="-", label="REMAG GPU, batched solver"),
}
ORDER = list(STYLE)

recs = [json.loads(l) for l in (DATA / "records.jsonl").read_text().splitlines() if l.strip()]
bad = [r for r in recs if r["rc"] != 0 or not r["normal_exit"]]
ok = [r for r in recs if r["rc"] == 0 and r["normal_exit"]]


def rundir(r):
    tag = f"{r['deck']}-{'x'.join(map(str, r['kmesh']))}"
    return DATA / "runs" / r["group"] / f"{tag}-{r['label']}-r{r['rep']}"


def timers(r):
    """Flatten the SIESTA tree timer of one run: name -> seconds (first occurrence, depth-first)."""
    p = rundir(r) / "time.json"
    if not p.exists():
        return {}
    flat = {}

    def walk(node):
        for k, v in node.items():
            if isinstance(v, dict) and "_time" in v:
                flat.setdefault(k, v["_time"])
                walk(v)
    walk(json.loads(p.read_text()))
    return flat


def plain_log(axis):
    """Log axis labelled 1, 2, 5, 10, 20, 50, ... in plain numbers."""
    axis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    axis.set_minor_locator(matplotlib.ticker.NullLocator())


def select(**kw):
    return [r for r in ok if all(r.get(k) == v for k, v in kw.items())]


def stats(values):
    v = sorted(values)
    return dict(median=statistics.median(v), lo=v[0], hi=v[-1], n=len(v))


summary = dict(failed_runs=[dict(group=r["group"], impl=r["impl"], kmesh=r["kmesh"], rep=r["rep"], rc=r["rc"])
                            for r in bad])

# ---------------------------------------------------------------- Figure 1: time to solution vs k points
scal = defaultdict(list)
for r in select(group="scaling", deck="legacy"):
    scal[(r["impl"], r["nk"])].append(r)
nks = sorted({nk for _, nk in scal})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(TWO, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
table = {}
for name in ORDER:
    pts = [(nk, stats([r["wall_s"] for r in scal[(name, nk)]])) for nk in nks if scal.get((name, nk))]
    if not pts:
        continue
    x = np.array([p[0] for p in pts]); s = [p[1] for p in pts]
    y = np.array([q["median"] for q in s])
    err = np.array([[q["median"] - q["lo"] for q in s], [q["hi"] - q["median"] for q in s]])
    st = STYLE[name]
    ax.errorbar(x, y, yerr=err, color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"],
                capsize=1.5, elinewidth=0.6, mfc="white" if "pristine" in name else st["color"])
    table[name] = {str(nk): q for nk, q in pts}
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("k points in the Brillouin-zone mesh")
ax.set_ylabel("wall time to a converged solution (s)")
ax.set_xticks(nks); ax.set_xticklabels([f"{n:,}" for n in nks]); ax.minorticks_off()
plain_log(ax.yaxis)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.8)
ax.text(-0.22, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=9, family="STIXGeneral")
summary["scaling"] = table
# speed-up of the fastest REMAG path over the two SIESTA baselines
for base, ls in (("pristine-serial", "-"), ("pristine-mpi-28", "--")):
    for name in ("gpu-batched-256", "host-28"):
        if base not in table or name not in table:
            continue
        common = [nk for nk in nks if str(nk) in table[base] and str(nk) in table[name]]
        sp = [table[base][str(nk)]["median"] / table[name][str(nk)]["median"] for nk in common]
        st = STYLE[name]
        bx.plot(common, sp, color=st["color"], marker=st["marker"], ls=ls,
                mfc="white" if base == "pristine-mpi-28" else st["color"],
                label=f"{'REMAG GPU, batched' if 'gpu' in name else 'REMAG CPU, 28 threads'} vs SIESTA {'1 core' if base.endswith('serial') else '28 MPI processes'}")
        summary.setdefault("speedup", {})[f"{name}/{base}"] = dict(zip(map(str, common), sp))
bx.axhline(1, color="0.6", lw=0.6, ls=":")
bx.set_xscale("log"); bx.set_yscale("log")
bx.set_xticks(nks); bx.set_xticklabels([f"{n:,}" for n in nks]); bx.minorticks_off()
plain_log(bx.yaxis)
bx.set_xlabel("k points in the Brillouin-zone mesh")
bx.set_ylabel("speed-up (ratio of median wall times)")
bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, handlelength=2.2)
bx.text(-0.24, 1.02, "b", transform=bx.transAxes, fontweight="bold", fontsize=9, family="STIXGeneral")
fig.tight_layout(w_pad=2)
fig.savefig(OUT / "fig1_scaling.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Figure 2: where the time goes
nkmax = max(nks)
phases = [("setup", "#CC79A7"), ("SCF", "#009E73"), ("final forces and outputs", "#F0E442"), ("other", "#BBBBBB")]
rows = []
for name in ORDER:
    rr = scal.get((name, nkmax), [])
    if not rr:
        continue
    r = sorted(rr, key=lambda q: q["wall_s"])[len(rr) // 2]  # the median run
    t = timers(r)
    scf = t.get("IterSCF", t.get("remag_device_cycle", 0.0))
    parts = [t.get("geom_init", 0.0), scf, t.get("PostSCF", 0.0)]
    parts.append(max(r["wall_s"] - sum(parts), 0.0))
    rows.append((name, parts, r["wall_s"], r["iterations"]))
summary["breakdown_largest_mesh"] = {n: dict(zip([p for p, _ in phases], parts), wall=w, iterations=it)
                                     for n, parts, w, it in rows}
prof = {}
for r in select(group="profile"):
    p = rundir(r) / "remag-device-scf.json"
    if not p.exists():
        continue

    def find(o, key):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == key:
                    yield v
                yield from find(v, key)
        elif isinstance(o, list):
            for v in o:
                yield from find(v, key)
    sp = next(find(json.loads(p.read_text()), "stage_profile"), None)
    if sp:
        prof[r["impl"]] = {k: v["seconds"] for k, v in sp["stages"].items()}
summary["gpu_stage_profile"] = prof
fig, (ax, bx) = plt.subplots(1, 2, figsize=(TWO, 3.1), gridspec_kw=dict(width_ratios=[1.35, 1]))
for i, (name, parts, wall, it) in enumerate(rows):
    left = 0
    for (ph, col), v in zip(phases, parts):
        share = 100 * v / sum(parts)
        ax.barh(i, share, left=left, color=col, edgecolor="white", lw=0.4, height=0.7,
                label=ph if i == 0 else None)
        left += share
    ax.text(102, i, f"{wall:.1f} s", va="center", fontsize=7)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([STYLE[n]["label"] for n, *_ in rows])
ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.set_xlabel(f"share of wall time, {nkmax:,} k points (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, handlelength=1.0)
ax.text(-0.62, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=9, family="STIXGeneral")
stage_names = [("assembly_and_eigensolver", "assembly and eigensolver"), ("occupations", "occupations"),
               ("density_and_normalization", "density matrix"), ("density_grid_hartree_xc", "mesh: density, Hartree and\nexchange-correlation potentials"),
               ("hamiltonian", "adding the mesh potential\nto the Hamiltonian")]
cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
for i, name in enumerate(n for n in ("gpu-bj-64", "gpu-batched-256") if n in prof):
    left = 0
    for (key, lab), col in zip(stage_names, cols):
        v = prof[name].get(key, 0.0)
        bx.barh(i, v, left=left, color=col, edgecolor="white", lw=0.4, height=0.6, label=lab if i == 0 else None)
        left += v
    rest = sum(prof[name].values()) - left
    bx.barh(i, rest, left=left, color="#DDDDDD", edgecolor="white", lw=0.4, height=0.6,
            label="other" if i == 0 else None)
    bx.text(left + rest + 0.2, i, f"{left + rest:.1f} s", va="center", fontsize=7)
bx.set_yticks(range(2)); bx.set_yticklabels(["block-Jacobi", "batched"])
bx.invert_yaxis()
bx.set_xlabel("GPU SCF time by stage (s)")
bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, handlelength=1.0)
bx.text(-0.22, 1.02, "b", transform=bx.transAxes, fontweight="bold", fontsize=9, family="STIXGeneral")
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "fig2_breakdown.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Figure 3: GPU batch width
lan = defaultdict(list)
for r in ok:
    if r["deck"] == "legacy" and r["nk"] == nkmax and r["impl"].startswith("gpu-") and r["group"] in ("lanes", "scaling"):
        fam, width = r["impl"].rsplit("-", 1)
        lan[(fam, int(width))].append(r)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(TWO, 2.3))
summary["lanes"] = {}
for fam, name in (("gpu-bj", "gpu-bj-64"), ("gpu-batched", "gpu-batched-256")):
    widths = sorted(w for f, w in lan if f == fam)
    if not widths:
        continue
    s = [stats([r["wall_s"] for r in lan[(fam, w)]]) for w in widths]
    m = [max(r["gpu_peak_mib"] or 0 for r in lan[(fam, w)]) / 1024 for w in widths]
    st = STYLE[name]
    y = np.array([q["median"] for q in s])
    ax.errorbar(widths, y, yerr=[[q["median"] - q["lo"] for q in s], [q["hi"] - q["median"] for q in s]],
                color=st["color"], marker=st["marker"], label=st["label"], capsize=1.5, elinewidth=0.6)
    bx.plot(widths, m, color=st["color"], marker=st["marker"], label=st["label"])
    summary["lanes"][fam] = {str(w): dict(q, gpu_peak_gib=g) for w, q, g in zip(widths, s, m)}
for a in (ax, bx):
    a.set_xscale("log", base=2)
    a.set_xlabel("k points diagonalized together on the GPU")
    a.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylabel("wall time to a converged solution (s)")
ax.set_ylim(bottom=0)
bx.set_ylabel("peak GPU memory in use (GiB)")
bx.set_ylim(bottom=0)
ax.legend(loc="upper right"); bx.legend(loc="upper right")
ax.text(-0.17, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=9, family="STIXGeneral")
bx.text(-0.17, 1.02, "b", transform=bx.transAxes, fontweight="bold", fontsize=9, family="STIXGeneral")
fig.tight_layout(w_pad=2)
fig.savefig(OUT / "fig3_batch_width.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Figure 4: CPU scaling of the k-point stage
thr = defaultdict(list)
for r in select(group="threads"):
    t = timers(r)
    its = t and r["iterations"]
    if not its or "compute_dm" not in t:
        continue
    if r["impl"].startswith("host-"):
        thr[("host", int(r["impl"][5:]))].append(t["compute_dm"] / r["iterations"])
    elif r["impl"].startswith("pristine-mpi-"):
        thr[("mpi", int(r["impl"].rsplit("-", 1)[1]))].append(t["compute_dm"] / r["iterations"])
    elif r["impl"] == "pristine-serial":
        thr[("mpi", 1)].append(t["compute_dm"] / r["iterations"])
        thr[("serial", 1)].append(t["compute_dm"] / r["iterations"])
fig, ax = plt.subplots(figsize=(ONE, 2.5))
summary["cpu_scaling"] = {}
for fam, name, lab in (("mpi", "pristine-mpi-28", "SIESTA 5.4.2, MPI processes"),
                       ("host", "host-28", "REMAG CPU, threads")):
    cores = sorted(c for f, c in thr if f == fam)
    if not cores:
        continue
    y = [statistics.median(thr[(fam, c)]) for c in cores]
    st = STYLE[name]
    ax.plot(cores, y, color=st["color"], marker=st["marker"], label=lab,
            mfc="white" if fam == "mpi" else st["color"])
    summary["cpu_scaling"][fam] = dict(zip(map(str, cores), y))
if ("serial", 1) in thr:
    t1 = statistics.median(thr[("serial", 1)])
    c = np.array([1, 28])
    ax.plot(c, t1 / c, color="0.6", lw=0.6, ls=":", label="SIESTA 1 core divided by cores")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks([1, 2, 4, 8, 16, 28]); ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.minorticks_off(); plain_log(ax.yaxis)
ax.set_xlabel("CPU cores")
ax.set_ylabel("diagonalization and density matrix\nper SCF iteration (s)")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "fig4_cpu_scaling.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Figure 5: agreement with pristine SIESTA
ref = {}
for r in select(group="scaling", impl="pristine-serial"):
    ref.setdefault(r["nk"], r)
quantities = [("free_ev", "free energy", 1e3, "meV"), ("etot_ev", "total energy", 1e3, "meV"),
              ("fermi_ev", "Fermi level", 1e3, "meV"), ("moment_mub", "spin moment", 1e3, "m$\\mu_B$")]
fig, axes = plt.subplots(1, len(quantities), figsize=(TWO, 2.0), sharey=True)
names = [n for n in ORDER if n != "pristine-serial"]
summary["agreement"] = {}
for ax, (key, lab, scale, unit) in zip(axes, quantities):
    worst = {}
    for j, name in enumerate(names):
        d = [(r[key] - ref[r["nk"]][key]) * scale for r in select(group="scaling", impl=name)
             if r["nk"] in ref and r[key] is not None and ref[r["nk"]][key] is not None]
        if not d:
            continue
        st = STYLE[name]
        jitter = (np.arange(len(d)) - (len(d) - 1) / 2) * 0.05
        ax.scatter(d, j + jitter, s=9, color=st["color"], marker=st["marker"], lw=0, zorder=3)
        worst[name] = max(abs(x) for x in d)
    summary["agreement"][key] = worst
    res = {"moment_mub": 1e-5}.get(key, 1e-6) * scale  # last printed digit
    ax.axvspan(-res, res, color="0.9", zorder=0)
    ax.axvline(0, color="0.5", lw=0.5)
    ax.set_xlabel(f"{lab}, difference ({unit})")
    lim = max([res * 6] + [v * 1.3 for v in worst.values()])
    ax.set_xlim(-lim, lim)
axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels([STYLE[n]["label"] for n in names])
axes[0].invert_yaxis()
fig.tight_layout(w_pad=0.8)
fig.savefig(OUT / "fig5_agreement.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Figure 6: REMAG-only methods
meth = defaultdict(list)
for r in select(group="methods"):
    meth[(r["impl"], r["deck"])].append(r)
kinds = [("lf", "local-frame PBE"), ("mc", "multicollinear PBE"), ("tet", "local-frame PBE,\ntetrahedra")]
fig, ax = plt.subplots(figsize=(ONE, 2.3))
summary["methods"] = {}
for i, (kind, lab) in enumerate(kinds):
    for name in ("host-28", "gpu-batched-256"):
        rr = meth.get((name, kind), [])
        if not rr:
            continue
        s_ = stats([r["wall_s"] for r in rr])
        st = STYLE[name]
        ax.errorbar(i, s_["median"], yerr=[[s_["median"] - s_["lo"]], [s_["hi"] - s_["median"]]],
                    color=st["color"], marker=st["marker"], ms=5, capsize=1.5, elinewidth=0.6, ls="none",
                    label=st["label"] if i == 0 else None)
        summary["methods"].setdefault(kind, {})[name] = dict(s_, free_ev=[r["free_ev"] for r in rr],
                                                              iterations=[r["iterations"] for r in rr])
    m = summary["methods"].get(kind, {})
    if "host-28" in m and "gpu-batched-256" in m:
        lo, hi = m["gpu-batched-256"]["median"], m["host-28"]["median"]
        ax.plot([i, i], [lo * 1.15, hi / 1.15], color="0.7", lw=0.6, zorder=0)
        ax.text(i + 0.08, (lo * hi) ** 0.5, f"{hi / lo:.0f}$\\times$", va="center", fontsize=7)
ax.set_yscale("log"); plain_log(ax.yaxis)
ax.set_xlim(-0.5, len(kinds) - 0.5)
ax.set_xticks(range(len(kinds))); ax.set_xticklabels([l for _, l in kinds])
ax.set_ylabel(f"wall time, {nkmax:,} k points (s)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
fig.tight_layout()
fig.savefig(OUT / "fig6_methods.pdf"); plt.close(fig)

(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1)[:6000])

# ---------------------------------------------------------------- LaTeX macros for every number the text quotes
macros = {}


WORDS = {"1": "One", "2": "Two", "4": "Four", "8": "Eight", "16": "Sixteen", "28": "TwentyEight",
         "32": "ThirtyTwo", "64": "SixtyFour", "128": "OneTwentyEight", "256": "TwoFiftySix",
         "512": "FiveTwelve", "1024": "TenTwentyFour"}


def mac(name, value, fmt):
    """LaTeX command names cannot hold digits, so a numeric suffix is spelled out."""
    import re as _re
    m = _re.match(r"(.*?)(\d+)$", name)
    if m:
        name = m.group(1) + WORDS[m.group(2)]
    macros[name] = format(value, fmt)


def med(name, nk):
    return summary["scaling"].get(name, {}).get(str(nk), {}).get("median")


for name, tag in (("pristine-serial", "Serial"), ("pristine-mpi-28", "Mpi"), ("host-28", "Host"),
                  ("gpu-bj-64", "Bj"), ("gpu-batched-256", "Batched")):
    for nk, ktag in zip(nks, "ABCD"):
        v = med(name, nk)
        if v is not None:
            mac(f"t{tag}{ktag}", v, ".1f" if v < 100 else ".0f")
for key, v in summary.get("speedup", {}).items():
    name, base = key.split("/")
    tag = {"gpu-batched-256": "Batched", "host-28": "Host"}[name] + {"pristine-serial": "Serial", "pristine-mpi-28": "Mpi"}[base]
    for nk, ktag in zip(nks, "ABCD"):
        if str(nk) in v:
            mac(f"s{tag}{ktag}", v[str(nk)], ".0f" if v[str(nk)] >= 10 else ".1f")
for name, tag in (("gpu-bj-64", "Bj"), ("gpu-batched-256", "Batched")):
    p = prof.get(name)
    if p:
        mac(f"prof{tag}Total", sum(p.values()), ".1f")
        mac(f"prof{tag}Eig", p["assembly_and_eigensolver"], ".1f")
        mac(f"prof{tag}Dens", p["density_and_normalization"], ".1f")
for name, tag in (("gpu-batched-256", "Batched"), ("gpu-bj-64", "Bj"), ("host-28", "Host"),
                  ("pristine-mpi-28", "Mpi"), ("pristine-serial", "Serial")):
    b = summary["breakdown_largest_mesh"].get(name)
    if b:
        mac(f"setup{tag}", b["setup"], ".1f"); mac(f"scf{tag}", b["SCF"], ".1f" if b["SCF"] < 100 else ".0f")
for fam, tag in (("gpu-bj", "Bj"), ("gpu-batched", "Batched")):
    for w, q in summary["lanes"].get(fam, {}).items():
        mac(f"w{tag}{w}", q["median"], ".1f"); mac(f"m{tag}{w}", q["gpu_peak_gib"], ".1f")
for fam, tag in (("host", "Host"), ("mpi", "Mpi")):
    for c, v in summary.get("cpu_scaling", {}).get(fam, {}).items():
        mac(f"c{tag}{c}", v, ".2f")
for key, worst in summary["agreement"].items():
    for name, v in worst.items():
        tag = {"pristine-mpi-28": "Mpi", "host-28": "Host", "gpu-bj-64": "Bj", "gpu-batched-256": "Batched"}[name]
        mac(f"d{key.split('_')[0]}{tag}", v, ".3f")
for kind, m in summary["methods"].items():
    for name, q in m.items():
        tag = {"host-28": "Host", "gpu-batched-256": "Batched"}[name]
        mac(f"meth{kind}{tag}", q["median"], ".0f" if q["median"] >= 100 else ".1f")
(OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
print(len(macros), "macros")

# ---------------------------------------------------------------- table of converged results (identical across implementations)
rows_t, identical = [], True
for nk in nks:
    vals = {(r["etot_ev"], r["free_ev"], r["fermi_ev"], r["moment_mub"], r["iterations"])
            for r in select(group="scaling", nk=nk)}
    identical &= len(vals) == 1
    e, f, ef, m, it = sorted(vals)[0]
    mesh = next(r["kmesh"] for r in select(group="scaling", nk=nk))
    nruns = len(select(group="scaling", nk=nk))
    rows_t.append(f"${mesh[0]}\\times{mesh[1]}\\times{mesh[2]}$ & \\num{{{nk}}} & {nruns} & {it} & "
                  f"\\num{{{e:.6f}}} & \\num{{{f:.6f}}} & \\num{{{ef:.6f}}} & \\num{{{m:.5f}}} \\\\")
summary["all_scaling_runs_identical"] = identical
(OUT / "table_results.tex").write_text(
    "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
    "Mesh & k points & Runs & Iterations & Total energy & Free energy & Fermi level & Moment \\\\\n"
    "\\midrule\n" + "\n".join(rows_t) + "\n\\bottomrule\n\\end{tabular}\n")
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print("identical:", identical, "rows:", len(rows_t))
