"""Exclusive accounting for SIESTA's hierarchical timer; never sum nested inclusive times."""
from collections import defaultdict

GROUPS = {
    "device electronic cycle": ("remag_device_cycle",),
    "diagonalization": ("cdiag", "rdiag", "remag_gpu"),
    "Hamiltonian construction": ("setup_h", "hsparse", "vmat", "hamilt", "kinefsm", "nlefsm"),
    "density update": ("rho", "new_dm", "denmat", "m_new_dm", "reofd", "c-buildd", "r-buildd", "remag_dm"),
    "sparse matrix operations": ("c-buildhs", "r-buildhs"),
    "mixing": ("mix", "pulay", "broyden"),
    "Poisson/Hartree": ("poisson", "poison", "fft", "hartree"),
    "XC": ("xc", "gxc", "cellxc", "gridxc"),
    "SOC/noncollinear operations": ("spinorb", "spin_orbit", "spinorbit"),
    "communication": ("mpi", "comm", "redistrib"),
    "forces": ("dfscf", "dnonlocal", "force"),
    "initialization": ("setup", "initmesh", "read"),
}

def category(name, inherited="miscellaneous"):
    name = name.lower()
    for group, prefixes in GROUPS.items():
        if any(name.startswith(p) for p in prefixes):
            return group
    return inherited

def summarize(tree, wall_seconds):
    components = defaultdict(lambda: {"wall_seconds": 0., "calls": 0})
    routines, iterations, anomalies = [], [], []
    def walk(name, node, parent_group="miscellaneous", path=""):
        group = category(name, parent_group)
        children = {k: v for k, v in node.items() if not k.startswith("_") and isinstance(v, dict)}
        inclusive = float(node["_time"])
        exclusive = inclusive - sum(float(v["_time"]) for v in children.values())
        if exclusive < -0.005:
            anomalies.append(f"Overlapping or inconsistent timers at {path}/{name}: {exclusive}s")
        exclusive = max(0., exclusive)  # time.json rounds to milliseconds
        components[group]["wall_seconds"] += exclusive
        components[group]["calls"] += int(node.get("_calls", 0))
        routines.append(dict(path=f"{path}/{name}", name=name, calls=node.get("_calls"),
                             inclusive_seconds=inclusive, exclusive_seconds=exclusive, component=group))
        if name.startswith("IterSCF"):
            iterations.append(dict(path=f"{path}/{name}", seconds=inclusive, calls=node.get("_calls", 1)))
        for key, val in children.items():
            walk(key, val, group, f"{path}/{name}")
    for name, node in tree.items():
        walk(name, node)
    for value in components.values():
        value["percent_runtime"] = 100 * value["wall_seconds"] / wall_seconds if wall_seconds > 0 else None
    scf_seconds = sum(t["seconds"] for t in iterations)
    count = sum(t["calls"] for t in iterations)
    return dict(components=dict(components), routines=routines, scf_iterations=iterations,
                scf_seconds=scf_seconds if count else None, scf_timer_calls=count,
                average_iteration_seconds=scf_seconds/count if count else None,
                timing_anomalies=anomalies)
