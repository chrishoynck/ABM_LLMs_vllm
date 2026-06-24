"""Network-level PHQ-9 *mobility* map: structure (init PHQ-9 assortativity x
clustering) coloured by how much agents move on the PHQ-9 axis.

Each saved run (``net.json``) becomes ONE point:

    x = initial PHQ-9 assortativity   (numeric assortativity of the *undirected*
                                        graph on each agent's phq9[0])
    y = clustering coefficient C      (average clustering of the undirected graph)
    colour = network mobility         (mean over agents of the temporal variance
                                        of that agent's PHQ-9 trajectory)

Both x and y are recomputed *from the run data itself* (the graph in
``Connections`` + each agent's ``phq9`` series), not read from ``meta.json`` --
they reproduce ``meta.json``'s ``topology.clustering`` /
``topology.phq9_assort_initial`` exactly (verified), so the figure is
self-contained and the computation is auditable.

Two panels side by side: SDA and SDC.

Run from the repo root with the project venv::

    PYTHONPATH=src .venv_vllm/bin/python -m utils.tools.plot_phq9_mobility \
        --scan data/networks_post/basis

Stage 1 harvests every run into a small CSV (cache); Stage 2 draws from it.
Re-running reuses the CSV unless ``--reharvest`` is passed.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize


# ── brand colourmaps (no white anywhere) ──────────────────────────────────────
# Paper palette: blue #2e7ebc, orange #d96907, sea green #2e8b57, brown #8d2c03.
# Sequential maps are ordered light->dark so magnitude reads monotonically.
BRAND_CMAPS = {
    # default: sea green -> brown -> orange
    "brand_gbo": ["#2e8b57", "#8d2c03", "#d96907"],
    # warm, single-family: light orange -> deep brown
    "brand_warm": ["#f3b15a", "#d96907", "#8d2c03"],
    # cool counterpart: light sea green -> blue -> deep blue
    "brand_cool": ["#7fc6a3", "#2e8b57", "#2e7ebc", "#1c4f77"],
    # uses all four brand hues, ordered by luminance (orange>blue>green>brown)
    "brand_full": ["#d96907", "#2e7ebc", "#2e8b57", "#5a1c02"],
}


def make_cmap(name):
    if name in BRAND_CMAPS:
        return LinearSegmentedColormap.from_list(name, BRAND_CMAPS[name])
    return plt.get_cmap(name)  # allow any matplotlib name as an escape hatch


# ── run discovery + per-run computation ───────────────────────────────────────
_PATH_RE = re.compile(
    r"/(?P<net>sd[ac])/(?P<dir>directed|undirected)/(?P<deb>[^/]+)/"
    r"(?P<combo>[^/]+)/rounds(?P<rounds>\d+)_N(?P<N>\d+)/seed_(?P<seed>\d+)/net\.json$"
)


def iter_runs(root, rounds):
    """Yield (path, meta-dict) for every ``net.json`` under root matching rounds."""
    for path in sorted(glob.glob(os.path.join(root, "**", "net.json"),
                                 recursive=True)):
        m = _PATH_RE.search(path.replace(os.sep, "/"))
        if not m:
            continue
        if rounds and int(m.group("rounds")) != rounds:
            continue
        yield path, m.groupdict()


def analyse_run(path):
    """Compute (assortativity, clustering, mobility, n_agents) from one net.json.

    The graph is the *undirected* projection of ``Connections`` (directed runs
    are projected so C and assortativity are computed the same way everywhere,
    matching how ``reading_in`` derives ``phq9_assort_initial``). Mobility is the
    mean over agents of the population variance of each agent's PHQ-9 trajectory.
    Degree-0 (edge-less) runs are kept with C = assortativity = mean_degree = 0
    (no edges -> no clustering and no assortative structure, by convention).
    """
    with open(path) as fh:
        d = json.load(fh)
    agents = d["Agents"]

    # mobility: per-agent temporal variance of the PHQ-9 series, then mean
    per_agent_var = [float(np.var(a["phq9"])) for a in agents]   # ddof=0
    mobility = float(np.mean(per_agent_var))

    # structural axes, recomputed from the graph itself
    G = nx.Graph()
    G.add_nodes_from(a["id"] for a in agents)
    G.add_edges_from((u, v) for u, v in d["Connections"])
    if G.number_of_edges() == 0:
        clustering, assort, mean_degree = 0.0, 0.0, 0.0     # degree-0 run
    else:
        nx.set_node_attributes(G, {a["id"]: a["phq9"][0] for a in agents}, "p0")
        clustering = float(nx.average_clustering(G))
        assort = float(nx.numeric_assortativity_coefficient(G, "p0"))
        mean_degree = 2.0 * G.number_of_edges() / G.number_of_nodes()

    return dict(assort=assort, clustering=clustering, mobility=mobility,
                mean_degree=float(mean_degree), n_agents=len(agents),
                mobility_median=float(np.median(per_agent_var)))


def harvest(root, rounds, csv_path):
    """Scan all runs, write one row per run to csv_path, return list of rows."""
    rows = []
    for path, meta in iter_runs(root, rounds):
        res = analyse_run(path)
        if res is None:
            print(f"[skip] no edges: {path}")
            continue
        row = {**meta, **res, "path": os.path.relpath(path, root)}
        rows.append(row)
        print(f"  {meta['net']} {meta['dir'][:3]} {meta['deb'][:3]} "
              f"{meta['combo']:<20} seed{meta['seed']:>3}  "
              f"assort={res['assort']:+.3f} C={res['clustering']:.3f} "
              f"mob={res['mobility']:.1f}")
    if rows:
        cols = ["net", "dir", "deb", "combo", "rounds", "N", "seed",
                "assort", "clustering", "mean_degree", "mobility",
                "mobility_median", "n_agents", "path"]
        with open(csv_path, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(str(r[c]) for c in cols) + "\n")
        print(f"\n[harvest] wrote {len(rows)} runs -> {csv_path}")
    return rows


def load_csv(csv_path):
    rows = []
    with open(csv_path) as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            vals = line.rstrip("\n").split(",")
            r = dict(zip(header, vals))
            for k in ("assort", "clustering", "mean_degree", "mobility",
                      "mobility_median"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


# ── plotting ──────────────────────────────────────────────────────────────────
def _is_debiased(deb):
    """debiased / old_debiased -> True (star); non_debiased -> False (circle)."""
    return not deb.startswith("non")


def plot(rows, out_path, xkey="assort", xlabel=r"(init) PHQ-9$_\rho$",
         cmap_name="brand_gbo", log_color=False, title=""):
    nets = [("sda", "SDA"), ("sdc", "SDC")]
    cmap = make_cmap(cmap_name)

    mob = np.array([r["mobility"] for r in rows], float)
    vmin = float(mob[mob > 0].min()) if log_color else float(mob.min())
    vmax = float(mob.max())
    norm = (LogNorm(vmin=vmin, vmax=vmax) if log_color
            else Normalize(vmin=vmin, vmax=vmax))

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1), sharey=True)
    sc = None
    for ax, (net, label) in zip(axes, nets):
        # degree-0 runs have no edges -> topology-independent baseline, shown in
        # both panels regardless of which tree (sda/sdc) they were filed under
        sub = [r for r in rows
               if r["net"] == net or r["mean_degree"] == 0.0]
        if not sub:
            ax.set_title(f"{label} (no runs)", fontsize=9)
            continue
        # debiased configs -> star, non-debiased -> circle
        for marker, msize, keep in (("o", 26, False), ("*", 90, True)):
            grp = [r for r in sub if _is_debiased(r["deb"]) == keep]
            if grp:
                sc = ax.scatter([r[xkey] for r in grp], [r["clustering"] for r in grp],
                                c=[r["mobility"] for r in grp], cmap=cmap, norm=norm,
                                s=msize, marker=marker, edgecolors="black",
                                linewidths=0.3, alpha=0.9)
        ax.set_title(f"{label}  ({len(sub)} runs)", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Clustering Coeff.", fontsize=9)

    # marker-shape legend (debiased vs non-debiased), colour-neutral
    handles = [
        plt.Line2D([], [], marker="*", linestyle="", markersize=9,
                   markerfacecolor="0.5", markeredgecolor="black", label="debiased"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=5,
                   markerfacecolor="0.5", markeredgecolor="black", label="non-debiased"),
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="best", framealpha=0.9)

    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.02)
        cb.set_label("PHQ-9 mobility (mean per-agent var.)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scan", default="data/networks_post/basis",
                   help="root to scan for net.json")
    p.add_argument("--rounds", type=int, default=300,
                   help="only runs with this round count (0 = all)")
    p.add_argument("--csv", default=None,
                   help="cache CSV path (default <scan>/plots/mobility_phq9.csv)")
    p.add_argument("--reharvest", action="store_true",
                   help="rebuild the CSV even if it exists")
    p.add_argument("--cmap", default="brand_gbo",
                   choices=list(BRAND_CMAPS) + ["viridis"],
                   help="colour map (brand_* are no-white paper palettes)")
    p.add_argument("--log-color", action="store_true",
                   help="log-scale the mobility colour (variance is right-skewed)")
    opts = p.parse_args()

    plots_dir = os.path.join(opts.scan, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    csv_path = opts.csv or os.path.join(plots_dir, "mobility_phq9.csv")

    if opts.reharvest or not os.path.exists(csv_path):
        rows = harvest(opts.scan, opts.rounds, csv_path)
    else:
        rows = load_csv(csv_path)
        print(f"[cache] loaded {len(rows)} runs from {csv_path} "
              f"(use --reharvest to rebuild)")
    if not rows:
        print("no runs found")
        return

    # two x-axis variants, same y (clustering) and colour (mobility)
    plot(rows, os.path.join(plots_dir, "phq9_mobility_structure.png"),
         xkey="assort", xlabel=r"(init) PHQ-9$_\rho$",
         cmap_name=opts.cmap, log_color=opts.log_color)
    plot(rows, os.path.join(plots_dir, "phq9_mobility_degree.png"),
         xkey="mean_degree", xlabel=r"degree $\langle k \rangle$",
         cmap_name=opts.cmap, log_color=opts.log_color)


if __name__ == "__main__":
    main()
