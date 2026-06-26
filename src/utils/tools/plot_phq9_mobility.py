"""Network-level PHQ-9 *mobility* map: structure (init PHQ-9 assortativity x
clustering) coloured by how much agents move on the PHQ-9 axis.

Each saved run (``net.json``) becomes ONE point. Its y is always the clustering
coefficient C; x is one of two structural axes (one per column, see below):

    x = mean degree <k>               (2E/N undirected; out-degree E/N directed)
        OR initial PHQ-9 assortativity (numeric assortativity of the *undirected*
                                        graph on each agent's phq9[0])
    y = clustering coefficient C      (undirected average clustering for
                                        undirected runs; out-clustering (Fagiolo
                                        2007, out-degree based) for directed runs)
    colour = network mobility         (mean over agents of each agent's RMSSD --
                                        root mean square of successive PHQ-9
                                        changes on the 10-round update grid)

Both axes are recomputed *from the run data itself* (the graph in
``Connections`` + each agent's ``phq9`` series), not read from ``meta.json`` --
they reproduce ``meta.json``'s ``topology.clustering`` /
``topology.phq9_assort_initial`` / degree exactly (verified), so the figure is
self-contained and the computation is auditable.

Output is TWO figures -- one non-debiased, one debiased -- each a 2x2 grid: rows
are the two networks (SDA, SDC); columns are the two structure mappings
(mean degree -> C, and initial PHQ-9 assortativity -> C). Splitting debiasing
across figures (rather than columns) keeps the two structure mappings side by
side within each debiasing condition, and the two figures share identical axes
and colourbar so they compare directly.

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
from matplotlib.colors import (FuncNorm, LinearSegmentedColormap, LogNorm,
                               Normalize, PowerNorm)


# ── brand colourmaps (no white anywhere) ──────────────────────────────────────
# Paper palette: blue #2e7ebc, orange #d96907, sea green #2e8b57, brown #8d2c03.
# Sequential maps are ordered light->dark so magnitude reads monotonically.
BRAND_CMAPS = {
    # default: bright green -> brown -> bright orange (brightened brand_gbo so
    # the colour differences across the mobility range stay easy to read)
    "brand_go": ["#2ecc40", "#8d2c03", "#ff8c00"],
    # sea green -> brown -> orange
    "brand_gbo": ["#2e8b57", "#8d2c03", "#d96907"],
    # warm, single-family: light orange -> deep brown
    "brand_warm": ["#f3b15a", "#d96907", "#8d2c03"],
    # cool counterpart: light sea green -> blue -> deep blue
    "brand_cool": ["#7fc6a3", "#2e8b57", "#2e7ebc", "#1c4f77"],
    # uses all four brand hues, ordered by luminance (orange>blue>green>brown)
    "brand_full": ["#d96907", "#2e7ebc", "#2e8b57", "#5a1c02"],
    # requested ramp: blue -> sea green -> brown -> orange. NOT luminance-ordered
    # (big hue jumps between adjacent steps), which is the point: near-equal
    # values land on visibly different hues, so similar runs separate more than
    # they would on a smooth single-family map. Pair with --rank-color for the
    # strongest separation; magnitude reads less monotonically as the trade-off.
    # Endpoints brightened (blue #2e7ebc->#2e9bef, orange #d96907->#ffa500) for
    # more punch; green brightened (#2e8b57->#2ecc71); brown darkened
    # (#8d2c03->#5a1c02) so the mid-ramp reads deeper against the bright ends.
    # The lighter orange end gives the dense high-mobility cluster more spread.
    "brand_bsbo": ["#2e9bef", "#2ecc71", "#5a1c02", "#ffa500"],
    # flat-UI green -> amber -> red (kept as an option). The default colour map
    # is matplotlib's built-in "RdYlGn_r", called by name (see make_cmap).
    "green_red": ["#2ecc71", "#f1c40f", "#e74c3c"],
}

# default: brand_bsbo with its blue start dropped, so the ramp begins at green
# (green #2ecc71 -> brown #5a1c02 -> orange #ffa500) -- the exact same bright
# hues as brand_bsbo, just without the leading blue.
BRAND_CMAPS["brand_gbo_bright"] = BRAND_CMAPS["brand_bsbo"][1:]


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


# PHQ-9 is re-evaluated only every 10 rounds; the stored series repeats each
# value 10x in between, so mobility is computed on the genuine update grid
# (``series[::PHQ9_UPDATE_EVERY]``) rather than the filler-inflated raw series.
PHQ9_UPDATE_EVERY = 10


def _rmssd(series):
    """Mobility of one PHQ-9 trajectory = RMSSD on its update grid.

    RMSSD (root mean square of successive differences) =
    ``sqrt(mean((x_{t+1} - x_t)**2))`` over the down-sampled update points. It is
    the root one-step mean-squared displacement -- i.e. the typical step size of
    the symptom score over time. Unlike net displacement it counts in-trajectory
    excursions (an agent that returns to its start still scores high), and unlike
    overall variance it is order-sensitive and not dominated by dwell time.
    Edge case: a series with <2 update points has no successive difference -> 0.
    """
    x = np.asarray(series, float)[::PHQ9_UPDATE_EVERY]
    if x.size < 2:
        return 0.0
    d = np.diff(x)
    return float(np.sqrt(np.mean(d * d)))


def _grid_var(series):
    """Variance of one PHQ-9 trajectory on its update grid.

    Order-insensitive spread around the agent's own mean -- the *total*
    excursion of the symptom score, regardless of path. Contrast with ``_rmssd``
    (step-to-step movement). In squared PHQ-9 units, so its across-agent spread
    is more right-skewed than RMSSD's; prefer --rank-color / --log-color when
    colouring by it. Edge case: <2 update points -> 0.
    """
    x = np.asarray(series, float)[::PHQ9_UPDATE_EVERY]
    if x.size < 2:
        return 0.0
    return float(np.var(x))


def _out_clustering(DG):
    """Mean local *out*-clustering of a directed graph (Fagiolo 2007).

    For each node: the number of ordered out-neighbour pairs (j, k) closed by an
    arc j->k, divided by d_out*(d_out-1). Only the out-degree enters, so this is
    the clustering "calculated solely with out degree" -- the directed
    friend-of-a-friend measure on an influence network. Nodes with out-degree < 2
    contribute 0 (count_zeros convention), averaging over all nodes so the value
    stays comparable to ``nx.average_clustering`` on the undirected runs. This
    mirrors ``sa_network._out_clustering`` / ``reading_in``'s ``clustering_out``,
    so directed runs here use the same C as the rest of the directed analysis.
    """
    if DG.number_of_nodes() == 0:
        return 0.0
    A = nx.to_numpy_array(DG, nodelist=list(DG.nodes()))
    np.fill_diagonal(A, 0.0)                      # ignore self-loops
    dout = A.sum(axis=1)
    tri = np.einsum("ij,jk,ik->i", A, A, A)       # arcs among each node's out-nbrs
    den = dout * (dout - 1)                        # ordered out-neighbour pairs
    with np.errstate(invalid="ignore", divide="ignore"):
        per = np.where(den > 0, tri / den, 0.0)
    return float(np.mean(per))


def analyse_run(path, directed=False):
    """Compute (assortativity, clustering, mobility, n_agents) from one net.json.

    Assortativity is measured on the *undirected* projection of ``Connections``
    for every run (matching ``meta.json``'s ``phq9_assort_initial``). Clustering
    and mean degree follow the run's directedness: undirected runs use ordinary
    average clustering and degree 2E/N; directed runs use out-clustering (Fagiolo
    2007) and mean out-degree E/N -- i.e. ``meta.json``'s ``clustering_out``, the
    same convention as the sensitivity plots (on these graphs out-clustering runs
    ~half the undirected projection). Mobility is the mean over agents
    of each agent's RMSSD on the 10-round PHQ-9 update grid (see ``_rmssd``).
    Degree-0 (edge-less) runs are kept with C = assortativity = mean_degree = 0
    (no edges -> no clustering and no assortative structure, by convention).
    """
    with open(path) as fh:
        d = json.load(fh)
    agents = d["Agents"]

    # mobility: per-agent RMSSD on the update grid, then mean over agents
    per_agent_rmssd = [_rmssd(a["phq9"]) for a in agents]
    mobility = float(np.mean(per_agent_rmssd))

    # alternative colour metrics on the same update grid: per-agent variance and
    # SD of the PHQ-9 trajectory (order-insensitive total spread, vs RMSSD's
    # step size), averaged over agents. Selected at plot time via --color-metric.
    per_agent_var = [_grid_var(a["phq9"]) for a in agents]
    mobility_var = float(np.mean(per_agent_var))
    mobility_std = float(np.mean(np.sqrt(per_agent_var)))

    # structural axes, recomputed from the graph itself
    G = nx.Graph()
    G.add_nodes_from(a["id"] for a in agents)
    G.add_edges_from((u, v) for u, v in d["Connections"])
    if G.number_of_edges() == 0:
        clustering, assort, mean_degree = 0.0, 0.0, 0.0     # degree-0 run
    else:
        # assortativity stays on the undirected projection for both (matches
        # meta.json's phq9_assort_initial); only clustering/degree differ by type
        nx.set_node_attributes(G, {a["id"]: a["phq9"][0] for a in agents}, "p0")
        assort = float(nx.numeric_assortativity_coefficient(G, "p0"))
        if directed:
            # directed runs: out-clustering (Fagiolo 2007) and mean out-degree,
            # measured on the DiGraph -- C here is the out-degree-based clustering
            # (meta.json's `clustering_out`, ~half the undirected projection on
            # these graphs), matching the sensitivity plots rather than projecting
            DG = nx.DiGraph()
            DG.add_nodes_from(a["id"] for a in agents)
            DG.add_edges_from((u, v) for u, v in d["Connections"])
            clustering = _out_clustering(DG)
            mean_degree = DG.number_of_edges() / DG.number_of_nodes()
        else:
            # undirected runs: ordinary average clustering, mean degree 2E/N
            clustering = float(nx.average_clustering(G))
            mean_degree = 2.0 * G.number_of_edges() / G.number_of_nodes()

    return dict(assort=assort, clustering=clustering, mobility=mobility,
                mean_degree=float(mean_degree), n_agents=len(agents),
                mobility_median=float(np.median(per_agent_rmssd)),
                mobility_var=mobility_var, mobility_std=mobility_std)


def harvest(root, rounds, csv_path):
    """Scan all runs, write one row per run to csv_path, return list of rows."""
    rows = []
    for path, meta in iter_runs(root, rounds):
        res = analyse_run(path, directed=(meta["dir"] == "directed"))
        if res is None:
            print(f"[skip] no edges: {path}")
            continue
        row = {**meta, **res, "path": os.path.relpath(path, root)}
        rows.append(row)
        print(f"  {meta['net']} {meta['dir'][:3]} {meta['deb'][:3]} "
              f"{meta['combo']:<20} seed{meta['seed']:>3}  "
              f"assort={res['assort']:+.3f} C={res['clustering']:.3f} "
              f"mob={res['mobility']:.2f}")
    if rows:
        cols = ["net", "dir", "deb", "combo", "rounds", "N", "seed",
                "assort", "clustering", "mean_degree", "mobility",
                "mobility_median", "mobility_var", "mobility_std",
                "n_agents", "path"]
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
            # optional columns: absent in CSVs harvested before var/SD existed
            for k in ("mobility_var", "mobility_std"):
                if r.get(k, "") != "":
                    r[k] = float(r[k])
            rows.append(r)
    return rows


# ── plotting ──────────────────────────────────────────────────────────────────
# Each FIGURE matches an EXACT deb category. The legacy ``old_debiased`` batch is
# *not* a debiased counterpart of these runs (it has unmatched topologies, e.g.
# combo 3_4312_d6_dim4), so it is excluded entirely rather than folded into the
# debiased figure. Add a category here only if it is a genuine like-for-like set.
DEB_FIGS = [("non-debiased", "non_debiased"), ("debiased", "debiased")]


def _lim(vals, pad_frac=0.05):
    """(min, max) of vals expanded by a small fractional pad (tiny pad on ties)."""
    lo, hi = min(vals), max(vals)
    pad = pad_frac * ((hi - lo) or 1.0)
    return (lo - pad, hi + pad)


def plot(rows, out_path, deb_filter, cmap_name="RdYlGn_r", log_color=False,
         color_key="mobility", color_label="PHQ-9 mobility (RMSSD)",
         rank_color=False, power=None):
    # One figure per debias category (``deb_filter``). Grid rows = networks
    # (SDA, SDC); grid columns = the two structure mappings -- mean degree ->
    # clustering and (init) PHQ-9 assortativity -> clustering. Marker encodes the
    # run's directedness (star = directed, dot = undirected), independent of the
    # column.
    nets = [("sda", "SDA"), ("sdc", "SDC")]
    xcols = [("mean_degree", r"degree $\langle k \rangle$"),
             ("assort", r"(init) PHQ-9$_\rho$")]
    keep_debs = {cdeb for _, cdeb in DEB_FIGS}    # excludes old_debiased
    dir_markers = [("directed", "*", 72), ("undirected", "o", 22)]
    cmap = make_cmap(cmap_name)

    # colour scale spans only THIS figure's drawn runs (its debias category,
    # edge-bearing), so each figure uses the full colourmap for its own mobility
    # range. Consequence: the two figures get DIFFERENT colourbars -- read each
    # figure's own bar; cross-figure colour is no longer directly comparable (the
    # shared x/y axes, computed over both categories below, still are).
    cval = np.array([r[color_key] for r in rows
                     if r["deb"] == deb_filter and r["mean_degree"] != 0.0], float)
    vmin = float(cval[cval > 0].min()) if log_color else float(cval.min())
    vmax = float(cval.max())
    if rank_color and np.unique(cval).size > 1:
        # quantile / rank normalisation: colour by *order*, not magnitude, so the
        # full colourmap is used even when values cluster tightly -- this is the
        # real lever for separating "similar simulations". Colourbar ticks still
        # show true values (placed by rank). Forward maps value->[0,1] via the
        # empirical CDF; inverse is its interpolant. Overrides --log-color.
        v = np.unique(cval)
        q = np.linspace(0.0, 1.0, v.size)
        norm = FuncNorm((lambda x: np.interp(x, v, q),
                         lambda y: np.interp(y, q, v)),
                        vmin=float(v[0]), vmax=float(v[-1]))
    elif power is not None:
        # monotonic power scaling: colour stays magnitude-faithful (no rank
        # distortion) but gamma>1 gives the dense high band more of the palette,
        # gamma<1 the low band. gamma=1 is identical to linear.
        norm = PowerNorm(gamma=power, vmin=vmin, vmax=vmax)
    elif log_color:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # sharey="row": both columns of a network row share a clustering y-scale, but
    # each network row keeps its own range (SDA tops out lower). x is NOT shared
    # -- the two columns plot different x variables (degree vs assortativity).
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.3), sharey="row")
    sc = None
    for i, (net, nlabel) in enumerate(nets):
        # axis limits computed over BOTH debias categories (degree-0 edge-less
        # runs excluded and reported separately; old_debiased excluded too) so
        # the non-debiased and debiased figures share identical axes -- but only
        # this figure's category (deb_filter) is actually drawn.
        net_all = [r for r in rows if r["net"] == net and r["mean_degree"] != 0.0
                   and r["deb"] in keep_debs]
        net_sub = [r for r in net_all if r["deb"] == deb_filter]
        ylim = _lim([r["clustering"] for r in net_all]) if net_all else None
        for j, (xkey, xlabel) in enumerate(xcols):
            ax = axes[i, j]
            xlim = _lim([r[xkey] for r in net_all]) if net_all else None
            for dlabel, marker, msize in dir_markers:
                pts = [r for r in net_sub if r["dir"] == dlabel]
                if pts:
                    sc = ax.scatter([r[xkey] for r in pts],
                                    [r["clustering"] for r in pts],
                                    c=[r[color_key] for r in pts], cmap=cmap, norm=norm,
                                    s=msize, marker=marker, edgecolors="black",
                                    linewidths=0.3, alpha=0.9)
            if xlim:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
            ax.set_xlabel(xlabel, fontsize=9)   # label under both network rows
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.25)
        axes[i, 0].set_ylabel(f"{nlabel}\nClustering Coeff.", fontsize=9)
    fig.subplots_adjust(hspace=0.45)

    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.02)
        cb.set_label(color_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        if rank_color and np.unique(cval).size > 1:
            # the rank norm's default locator crams ticks near the compressed
            # ends (they overlap); place 5 ticks at evenly spaced *quantiles*
            # instead, labelled with the true values they sit at
            v = np.unique(cval)
            tvals = np.interp(np.linspace(0, 1, 5),
                              np.linspace(0, 1, v.size), v)
            cb.set_ticks(tvals)
            cb.set_ticklabels([f"{t:.2g}" for t in tvals])

    # marker key: star = directed, dot = undirected (colour still = mobility)
    from matplotlib.lines import Line2D
    mk = [Line2D([0], [0], marker="*", color="none", markerfacecolor="0.6",
                 markeredgecolor="black", markersize=12, label="directed"),
          Line2D([0], [0], marker="o", color="none", markerfacecolor="0.6",
                 markeredgecolor="black", markersize=6, label="undirected")]
    fig.legend(handles=mk, loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.92))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")
    plt.close(fig)


# colour-metric -> (csv column, colourbar label). All computed on the 10-round
# PHQ-9 update grid. rmssd = step-to-step movement; var/std = total spread.
COLOR_METRICS = {
    "rmssd": ("mobility", "PHQ-9 mobility (RMSSD)"),
    "var":   ("mobility_var", "PHQ-9 spread (variance)"),
    "std":   ("mobility_std", "PHQ-9 spread (SD)"),
}


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
    p.add_argument("--cmap", default="RdYlGn_r",
                   choices=list(BRAND_CMAPS) + ["RdYlGn_r", "viridis"],
                   help="colour map (default RdYlGn_r = matplotlib green->red; "
                        "brand_* are no-white paper palettes)")
    p.add_argument("--log-color", action="store_true",
                   help="log-scale the colour (the spread is right-skewed)")
    p.add_argument("--color-metric", default="rmssd", choices=list(COLOR_METRICS),
                   help="what the colour encodes: rmssd (step size, default), "
                        "var or std (total spread). var/std need a (re)harvested CSV")
    p.add_argument("--power-color", type=float, default=1.0, metavar="GAMMA",
                   help="colour scaling exponent (default 1.0 = honest linear). "
                        "gamma>1 is a monotonic PowerNorm that gives the dense "
                        "high band more of the palette (still magnitude-faithful) "
                        "but the bar reads nonlinearly; 1.0 keeps it easy to read")
    p.add_argument("--rank-color", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="quantile/rank colour norm (default off) -- spreads the "
                        "full colourmap by order, separating similar runs the most "
                        "but no longer magnitude-faithful (colour distance != value "
                        "distance). Overrides --power-color / --log-color")
    opts = p.parse_args()

    # norm precedence: rank (order-based) > power (monotonic) > log > linear.
    # default is linear (gamma 1.0 -> None -> Normalize); rank/log/gamma!=1 opt in.
    use_power = (None if (opts.rank_color or opts.log_color
                          or opts.power_color == 1.0)
                 else opts.power_color)

    color_key, color_label = COLOR_METRICS[opts.color_metric]

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

    if color_key not in rows[0]:
        raise SystemExit(
            f"--color-metric {opts.color_metric!r} needs column {color_key!r}, "
            f"absent from {csv_path} (harvested before it existed). "
            f"Re-run with --reharvest.")

    # one figure per debias category; within each, columns are the two structure
    # mappings (degree -> clustering, PHQ-9 assortativity -> clustering)
    for _, cdeb in DEB_FIGS:
        plot(rows, os.path.join(plots_dir, f"phq9_mobility_{cdeb}.png"),
             deb_filter=cdeb,
             cmap_name=opts.cmap, log_color=opts.log_color, power=use_power,
             color_key=color_key, color_label=color_label,
             rank_color=opts.rank_color)


if __name__ == "__main__":
    main()
