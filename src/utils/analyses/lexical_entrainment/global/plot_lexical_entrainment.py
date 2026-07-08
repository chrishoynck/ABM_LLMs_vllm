"""Per-seed lexical-entrainment trajectories on MentalBERT embeddings.

Each saved run (``net.json``) is embedded with MentalBERT (mean-pooled per
sliding time window); the per-window mean-embedding trajectory is reduced to 2D
(PCA by default) and drawn as a scatter, every window a dot coloured by its mean
PHQ-9 (green->red). Runs are NOT averaged: one trajectory per seed. This is the
"entrainment plot" — panel (a) of ``vis.plot_embedding_PCA_runs`` — with the
assortativity / degree-weighted-PHQ-9 panel dropped.

The four settings are the calibrated, debiased configs SDA/SDC × undirected/
directed; columns/seeds default to 14 15 16 17 18.

Output (under ``plots/lexical_entrainment/`` at the repo root):

  * per-setting overlay  — all seeds of one setting in one shared-PCA axes
    (``<net>_<dir>_<emb>_<red>_overlay.png``), one PCA fit per setting on its
    pooled seeds, no seed legend.
  * per-seed grid        — topology (rows) x seed (cols); each panel is ONE seed
    in its OWN PCA (no pooling across seeds), so panels show each run's best-fit
    shape but are not comparable across panels
    (``entrainment_grid_perseed_<emb>_<red>.png``).
  * SDA+SDC shared map   — one PCA per direction pooling SDA & SDC (calibrated +
    high degree), so the two network types share axes; group = marker, colour =
    PHQ-9 (``entrainment_shared_sda_sdc_<emb>_<red>.png``). Disable with
    ``--no-shared``.

Run from the repo root with the project venv (sentence-transformers / torch)::

    PYTHONPATH=src .venv_vllm/bin/python \\
        -m utils.analyses.lexical_entrainment.global.plot_lexical_entrainment \\
        --scan data/networks_post/basis

The first run encodes tweets with MentalBERT (GPU recommended) and caches them
per-seed under ``plots/lexical_entrainment/cache/<net>_<dir>_<combo>/``; later
runs reuse the cache. ``--overwrite`` redraws existing figures; ``--reduction
umap`` and ``--sbert`` are escape hatches (the pipeline stays embedding-agnostic).
"""

import argparse
import gc
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe; we only save figures

# Allow running as a plain script as well as ``-m utils.analyses...``.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

import utils.metrics as metrics
import utils.visualization as vis
import utils.tools.reading_in as ri
from classes.network import _Network

# Plotting only needs each checkpoint's tweet text + PHQ-9 + connections, never
# the BERT regressor, so neutralise the ~30 s model load generate_network does
# for bert-mode runs (same trick as plot_network_evolution).
_Network._init_bert_components = lambda self, *a, **k: None


# Rows of the grid. (net, direction, debias-leaf, calibrated combo folder, label).
# The calibrated ("main") combos match plot_network_evolution.PHASE_TS_CALIBRATED.
SETTINGS = [
    ("sda", "undirected", "debiased", "2_1655_d4_5_dim5",    "SDA undirected"),
    ("sda", "directed",   "debiased", "2_1655_d4_5_dim5",    "SDA directed"),
    ("sdc", "undirected", "debiased", "4_9429_d8_2539_dim3", "SDC undirected"),
    ("sdc", "directed",   "debiased", "4_9429_d8_2539_dim3", "SDC directed"),
]

# Shared SDA+SDC mapping experiment: one PCA per direction pools these groups,
# so SDA and SDC sit in the same space. Group = (net, combo, marker, label);
# marker encodes group (dot colour stays free for PHQ-9). calibrated + high degree.
SHARED_GROUPS = [
    ("sda", "2_1655_d4_5_dim5",     "o", "SDA calibrated"),
    ("sda", "2_1655_d6_dim5",       "s", "SDA high degree"),
    ("sdc", "4_9429_d8_2539_dim3",  "^", "SDC calibrated"),
    ("sdc", "4_9429_d10_dim3",      "D", "SDC high degree"),
]
SHARED_DIRECTIONS = ["undirected", "directed"]


def _seed_paths(root, net, direction, deb, combo, rounds, num_agents, seeds):
    """Sorted (seed, net.json path) for the requested seeds of one topology."""
    rounds_dir = f"rounds{rounds}_N{num_agents}"
    out = []
    for s in seeds:
        p = os.path.join(root, net, direction, deb, combo, rounds_dir,
                         f"seed_{s}", "net.json")
        if os.path.exists(p):
            out.append((s, p))
        else:
            print(f"[skip] missing {p}")
    return out


def _encode_means(paths_with_seeds, num_steps, shift, mentalbert, cache_dir):
    """Per-run mean-embedding window matrices + per-window PHQ-9 (no reduction).

    Returns (seeds, means, phq9s): means[i] is (T, dim), phq9s[i] is (T,). The
    cache_dir keys files by seed only, so it MUST be unique per (net, direction,
    combo) — different combos share seed numbers but have different tweets.
    """
    seeds = [s for s, _ in paths_with_seeds]
    nets = [{"network": ri.generate_network(args=None, pipe=None, file_path=p)[0]}
            for _s, p in paths_with_seeds]

    os.makedirs(cache_dir, exist_ok=True)
    tweet_to_emb, dim = metrics.build_tweet_embedding_cache(
        nets, mentalbert=mentalbert, cache_dir=cache_dir)
    means, _vars, phq9s = metrics.mean_sbert_per_networks(
        model=None, all_networks=nets, num_steps=num_steps, shift=shift,
        tweet_to_emb=tweet_to_emb, embedding_dim=dim)

    del nets
    gc.collect()
    return seeds, means, [np.asarray(p) for p in phq9s]


def _reduce(means, reduction):
    """2D reduction of a list of (T, dim) matrices (one shared fit, per run out)."""
    if reduction == "umap":
        trajs, _reducer = metrics.reduce_dimensionality_umap(means, n_components=2)
        return trajs
    return metrics.reduce_dimensionality(means, n_components=2)


def _reduce_each(means, reduction):
    """Per-run reduction: each matrix is fit on ITSELF (its own PCA), no pooling.

    Used for the per-seed grid — every panel shows one run in its own best-fit 2D
    space, so the axes are not comparable across panels (that's the point).
    """
    return [_reduce([m], reduction)[0] for m in means]


def _load_setting(paths_with_seeds, num_steps, shift, mentalbert, cache_dir,
                  reduction):
    """Per-seed reduced trajectories + per-window PHQ-9 for one topology.

    Runs are kept separate (no averaging). The reducer is fit once on the pooled
    per-run window matrices, so all seeds of this topology share one 2D space.
    Returns (seeds, trajs, phq9s) — trajs[i] is (T, 2), phq9s[i] is (T,).
    """
    seeds, means, phq9s = _encode_means(
        paths_with_seeds, num_steps, shift, mentalbert, cache_dir)
    return seeds, _reduce(means, reduction), phq9s


def _run_shared(opts, out_dir, emb_name, emb_slug, red_name):
    """SDA+SDC shared-mapping experiment: one reduction per direction.

    For each direction, pools every SHARED_GROUPS combo (calibrated + high degree,
    both nets) into a single PCA so SDA and SDC share axes, then plots with group
    encoded by marker and colour by PHQ-9.
    """
    name = f"entrainment_shared_sda_sdc_{emb_slug}_{opts.reduction}"
    out_png = os.path.join(out_dir, name + ".png")
    if not opts.overwrite and os.path.exists(out_png):
        print(f"[skip] exists: {out_png}")
        return

    groups_per_col = {}
    for ci, direction in enumerate(SHARED_DIRECTIONS):
        pooled_means, meta = [], []
        for net, combo, marker, label in SHARED_GROUPS:
            paths = _seed_paths(opts.scan, net, direction, "debiased", combo,
                                opts.rounds, opts.num_agents, opts.seeds)
            if not paths:
                print(f"[skip] no runs for {label} {direction}")
                continue
            cache_dir = os.path.join(out_dir, "cache",
                                     f"{net}_{direction}_{combo}")
            print(f"\n[shared] {label} {direction}  ({len(paths)} seeds)")
            _seeds, means, phq9s = _encode_means(
                paths, opts.num_steps, opts.shift, opts.mentalbert, cache_dir)
            start = len(pooled_means)
            pooled_means.extend(means)
            meta.append((label, marker, phq9s, start, len(pooled_means)))
        if not pooled_means:
            continue
        # One shared reduction per direction (SDA + SDC pooled).
        reduced = _reduce(pooled_means, opts.reduction)
        groups_per_col[ci] = [
            {"label": label, "marker": marker,
             "trajs": reduced[a:b], "phq9s": phq9s}
            for (label, marker, phq9s, a, b) in meta]

    if groups_per_col:
        vis.plot_entrainment_shared(
            groups_per_col, col_titles=SHARED_DIRECTIONS,
            reduction=red_name, embedding=emb_name,
            path=out_dir, filename=name, save=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", default="data/networks_post/basis",
                    help="root holding <net>/<dir>/<deb>/<combo>/... runs")
    ap.add_argument("--seeds", nargs="+", type=int, default=[14, 15, 16, 17, 18],
                    help="seeds = grid columns")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--num_agents", type=int, default=100)
    ap.add_argument("--num_steps", type=int, default=35,
                    help="sliding-window size (tweets) for the embedding trajectory")
    ap.add_argument("--shift", type=int, default=5, help="sliding-window stride")
    ap.add_argument("--reduction", choices=["pca", "umap"], default="pca",
                    help="2D reduction for the universal per-topology mapping")
    emb = ap.add_mutually_exclusive_group()
    emb.add_argument("--mentalbert", dest="mentalbert", action="store_true",
                     help="embed with MentalBERT (default)")
    emb.add_argument("--sbert", dest="mentalbert", action="store_false",
                     help="embed with default SBERT instead of MentalBERT")
    ap.set_defaults(mentalbert=True)
    ap.add_argument("--out", default="plots/lexical_entrainment",
                    help="output directory (created if missing)")
    ap.add_argument("--overwrite", action="store_true",
                    help="redraw figures even if the PNG already exists")
    ap.add_argument("--no-shared", dest="shared", action="store_false",
                    help="skip the SDA+SDC shared-mapping experiment")
    ap.set_defaults(shared=True)
    opts = ap.parse_args()

    emb_name = "MentalBERT" if opts.mentalbert else "SBERT"
    emb_slug = "mentalbert" if opts.mentalbert else "sbert"
    red_name = opts.reduction.upper()
    out_dir = opts.out
    os.makedirs(out_dir, exist_ok=True)

    # Per-seed grid: rows = topology (the 4 settings), cols = seed. Each cell is
    # ONE seed in its OWN PCA (no pooling across seeds). The per-setting overlays
    # keep the shared-per-setting PCA.
    grid_trajs, grid_phq9 = {}, {}
    row_titles, col_seeds = [], None
    # 2x2 overlay grid (shared-per-setting PCA): rows SDA/SDC, cols undir/dir.
    overlay_rows, overlay_cols = ["sda", "sdc"], ["undirected", "directed"]
    overlay_cells, overlay_cphq9 = {}, {}

    for r, (net, direction, deb, combo, label) in enumerate(SETTINGS):
        row_titles.append(label)
        paths = _seed_paths(opts.scan, net, direction, deb, combo,
                            opts.rounds, opts.num_agents, opts.seeds)
        if not paths:
            print(f"[skip] no runs for {label} under {opts.scan}")
            continue

        cache_dir = os.path.join(out_dir, "cache", f"{net}_{direction}_{combo}")
        print(f"\n{'='*70}\n[setting] {label}  ({len(paths)} seeds)\n{'='*70}")
        seeds, means, phq9s = _encode_means(
            paths, opts.num_steps, opts.shift, opts.mentalbert, cache_dir)
        if col_seeds is None:
            col_seeds = seeds

        # Per-setting overlay: all seeds in ONE shared-PCA axes (same scale).
        shared_trajs = _reduce(means, opts.reduction)
        ov_name = f"{net}_{direction}_{emb_slug}_{opts.reduction}_overlay"
        ov_png = os.path.join(out_dir, ov_name + ".png")
        if opts.overwrite or not os.path.exists(ov_png):
            vis.plot_entrainment_overlay(
                shared_trajs, phq9s, seeds, row_title=label,
                reduction=red_name, embedding=emb_name,
                path=out_dir, filename=ov_name, save=True)
        else:
            print(f"[skip] exists: {ov_png}")

        # 2x2 overlay-grid cell (shared-per-setting PCA = the overlay trajectories).
        rc = (overlay_rows.index(net), overlay_cols.index(direction))
        overlay_cells[rc] = shared_trajs
        overlay_cphq9[rc] = phq9s

        # Per-seed grid cells: per-seed PCA (each seed fit on its own windows).
        perseed_trajs = _reduce_each(means, opts.reduction)
        for c in range(len(perseed_trajs)):
            grid_trajs[(r, c)] = perseed_trajs[c]
            grid_phq9[(r, c)] = phq9s[c]

    # Per-seed grid (topology x seed), each panel its own PCA.
    if grid_trajs:
        n_cols = max(c for _r, c in grid_trajs) + 1
        grid_name = f"entrainment_grid_perseed_{emb_slug}_{opts.reduction}"
        grid_png = os.path.join(out_dir, grid_name + ".png")
        if opts.overwrite or not os.path.exists(grid_png):
            cols = (col_seeds or opts.seeds)[:n_cols]
            vis.plot_entrainment_grid(
                grid_trajs, grid_phq9,
                row_titles=row_titles,
                col_titles=[f"seed {s}" for s in cols],
                reduction=red_name, embedding=emb_name,
                path=out_dir, filename=grid_name, save=True)
        else:
            print(f"[skip] exists: {grid_png}")

    # 2x2 overlay grid (each cell = one setting's seeds overlaid, own cmap).
    if overlay_cells:
        og_name = f"entrainment_overlay_2x2_{emb_slug}_{opts.reduction}"
        og_png = os.path.join(out_dir, og_name + ".png")
        if opts.overwrite or not os.path.exists(og_png):
            vis.plot_entrainment_overlay_grid(
                overlay_cells, overlay_cphq9,
                row_titles=[n.upper() for n in overlay_rows],
                col_titles=overlay_cols,
                reduction=red_name, embedding=emb_name,
                path=out_dir, filename=og_name, save=True)
        else:
            print(f"[skip] exists: {og_png}")

    # SDA+SDC shared-mapping experiment (one PCA per direction).
    if opts.shared:
        _run_shared(opts, out_dir, emb_name, emb_slug, red_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
