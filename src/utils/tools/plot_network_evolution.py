"""Regenerate the network-evolution figures for already-built (saved) runs.

A no-LLM, no-notebook driver for ``utils.network_evolution``: it loads saved
``net.json`` checkpoints (the same files ``experiment.ipynb`` reads via
``main(..., use_saved_network=0)`` / ``reading_in.generate_network``) and writes,
for each run, the CDS-evolution graph + the compressed critical-slowing-down
heatmaps + the 10-panel PHQ-9 network sequence (the same
``vis.print_subnetworks_phq9`` filmstrip as the notebook) next to the checkpoint
(in its ``plots/`` sub-directory), plus a CDS validation table to stdout.

Figures are written idempotently: a figure whose PNG already exists is skipped
(per figure, not per run — a run missing only one figure still gets that one
drawn), and a run whose figures all exist is not even loaded. Pass
``--overwrite`` to force a full redraw.

Two ways to select runs:

  * **Explicit parameters** — the same flags ``run_simulation.sh`` uses
    (``net``, ``--alpha/--degree/--dim`` for sda/sdc, ``--rounds``, ``--num_agents``,
    ``--seeds``). The on-disk path is resolved through ``PathManager`` exactly as in
    the simulation, so it lands on the standard checkpoints.

        PYTHONPATH=src .venv_vllm/bin/python -m utils.tools.plot_network_evolution \\
            sdc --alpha 4.9429 --degree 8.2539 --dim 3 \\
            --rounds 300 --num_agents 100 --seeds 14 15 16 17 18

  * **Scan a directory** — per-seed figures for every ``net.json`` under a root.
    By default this covers the same universe as ``--grid`` (rounds=300, with the
    ``debiased`` / ``old_debiased`` / ``old_pop`` / ``different_debias_settings``
    sub-trees skipped), so every run that gets a combo grid also gets its
    per-seed snapshot. Pass ``--scan_rounds 0`` and ``--exclude`` (no values) to
    plot every checkpoint, non-standard sub-folders (``debiased/``, ``init_0/`` …)
    and partial runs included:

        PYTHONPATH=src .venv_vllm/bin/python -m utils.tools.plot_network_evolution \\
            --scan data/networks_post/basis

  * **Per-combination grids** (``--grid``) — group the discovered seeds by
    parameter combination and write one combined 4-row grid per combo (mean
    PHQ-9 + assortativity lines on top, then SD / lag-1 autocorrelation / PHQ-9
    dot-grids; one column per seed; shared colourbar per row) into each combo's
    ``plots/``. Defaults to the fully-finished runs only (rounds=300) and skips
    the ``debiased`` / ``old_debiased`` / ``old_pop`` /
    ``different_debias_settings`` sub-trees (the standard runs now live under
    ``non_debiased/``, which is kept):

        PYTHONPATH=src .venv_vllm/bin/python -m utils.tools.plot_network_evolution \\
            --grid --scan data/networks_post/basis

Run from the repo root with the project venv (see network-sa-python-env):
``PYTHONPATH=src .venv_vllm/bin/python -m utils.tools.plot_network_evolution ...``
"""

import argparse
import gc
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe; we only save figures

# Allow running as a plain script as well as ``-m utils.tools...``.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import utils.network_evolution as nev
import utils.tools.reading_in as ri
from classes.network import _Network

# Plotting only reads saved per-round histories — never the BERT regressor — so
# neutralise the ~30 s GPU/CPU model load that generate_network triggers for
# bert-mode checkpoints.
_Network._init_bert_components = lambda self, *a, **k: None


def _seed_figure_prefixes(opts):
    """Per-seed figure prefixes ``visualize_run`` writes (for the skip pre-check)."""
    return ["cds_evolution", f"csd_heatmaps_w{opts.csd_window}",
            "network_snapshot_phq9"]


def _plot_one(file_path, opts):
    """Load one saved net.json and write the per-seed figures beside it."""
    # Save into a plots/ sub-folder of the checkpoint's own directory, matching
    # PathManager.get_run_directory(is_plot=True) for standard paths and
    # co-locating with the source for non-standard sub-folders.
    plot_dir = os.path.join(os.path.dirname(str(file_path)), "plots")

    # Plot-level skip, but cheap: if every figure for this seed is already on
    # disk, don't pay the (~80 MB) net.json load at all. The seed is read off
    # the path so we can decide before loading; --overwrite forces a redraw.
    m = re.search(r"seed_(\d+)", str(file_path))
    if not opts.overwrite and m:
        filename = f"_{m.group(1)}"
        if all(os.path.exists(os.path.join(plot_dir, f"{p}_{filename}.png"))
               for p in _seed_figure_prefixes(opts)):
            print(f"[skip] all figures exist for {file_path}")
            return

    print(f"\n{'='*70}\n[plot] {file_path}\n{'='*70}")
    network, _running_fracs, _fracs_dist_step = ri.generate_network(
        args=None, pipe=None, file_path=file_path
    )

    seed = getattr(network, "seed", None)
    filename = f"_{seed}" if seed is not None else "_run"

    nev.visualize_run(
        network, plot_dir, filename,
        phq9_interval=opts.check_point,
        csd_window=opts.csd_window,
        ngrams_path=opts.ngrams,
        smooth_window=opts.smooth_window,
        save=True, show=False,
        validate=not opts.no_validate,
        overwrite=opts.overwrite,
    )

    # Free the (large) per-agent histories before the next run.
    del network
    gc.collect()


def _explicit_paths(opts):
    """Resolve standard checkpoint paths from run_simulation.sh-style flags."""
    from utils.tools.path_manager import PathManager

    paths = []
    for seed in opts.seeds:
        args = argparse.Namespace(
            net=opts.net, alpha=opts.alpha, degree=opts.degree, dim=opts.dim,
            m=opts.m, p=opts.p, rounds=opts.rounds, num_agents=opts.num_agents,
            seed=seed, directed=opts.directed,
            enforce_ngrams=False, happy=False,
            sample_phq9=opts.sample_phq9, cap_phq9=opts.cap_phq9,
            phq9_threshold=opts.phq9_threshold, init_phq9_zero=opts.init_phq9_zero,
            # Selects the debiased/ vs non_debiased/ sub-tree (default: non_debiased).
            bias_table_path=opts.bias_table_path,
        )
        path = PathManager(args=args).get_full_network_path()
        if os.path.exists(path):
            paths.append(path)
        else:
            print(f"[skip] no checkpoint at {path}")
    return paths


def _iter_filtered_nets(root, exclude, only_rounds):
    """Yield ``net.json`` paths under `root`, filtered like grid mode.

    Drops paths with any `exclude` token as one of their path *segments* (segment
    match, not substring, so ``debiased`` does not also knock out
    ``non_debiased``), keeps only ``seed_*`` checkpoints, and — when `only_rounds`
    is set — only those whose ``rounds<N>_N`` component matches (so only
    fully-finished runs are kept). Shared by the per-seed ``--scan`` path and
    ``--grid`` so both cover the same universe of runs.
    """
    exclude = set(exclude)
    for path in glob.glob(os.path.join(root, "**", "net.json"), recursive=True):
        if exclude & set(path.split(os.sep)):
            continue
        seed_dir = os.path.dirname(path)
        if not os.path.basename(seed_dir).startswith("seed_"):
            continue
        if only_rounds:
            m = re.search(r"rounds(\d+)_N", os.path.dirname(seed_dir))
            if not (m and int(m.group(1)) == only_rounds):
                continue
        yield path


def _combos(root, exclude, only_rounds):
    """Map {combo_dir: [seed net.json]} under `root` (grid mode), with filtering.

    A combo dir is the parent of a ``seed_*`` directory; selection matches
    ``_iter_filtered_nets`` (excluded path segments / round count).
    """
    out = {}
    for path in _iter_filtered_nets(root, exclude, only_rounds):
        combo = os.path.dirname(os.path.dirname(path))  # parent of the seed_* dir
        out.setdefault(combo, []).append(path)
    return out


def _slug(combo_dir, root):
    return os.path.relpath(combo_dir, root).replace(os.sep, "_").replace(".", "_")


def _run_grids(opts):
    """Grid mode: one combined per-combination figure per parameter combo."""
    root = opts.scan or "data/networks_post/basis"
    combos = _combos(root, opts.exclude, opts.grid_rounds)
    if not combos:
        print(f"No combinations under {root} "
              f"(rounds={opts.grid_rounds or 'any'}, excluding {opts.exclude}).")
        return
    print(f"Found {len(combos)} parameter combination(s) under {root}.")
    n_ok = 0
    for combo_dir, paths in sorted(combos.items()):
        paths = sorted(paths)
        if len(paths) < opts.min_seeds:
            print(f"[skip] {combo_dir}: only {len(paths)} seed(s)")
            continue
        plot_dir = os.path.join(combo_dir, "plots")
        slug = _slug(combo_dir, root)
        # Plot-level skip: don't load all seeds just to redraw an existing grid.
        grid_png = os.path.join(plot_dir, f"param_grid_w{opts.csd_window}_{slug}.png")
        if not opts.overwrite and os.path.exists(grid_png):
            print(f"[skip] grid exists: {grid_png}")
            continue
        print(f"\n{'='*70}\n[grid] {combo_dir}  ({len(paths)} seeds)\n{'='*70}")
        try:
            nets = [ri.generate_network(args=None, pipe=None, file_path=p)[0]
                    for p in paths]
            nev.plot_param_combo_grid(
                nets, phq9_interval=opts.check_point, window=opts.csd_window,
                path=plot_dir, filename=slug, save=True, show=False,
                overwrite=opts.overwrite)
            n_ok += 1
            del nets
            gc.collect()
        except Exception as e:
            print(f"[error] {combo_dir}: {e}")
    print(f"\nDone: {n_ok}/{len(combos)} combo grid(s) written.")


# ──────────────────────────────────────────────────────────────────────────────
#  Phase-portrait mode  (--phase)
# ──────────────────────────────────────────────────────────────────────────────
#
# One gridded phase portrait per network type (sda / sdc): degree-weighted PHQ-9
# (x) vs PHQ-9 assortativity (y), traced over the run. The 2x2 grid is
#     rows -> directed / undirected      cols -> non_debiased / debiased
# and within each subplot every saved seed of every configuration is one line,
# coloured by configuration.
#
# Configuration -> human label is keyed by the on-disk combo folder (the
# ``<alpha>_<degree>_dim<dim>`` directory under each leaf), matching the thesis
# table. Edit PHASE_LABELS to rename / re-map; any unmatched folder falls back to
# a parameter-derived label. An ``init_0`` sub-tree (every agent starts at PHQ-9
# 0) is suffixed automatically. PHASE_LABEL_ORDER only fixes legend / colour
# order; labels not listed there are appended and still get a colour.

PHASE_LABELS = {
    # SDA (alpha_degree_dim folder -> label)
    ("sda", "2_1655_d4_5_dim5"): "calibrated (main)",
    ("sda", "2_1655_d6_dim5"):   "high degree",
    ("sda", "1_1655_d4_5_dim5"): "low C",
    ("sda", "1_1655_d3_dim5"):   "low C + low degree",
    ("sda", "2_1655_d0_dim5"):   "degree 0",
    # SDC. The high-PHQ-9 probe is the high PHQ-9 *assortativity* config; ρ is the
    # assortativity coefficient, so it's labelled "high PHQ-9$_\rho$".
    ("sdc", "4_9429_d8_2539_dim3"): "calibrated (main)",
    ("sdc", "4_9429_d10_dim3"):     "high degree",
    ("sdc", "8_0_d8_2539_dim2"):    r"high PHQ-9$_\rho$",
}

PHASE_LABEL_ORDER = [
    "calibrated (main)",
    "calibrated (main) (PHQ-9=0)",
    "high degree",
    r"high PHQ-9$_\rho$",
    "low C",
    "low C + low degree",
    "degree 0",
]

# Combos to keep OUT of the phase portraits (matched as path segments, so they
# are skipped wherever they appear). The init_0 (all start at PHQ-9 0) sub-tree
# and sda/.../1_1655_d4_5_dim5 (a bad "low C" run) are excluded by request, as
# are the SDA high-PHQ-9 probes (2_1655_d4_5_dim3, 1_1655_d4_5_dim3 "low C") —
# dropped from the figure by request. The SDC high-PHQ-9-assortativity probe
# (8_0_d8_2539_dim2) is kept (shown as "high PHQ-9$_\rho$").
PHASE_SKIP_SEGMENTS = ["init_0", "1_1655_d4_5_dim5",
                       "2_1655_d4_5_dim3", "1_1655_d4_5_dim3"]

# Per-network upper y-limit for the *directed* row (the undirected row keeps the
# default zoom). Lets each figure crop the directed assortativity to where the
# action is. None / missing -> autoscale.
PHASE_DIRECTED_YMAX = {"sda": 0.3, "sdc": 0.7}

# Per-network lower y-limit for the *directed* row (default -0.05). SDA needs more
# headroom below zero to show the directed assortativity dip.
PHASE_DIRECTED_YMIN = {"sda": -0.15}

# Centered rolling-mean window (in PHQ-9 assessments) applied to each trajectory.
PHASE_SMOOTH = 5


def _phase_label(net, combo_dir, leaf):
    """Human config label for a combo dir under ``leaf`` (a directed/debias leaf).

    The combo's first path segment relative to the leaf is the config folder; an
    ``init_0`` segment (all agents start at PHQ-9 0) adds a suffix.
    """
    rel = os.path.relpath(combo_dir, leaf).split(os.sep)
    cfg = rel[0]
    base = PHASE_LABELS.get((net, cfg))
    if base is None:  # fallback: make the folder readable (2_1655_d4_5_dim5 -> ...)
        base = cfg.replace("_dim", " dim").replace("_", ".")
    if "init_0" in rel:
        base = f"{base} (PHQ-9=0)"
    return base


# Configuration colours, taken from the sa_analyze.py palette (blue / orange /
# sea green / deep brown). "low C + low degree" gets the sea green that replaces
# the old tab10 green.
PHASE_PALETTE = ["#2e7ebc", "#d96907", "#2e8b57", "#8d2c03"]
PHASE_LABEL_COLORS = {
    "calibrated (main)":           "#2e7ebc",  # blue
    "calibrated (main) (PHQ-9=0)": "#2e7ebc",  # blue
    "high degree":                 "#d96907",  # orange
    "low C":                       "#2e8b57",  # sea green
    "low C + low degree":          "#2e8b57",  # sea green (SDA)
    r"high PHQ-9$_\rho$":          "#2e8b57",  # sea green (SDC)
    "degree 0":                    "#8d2c03",  # deep brown
}


def _phase_color_map(labels):
    """Stable {label: colour}: canonical order first, then any extras.

    Known labels use the fixed sa_analyze palette (``PHASE_LABEL_COLORS``); any
    unlisted label falls back to the next palette colour.
    """
    ordered = ([l for l in PHASE_LABEL_ORDER if l in labels]
               + [l for l in labels if l not in PHASE_LABEL_ORDER])
    out, extra = {}, 0
    for l in ordered:
        if l in PHASE_LABEL_COLORS:
            out[l] = PHASE_LABEL_COLORS[l]
        else:
            out[l] = PHASE_PALETTE[extra % len(PHASE_PALETTE)]
            extra += 1
    return out


# Degree-0 config folder (SDA). At degree 0 the network is edge-less, so its
# dynamics are geometry-independent and the same runs double as SDC's degree-0
# baseline (SDC has no degree-0 runs of its own).
PHASE_DEGREE0_COMBO = "2_1655_d0_dim5"


def _load_cell_trajs(net, leaf, exclude, grid_rounds, check_point,
                     only_combos=None):
    """Per-seed trajectories under a directed/debias ``leaf`` for ``net``.

    Each trajectory is the dict consumed by ``plot_phase_grid`` plus a
    ``baseline`` flag (True for edge-less degree-0 runs, which are
    topology-independent). ``only_combos`` (a set of config-folder names)
    restricts loading to those configs.
    """
    trajs = []
    for combo_dir, paths in sorted(_combos(leaf, exclude, grid_rounds).items()):
        cfg = os.path.relpath(combo_dir, leaf).split(os.sep)[0]
        if only_combos is not None and cfg not in only_combos:
            continue
        label = _phase_label(net, combo_dir, leaf)
        for p in sorted(paths):
            try:
                network = ri.generate_network(
                    args=None, pipe=None, file_path=p)[0]
                _, dw, assort = nev.phq9_dw_and_assortativity(
                    network, check_point)
                baseline = len(network.connections) == 0
                trajs.append({"label": label, "dw": dw,
                              "assort": assort, "baseline": baseline})
                del network
                gc.collect()
            except Exception as e:
                print(f"[error] {p}: {e}")
    return trajs


def _run_phase(opts):
    """Phase mode: one 2x2 phase-portrait grid per network type."""
    root = opts.scan or "data/networks_post/basis"
    nets = opts.phase_net or ["sda", "sdc"]
    rows = [("directed", "Directed"), ("undirected", "Undirected")]
    cols = [("non_debiased", "Non-debiased"), ("debiased", "Debiased")]
    # The leaf already fixes the debias level, so don't let it be excluded; add
    # the phase-only skips (init_0 + the bad low-C run).
    exclude = ([e for e in opts.exclude if e not in ("debiased", "non_debiased")]
               + PHASE_SKIP_SEGMENTS)

    for net in nets:
        plot_dir = os.path.join(root, net, "plots")
        out = nev.figure_path(plot_dir, net, "phase_dw_phq9_assort")
        if not opts.overwrite and out and os.path.exists(out):
            print(f"[skip] phase grid exists: {out}")
            continue

        print(f"\n{'='*70}\n[phase] {net.upper()}\n{'='*70}")
        cells, all_labels = {}, []
        for r, (ddir, _rt) in enumerate(rows):
            for c, (dbdir, _ct) in enumerate(cols):
                leaf = os.path.join(root, net, ddir, dbdir)
                if not os.path.isdir(leaf):
                    continue
                trajs = _load_cell_trajs(net, leaf, exclude, opts.grid_rounds,
                                         opts.check_point)
                for t in trajs:
                    if t["label"] not in all_labels:
                        all_labels.append(t["label"])
                print(f"  {ddir}/{dbdir}: {len(trajs)} seed trajectories")
                cells[(r, c)] = trajs

        # SDC has no degree-0 runs of its own, but degree 0 is edge-less and so
        # geometry-independent -> borrow SDA's degree-0 runs (one per debias
        # column). They're flagged baseline, so the broadcast below shows them in
        # both the directed and undirected rows.
        if net == "sdc":
            for c, (dbdir, _ct) in enumerate(cols):
                sda_leaf = os.path.join(root, "sda", "undirected", dbdir)
                if not os.path.isdir(sda_leaf):
                    continue
                borrowed = _load_cell_trajs(
                    "sda", sda_leaf, exclude, opts.grid_rounds, opts.check_point,
                    only_combos={PHASE_DEGREE0_COMBO})
                for t in borrowed:
                    if t["label"] not in all_labels:
                        all_labels.append(t["label"])
                cells[(1, c)] = cells.get((1, c), []) + borrowed
                print(f"  borrowed SDA degree-0 ({dbdir}): "
                      f"{len(borrowed)} trajectories")

        if not any(cells.values()):
            print(f"[skip] no runs found for {net} under {root}")
            continue

        # Edge-less baselines (degree-0) depend only on the column (debiased vs
        # not), so show every column's baselines in both the directed and
        # undirected rows.
        for c in range(len(cols)):
            col_baselines = [t for r in range(len(rows))
                             for t in cells.get((r, c), []) if t.get("baseline")]
            for r in range(len(rows)):
                non_base = [t for t in cells.get((r, c), [])
                            if not t.get("baseline")]
                cells[(r, c)] = non_base + col_baselines
        ymax = PHASE_DIRECTED_YMAX.get(net)
        ymin = PHASE_DIRECTED_YMIN.get(net, -0.05)
        row_ylims = [(ymin, ymax) if ymax is not None else None, None]
        nev.plot_phase_grid(
            cells, _phase_color_map(all_labels),
            row_titles=[rt for _, rt in rows],
            col_titles=[ct for _, ct in cols],
            suptitle="",            # no big figure title (by request)
            smooth=PHASE_SMOOTH, row_ylims=row_ylims,
            path=plot_dir, filename=net, save=True, show=False,
            overwrite=opts.overwrite)


# Calibrated ("main") config folder per network type (see PHASE_LABELS). The
# --phase_ts time-series grid uses only these.
PHASE_TS_CALIBRATED = {"sda": "2_1655_d4_5_dim5", "sdc": "4_9429_d8_2539_dim3"}

# The four columns of the --phase_ts grid: (directed?, debias-folder, title). The
# rounds=300 runs carry the full 0..300 trajectory, so one folder gives the whole
# time series; "unbiased" == debiased, "biased" == non_debiased.
PHASE_TS_COLS = [("directed",   "debiased",     "Directed\nunbiased"),
                 ("directed",   "non_debiased", "Directed\nbiased"),
                 ("undirected", "debiased",     "Undirected\nunbiased"),
                 ("undirected", "non_debiased", "Undirected\nbiased")]


def _aggregate_ts_cell(leaf, cfg, check_point):
    """Across-seed mean/SD PHQ-9 + assortativity time series for one cell.

    Loads every ``rounds300_N100`` seed of the calibrated config ``cfg`` under the
    directed/debias ``leaf`` (the rounds-300 run holds the full 0..300 trajectory),
    computes the per-assessment degree-weighted mean PHQ-9, unweighted mean PHQ-9
    and PHQ-9 assortativity for each seed, then reduces across seeds to mean ± SD
    (``np.nanstd``, matching the phase grid). Returns the aggregate dict consumed
    by :func:`nev.plot_phq9_assort_timeseries_grid`, or None when no seed loads.
    """
    seed_jsons = sorted(glob.glob(
        os.path.join(leaf, cfg, "rounds300_N100", "seed_*", "net.json")))
    dws, means, assorts, rounds_ref = [], [], [], None
    for p in seed_jsons:
        try:
            network = ri.generate_network(args=None, pipe=None, file_path=p)[0]
            cr, dw, assort = nev.phq9_dw_and_assortativity(network, check_point)
            _, mean, _ = nev.phq9_mean_and_assortativity(network, check_point)
            if rounds_ref is None:
                rounds_ref = cr
            dws.append(dw); means.append(mean); assorts.append(assort)
            del network
            gc.collect()
        except Exception as e:
            print(f"[error] {p}: {e}")
    if not dws:
        return None
    import numpy as np
    n = min(len(a) for a in dws + means + assorts + [rounds_ref])
    stack = lambda lst: np.vstack([np.asarray(a, float)[:n] for a in lst])
    DW, MN, AS = stack(dws), stack(means), stack(assorts)
    return {
        "t":          np.asarray(rounds_ref, float)[:n],
        "dw_mean":    np.nanmean(DW, 0), "dw_sd":     np.nanstd(DW, 0),
        "mean_mean":  np.nanmean(MN, 0), "mean_sd":   np.nanstd(MN, 0),
        "assort_mean": np.nanmean(AS, 0), "assort_sd": np.nanstd(AS, 0),
    }


def _run_phase_ts(opts):
    """Phase time-series mode: one combined 2x4 grid for the calibrated configs.

    Rows are the network types (SDA, SDC); columns are the four
    directed/undirected x debias conditions (see ``PHASE_TS_COLS``). Each cell is
    the across-seed PHQ-9 score + assortativity time series of that network's
    calibrated ("main") config.
    """
    root = opts.scan or "data/networks_post/basis"
    nets = opts.phase_net or ["sda", "sdc"]

    plot_dir = os.path.join(root, "plots")
    out = nev.figure_path(plot_dir, "calibrated", "ts_phq9_assort_grid")
    if not opts.overwrite and out and os.path.exists(out):
        print(f"[skip] phase-ts grid exists: {out}")
        return

    cells = {}
    for r, net in enumerate(nets):
        cfg = PHASE_TS_CALIBRATED.get(net)
        if cfg is None:
            print(f"[skip] no calibrated config for {net}")
            continue
        for c, (ddir, db, _ct) in enumerate(PHASE_TS_COLS):
            leaf = os.path.join(root, net, ddir, db)
            agg = _aggregate_ts_cell(leaf, cfg, opts.check_point)
            if agg is not None:
                cells[(r, c)] = agg
            print(f"  {net} {ddir}/{db}: {'ok' if agg is not None else 'MISSING'}")

    if not cells:
        print(f"[skip] no calibrated runs found under {root}")
        return
    nev.plot_phq9_assort_timeseries_grid(
        cells, row_titles=[n.upper() for n in nets],
        col_titles=[ct for *_, ct in PHASE_TS_COLS],
        path=plot_dir, filename="calibrated", save=True, show=False,
        overwrite=opts.overwrite)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Run selection
    parser.add_argument("net", nargs="?", choices=["sf", "r", "sda", "sdc"],
                        help="Network type (omit when using --scan).")
    parser.add_argument("--scan", metavar="ROOT", default=None,
                        help="Load every net.json found under ROOT (recursively), "
                             "ignoring the explicit network flags.")
    parser.add_argument("--grid", action="store_true",
                        help="Combined per-combination grids instead of per-seed "
                             "figures (groups seeds under --scan; default root "
                             "data/networks_post/basis).")
    parser.add_argument("--phase", action="store_true",
                        help="One gridded phase portrait per network type "
                             "(degree-weighted PHQ-9 vs PHQ-9 assortativity; rows "
                             "directed/undirected, cols non_debiased/debiased, one "
                             "line per seed coloured by configuration). Scans "
                             "--scan (default data/networks_post/basis).")
    parser.add_argument("--phase_ts", action="store_true",
                        help="One combined 2x4 time-series grid for the calibrated "
                             "configs (rows SDA/SDC; cols directed/undirected x "
                             "unbiased/biased): degree-weighted & unweighted mean "
                             "PHQ-9 (left axis) and PHQ-9 assortativity (right "
                             "axis) vs round, across-seed mean +/- SD. Scans "
                             "--scan (default data/networks_post/basis).")
    parser.add_argument("--phase_net", nargs="*", choices=["sda", "sdc"],
                        default=None,
                        help="Phase mode: restrict to these network types "
                             "(default: both sda and sdc).")
    parser.add_argument("--grid_rounds", type=int, default=300,
                        help="Grid mode: only combos with this round count "
                             "(0 = any). Default 300 = fully-finished runs.")
    parser.add_argument("--scan_rounds", type=int, default=300,
                        help="Per-seed --scan mode: only runs with this round "
                             "count (0 = any). Default 300 = fully-finished runs.")
    parser.add_argument("--exclude", nargs="*",
                        default=["debiased", "old_debiased", "old_pop",
                                 "different_debias_settings"],
                        help="Skip runs/combos that have any of these as a path "
                             "segment, in both --grid and per-seed --scan "
                             "(non_debiased is kept by default; pass --exclude "
                             "with no values to keep everything).")
    parser.add_argument("--min_seeds", type=int, default=1,
                        help="Grid mode: skip combos with fewer than this many seeds.")

    # Network-identifying parameters (mirror run_simulation.sh / llama_activate)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--degree", type=float, default=2)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--p", type=float, default=0.5, help="Random-network edge prob.")
    parser.add_argument("--m", type=int, default=1, help="Scale-free m.")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--num_agents", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[14, 15, 16, 17, 18])
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--init_phq9_zero", action="store_true",
                        help="Select the init_0 checkpoint variant.")
    parser.add_argument("--bias_table_path", default=None,
                        help="Explicit-path mode: select the debiased/ sub-tree by "
                             "passing the same bias table the run used (any existing "
                             "file). Default / 'none' resolves to non_debiased/.")
    parser.add_argument("--sample_phq9", type=float, default=None)
    parser.add_argument("--cap_phq9", action="store_true")
    parser.add_argument("--phq9_threshold", type=float, default=0)

    # Visualization options
    parser.add_argument("--check_point", type=int, default=10,
                        help="PHQ-9 update cadence in rounds (heatmap compression / "
                             "CSD subsampling). Must match the run's --check_point.")
    parser.add_argument("--csd_window", type=int, default=8,
                        help="Rolling window for critical slowing down, in PHQ-9 updates.")
    parser.add_argument("--smooth_window", type=int, default=None,
                        help="Rolling-mean window (rounds) for the CDS line; "
                             "defaults to --check_point.")
    parser.add_argument("--ngrams", default=nev.NGRAMS_PATH,
                        help="TSV of distorted-language n-grams.")
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip the per-PHQ-9-band CDS validation table.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redraw figures even if the PNG already exists "
                             "(default: skip any figure already on disk, per "
                             "figure — a run missing only one figure still gets "
                             "that one drawn).")
    opts = parser.parse_args()

    if opts.phase_ts:
        _run_phase_ts(opts)
        return

    if opts.phase:
        _run_phase(opts)
        return

    if opts.grid:
        _run_grids(opts)
        return

    if opts.scan:
        # Per-seed scan covers the same universe as --grid (same exclude set and
        # round filter), so every run that gets a combo grid also gets its
        # per-seed figures. Pass --scan_rounds 0 and --exclude (no values) for
        # ad-hoc plotting of every net.json under ROOT, debiased sub-trees and
        # partial runs included.
        paths = sorted(_iter_filtered_nets(opts.scan, opts.exclude,
                                           opts.scan_rounds))
        if not paths:
            parser.error(
                f"no net.json under {opts.scan} "
                f"(rounds={opts.scan_rounds or 'any'}, excluding {opts.exclude})")
    else:
        if opts.net is None:
            parser.error("provide a network type (e.g. 'sdc') or use --scan ROOT")
        paths = _explicit_paths(opts)
        if not paths:
            # No checkpoints for this parameter set is a normal "nothing to do"
            # outcome when looping run_simulation.sh configs — exit cleanly so a
            # `set -e` shell loop moves on to the next config.
            print("No matching checkpoints found for the given parameters — skipping.")
            return

    print(f"Found {len(paths)} checkpoint(s) to plot.")
    n_ok = 0
    for path in paths:
        try:
            _plot_one(path, opts)
            n_ok += 1
        except Exception as e:  # keep going over a batch even if one is corrupt
            print(f"[error] {path}: {e}")
    print(f"\nDone: {n_ok}/{len(paths)} run(s) plotted.")


if __name__ == "__main__":
    main()
