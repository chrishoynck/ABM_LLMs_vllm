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
            enforce_ngrams=False, depressed=False,
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
