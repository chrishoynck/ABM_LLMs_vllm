"""Pretty, correct network-evolution figures for a single simulation run.

A focused replacement for the network-evolution plots that are scattered through
``visualization.py``. Everything here is driven off the *saved per-round data* of
one loaded run (``network.all_agents`` -> ``tweethistory`` / ``all_phq9_sumscores``),
so the figures can be regenerated from any checkpoint without re-running the LLM.

Two things were wrong with the old plots and are fixed here:

1.  **CDS prevalence was effectively always zero.** ``basis`` runs simulate with
    an empty n-gram set (``cds_dynamic`` off -> ``n_grams=[]`` in
    ``llama_activate.update_network``), so every *stored* distortion flag
    (``running_fracs`` / ``fracs_dist_step`` / ``agent.distorted_tweets``) is
    ``False``. We instead **recompute CDS from the raw tweet text** with the
    *validated* detector from ``tools/validate_cds.py`` (compiled, word-boundary,
    category-aware n-gram regexes -- the same logic ``validate_cds`` uses to show
    CDS rises with PHQ-9).

2.  **PHQ-9 only changes every ``check_point`` (default 10) rounds**, yet
    ``all_phq9_sumscores`` stores one (constant) value per round. Critical-slowing-
    down over a 6-round window therefore sat inside a flat block (variance 0,
    autocorrelation undefined). We subsample at the update cadence so one heatmap
    column == one real PHQ-9 update, and the rolling window spans genuine changes.

Paths / filenames come from ``PathManager`` exactly as in ``experiment.ipynb``::

    import utils.network_evolution as nev
    nev.visualize_run(network, plot_path, plot_filename,
                      phq9_interval=args.check_point, save=args.save)
"""

import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

from . import metrics
from .tools.format_config import FC

NGRAMS_PATH = "data/distorted_language_ngrams.tsv"


# ──────────────────────────────────────────────────────────────────────────────
#  CDS detection  (reuses the validated detector from tools/validate_cds.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_cds_patterns(ngrams_path=NGRAMS_PATH):
    """Return {category: compiled regex} using the validated ``validate_cds`` logic.

    ``validate_cds`` forces the headless ``Agg`` matplotlib backend at import time;
    we snapshot and restore the caller's backend (e.g. the notebook's inline
    backend) so importing it here doesn't silently kill inline figures.
    """
    import matplotlib
    backend = matplotlib.get_backend()
    from .tools.validate_cds import load_ngrams_by_category, compile_category_patterns
    if matplotlib.get_backend() != backend:
        try:
            matplotlib.use(backend)
        except Exception:
            pass
    return compile_category_patterns(load_ngrams_by_category(ngrams_path))


def tweet_cds_categories(text, patterns):
    """List of CDS categories whose n-grams appear in ``text`` (empty if none)."""
    if not text or text == FC.NO_CONTENT:
        return []
    return [cat for cat, pat in patterns.items() if pat.search(text)]


def _is_cds(text, patterns):
    """True if ``text`` is a real tweet containing at least one CDS n-gram."""
    if not text or text == FC.NO_CONTENT:
        return False
    return any(pat.search(text) for pat in patterns.values())


# ──────────────────────────────────────────────────────────────────────────────
#  Per-round CDS / PHQ-9 series from saved data
# ──────────────────────────────────────────────────────────────────────────────

def _min_T(network):
    """Number of rounds available across all agents (histories share length)."""
    return min(len(a.tweethistory) for a in network.all_agents)


def cds_fraction_per_round(network, patterns):
    """Per-round CDS prevalence, recomputed from tweet text.

    Returns a dict of length-T arrays:
        frac_cds  : (# active tweets containing CDS) / (# active tweets) that round
        n_active  : number of agents who actually tweeted that round
        n_cds     : number of those active tweets flagged as CDS
    Rounds with no active tweets get ``frac_cds = nan``.
    """
    T = _min_T(network)
    n_active = np.zeros(T)
    n_cds = np.zeros(T)
    for agent in network.all_agents:
        for r in range(T):
            tweet = agent.tweethistory[r]
            if tweet and tweet != FC.NO_CONTENT:
                n_active[r] += 1
                if _is_cds(tweet, patterns):
                    n_cds[r] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_active > 0, n_cds / n_active, np.nan)
    return {"frac_cds": frac, "n_active": n_active, "n_cds": n_cds}


def mean_phq9_per_round(network):
    """Population mean PHQ-9 per round (a step function: flat between updates)."""
    T = min(len(a.all_phq9_sumscores) for a in network.all_agents)
    mat = np.array([a.all_phq9_sumscores[:T] for a in network.all_agents], dtype=float)
    return mat.mean(axis=0)


def infer_phq9_interval(network, default=10):
    """Best-effort guess of the PHQ-9 update cadence from the score histories.

    Looks at the spacing between successive score *changes* across all agents and
    returns the most common gap. Falls back to ``default`` when scores never move
    (e.g. a 0-round or fully-flat run). Prefer passing ``args.check_point`` when
    you have it.
    """
    gaps = Counter()
    for agent in network.all_agents:
        s = agent.all_phq9_sumscores
        change_idx = [i for i in range(1, len(s)) if s[i] != s[i - 1]]
        for a, b in zip(change_idx, change_idx[1:]):
            gaps[b - a] += 1
    return gaps.most_common(1)[0][0] if gaps else default


def _rolling_mean(arr, window):
    """NaN-aware centered rolling mean of a 1-D array."""
    arr = np.asarray(arr, dtype=float)
    if window <= 1:
        return arr
    import pandas as pd
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().values


# ──────────────────────────────────────────────────────────────────────────────
#  Styling helpers
# ──────────────────────────────────────────────────────────────────────────────

# Palette shared with the SA / validation figures (blue accent, firebrick PHQ-9).
COL_CDS = "#1f77b4"
COL_PHQ9 = "#b22222"


def _style_axis(ax):
    ax.grid(alpha=0.3, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save(fig, save, path, filename, prefix, show, do_tight=True):
    if do_tight:
        fig.tight_layout()
    if save and path is not None and filename is not None:
        os.makedirs(str(path), exist_ok=True)
        out = os.path.join(str(path), f"{prefix}_{filename}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[network_evolution] saved {out}")
    if show:
        plt.show()
    plt.close(fig)


def figure_path(path, filename, prefix):
    """The PNG ``_save`` would write for (path, filename, prefix), or None."""
    if path is None or filename is None:
        return None
    return os.path.join(str(path), f"{prefix}_{filename}.png")


def _skip_existing(path, filename, prefix, overwrite, save):
    """Plot-level idempotency: True when the target PNG already exists.

    Checked per figure (not per run), so a run missing only one figure still
    gets that one drawn. Honoured only when actually saving and not
    overwriting, so interactive ``save=False`` / ``show=True`` callers always
    (re)draw.
    """
    if save and not overwrite:
        out = figure_path(path, filename, prefix)
        if out is not None and os.path.exists(out):
            print(f"[network_evolution] skip existing {out}")
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
#  Figure 1 — CDS evolution ("the evolvement graph")
# ──────────────────────────────────────────────────────────────────────────────

def plot_cds_evolution(network, patterns=None, ngrams_path=NGRAMS_PATH,
                       phq9_interval=10, smooth_window=None,
                       path="", filename="default", save=False, show=True,
                       overwrite=False):
    """CDS prevalence over time, with population mean PHQ-9 on a twin axis.

    Left axis  : fraction of *active* tweets containing a CDS n-gram, per round
                 (faint dots) plus a smoothed line (rolling mean).
    Right axis : population mean PHQ-9, sampled at each update (every
                 ``phq9_interval`` rounds), the cadence at which it can move.

    Args:
        patterns:      pre-loaded {category: regex}; loaded from ``ngrams_path`` if None.
        phq9_interval: PHQ-9 update cadence in rounds (pass ``args.check_point``).
        smooth_window: rolling-mean window in rounds for the CDS line. Defaults to
                       ``phq9_interval`` so the smoothing matches the update block.
    """
    if _skip_existing(path, filename, "cds_evolution", overwrite, save):
        return None
    if patterns is None:
        patterns = load_cds_patterns(ngrams_path)
    if smooth_window is None:
        smooth_window = max(2, int(phq9_interval))

    series = cds_fraction_per_round(network, patterns)
    frac = series["frac_cds"]
    rounds = np.arange(len(frac))
    smoothed = _rolling_mean(frac, smooth_window)

    phq9 = mean_phq9_per_round(network)
    upd = np.arange(0, len(phq9), max(1, int(phq9_interval)))  # update sample points

    fig, ax = plt.subplots(figsize=(7.5, 3.6))

    # CDS prevalence (left axis)
    ax.scatter(rounds, frac, s=10, color=COL_CDS, alpha=0.25, label="per-round CDS rate")
    ax.plot(rounds, smoothed, color=COL_CDS, lw=2.2,
            label=f"CDS rate (rolling {smooth_window})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Fraction of active tweets with CDS", color=COL_CDS)
    ax.tick_params(axis="y", labelcolor=COL_CDS)
    ax.set_ylim(0, 1)
    _style_axis(ax)

    # Mean PHQ-9 (right axis), sampled at the update cadence
    ax2 = ax.twinx()
    ax2.plot(upd, phq9[upd], "s--", color=COL_PHQ9, lw=1.6, ms=4,
             label="mean PHQ-9 (per update)")
    ax2.set_ylabel("Mean PHQ-9", color=COL_PHQ9)
    ax2.tick_params(axis="y", labelcolor=COL_PHQ9)
    lo, hi = np.nanmin(phq9[upd]), np.nanmax(phq9[upd])
    margin = max((hi - lo) * 0.15, 0.5)
    ax2.set_ylim(max(0, lo - margin), min(27, hi + margin))
    ax2.spines["top"].set_visible(False)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8,
              frameon=False)
    ax.set_title("Cognitive-distortion prevalence over the run")

    _save(fig, save, path, filename, "cds_evolution", show)
    return series


# ──────────────────────────────────────────────────────────────────────────────
#  Figure 2 — critical-slowing-down dot grid (one dot per PHQ-9 update)
# ──────────────────────────────────────────────────────────────────────────────

def _sorted_agents_by_final_phq9(network):
    """Agent IDs ordered low->high by final PHQ-9 (for stable row order)."""
    def final_score(agent):
        if agent.well_being and "phq9_sumscore" in agent.well_being:
            return agent.well_being["phq9_sumscore"]
        return agent.all_phq9_sumscores[-1] if agent.all_phq9_sumscores else 0
    return [a.ID for a in sorted(network.all_agents, key=final_score)]


def phq9_update_series(agent, interval, iterations):
    """PHQ-9 value at every assessment, with its round.

    ``all_phq9_sumscores`` stores the score that prompted each round's tweet, so
    sampling ``[::interval]`` yields the assessed value at rounds 0, interval,
    2*interval, … But the *final* assessment (the one run after the last round,
    e.g. round 300 of a 300-round run) updates ``well_being`` with no following
    round to record it in the array — so we append it explicitly. Without this the
    last update (and the score the agents actually finished at) is never plotted.

    Returns:
        rounds (np.ndarray), values (np.ndarray) of equal length.
    """
    values = [float(v) for v in agent.all_phq9_sumscores[::interval]]
    rounds = list(range(0, len(values) * interval, interval))
    last_round = rounds[-1] if rounds else 0
    if (iterations % interval == 0 and iterations > last_round
            and agent.well_being and agent.well_being.get("phq9_sumscore") is not None):
        values.append(float(agent.well_being["phq9_sumscore"]))
        rounds.append(iterations)
    return np.array(rounds), np.array(values, dtype=float)


def _csd_matrices(network, interval, window):
    """Per-agent PHQ-9 / rolling-SD / lag-1-AC matrices for the CSD figures.

    Rows are agents sorted low->high by final PHQ-9; columns are PHQ-9 assessments
    (rounds 0, interval, …, plus the final one). SD = sqrt(rolling variance) — sqrt
    is roughly linear in score points so it doesn't squash the low-variability
    agents the way variance (quadratic) does.

    Returns:
        cols_round (np.ndarray), sd_m, auto_m, phq9_m  (each n_agents x n_cols).
    """
    id_to_agent = {a.ID: a for a in network.all_agents}
    cols_round = None
    var_m, auto_m, phq9_m = [], [], []
    for aid in _sorted_agents_by_final_phq9(network):
        rounds, values = phq9_update_series(id_to_agent[aid], interval, network.iterations)
        if cols_round is None:
            cols_round = rounds
        variances, autocorrs = metrics.calculate_agent_cd(values, window)
        var_m.append(variances)
        auto_m.append(autocorrs)
        phq9_m.append(values)
    return (cols_round, np.sqrt(np.array(var_m, dtype=float)),
            np.array(auto_m, dtype=float), np.array(phq9_m, dtype=float))


def phq9_mean_and_assortativity(network, interval):
    """Population-mean PHQ-9 and PHQ-9 assortativity at each assessment.

    Both are plain per-assessment series (no binning / no aggregation over
    rounds). Mean is the unweighted population mean (not degree-weighted).
    Assortativity is ``nx.numeric_assortativity_coefficient`` on the static graph
    with each node's PHQ-9 at that assessment (same calculation as
    ``visualization.plot_phq9_assortativity``).

    Returns:
        cols_round (np.ndarray), mean_phq9 (np.ndarray), assort (np.ndarray).
    """
    graph, _ = metrics.build_network_graph(network)
    undirected = graph.to_undirected() if network.directed else graph

    series = {a.ID: phq9_update_series(a, interval, network.iterations)[1]
              for a in network.all_agents}
    cols_round = phq9_update_series(network.all_agents[0], interval, network.iterations)[0]
    n_cols = len(cols_round)

    mean_phq9 = np.full(n_cols, np.nan)
    assort = np.full(n_cols, np.nan)
    for c in range(n_cols):
        vals = np.array([series[a.ID][c] for a in network.all_agents], dtype=float)
        mean_phq9[c] = np.nanmean(vals)
        for a in network.all_agents:
            undirected.nodes[a.ID]["phq9"] = float(series[a.ID][c])
        try:
            assort[c] = nx.numeric_assortativity_coefficient(undirected, "phq9")
        except Exception:
            assort[c] = np.nan
    return cols_round, mean_phq9, assort


def plot_csd_heatmaps(network, phq9_interval=10, window=8,
                      path="", filename="default", save=False, show=True,
                      marker_size=None, overwrite=False):
    """Critical-slowing-down grid — one square per agent per PHQ-9 update.

    PHQ-9 is subsampled every ``phq9_interval`` rounds (``all_phq9_sumscores[::interval]``)
    so each sample is a genuine score change; rolling variance and lag-1
    autocorrelation are then computed over ``window`` *updates* (reusing
    ``metrics.all_agent_phq9_cd``). Nothing is ever computed across the flat
    no-update stretches between assessments.

    The updates are drawn adjacently (x = update index, no dead space between
    assessments) and labelled by their real round (0, 10, 20, …). Each
    (agent, update) is a square marker coloured by value; warm-up cells (rolling
    window not yet full) are simply absent.

    Args:
        phq9_interval: PHQ-9 update cadence in rounds (pass ``args.check_point``).
        window:        rolling window length, in updates (not rounds).
        marker_size:   square size in points^2; ``None`` auto-fills the column
                       width so the squares sit next to one another.
    """
    if _skip_existing(path, filename, f"csd_heatmaps_w{window}", overwrite, save):
        return None
    interval = max(1, int(phq9_interval))
    cols_round, sd_m, auto_m, phq9_m = _csd_matrices(network, interval, window)
    n_agents, n_cols = phq9_m.shape
    # Columns sit at consecutive integer x positions; tick labels map them back
    # to the real round in `cols_round`.
    cols = np.arange(n_cols)
    X = np.tile(cols, (n_agents, 1))
    Y = np.tile(np.arange(n_agents).reshape(-1, 1), (1, n_cols))

    # Small, narrow figure so each panel reads roughly square.
    fig, axes = plt.subplots(3, 1, figsize=(2.6, 5.4), sharex=True)

    panels = [
        (sd_m, "YlOrRd", "(a) Rolling SD", "SD", None, None),
        (auto_m, "RdBu_r", "(b) Lag-1 Autocorrelation", "AC1", -1, 1),
        (phq9_m, "RdYlGn_r", "(c) PHQ-9 score", "PHQ-9", 0, 27),
    ]
    handles = []
    for ax, (mat, cmap, caption, cbar_label, vmin, vmax) in zip(axes, panels):
        valid = ~np.isnan(mat)
        sc = ax.scatter(X[valid], Y[valid], c=mat[valid], cmap=cmap,
                        marker="s", linewidths=0, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(cbar_label, fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        ax.set_xlabel(caption, fontsize=8)   # panel title underneath
        ax.set_ylabel("Agents (by final PHQ-9)", fontsize=7)
        ax.set_yticks([])
        ax.set_ylim(-0.5, n_agents - 0.5)
        ax.set_xlim(-0.5, n_cols - 0.5)
        handles.append((ax, sc))

    # x ticks every ~50 rounds (whichever columns land on a 50-round multiple).
    ticks = [c for c in cols if int(cols_round[c]) % 50 == 0] or list(cols[::5])
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([str(int(cols_round[c])) for c in ticks], fontsize=7)
    for ax, _ in handles:
        ax.tick_params(axis="x", labelsize=7)

    # Size the squares to the full column pitch *after* layout so neighbouring
    # columns touch — no whitespace between columns.
    fig.tight_layout()
    fig.canvas.draw()
    for ax, sc in handles:
        if marker_size is not None:
            sc.set_sizes([marker_size])
            continue
        w_pts = ax.get_window_extent().width * 72.0 / fig.dpi
        side = (w_pts / max(n_cols, 1)) * 1.05   # fill column pitch -> no gaps
        sc.set_sizes([side ** 2])

    _save(fig, save, path, filename, f"csd_heatmaps_w{window}", show)
    return {"std": sd_m, "variance": sd_m ** 2,
            "autocorrelation": auto_m, "phq9": phq9_m}


def plot_param_combo_grid(networks, phq9_interval=10, window=8,
                          path="", filename="default", save=False, show=True,
                          overwrite=False):
    """One 4-row figure per parameter combination, one column per seed.

    Rows (top to bottom):
        0  mean PHQ-9 (left axis) + PHQ-9 assortativity (right axis), per
           assessment — two simple lines, no aggregation over rounds.
        1  rolling SD              (one square per agent per assessment)
        2  lag-1 autocorrelation   (")
        3  PHQ-9 score             (")

    Heatmap rows share a single colourbar on the right (common scale across all
    seeds in that row). No titles. Columns are seeds in ascending order.
    """
    if _skip_existing(path, filename, f"param_grid_w{window}", overwrite, save):
        return None
    interval = max(1, int(phq9_interval))
    networks = sorted(networks, key=lambda n: getattr(n, "seed", 0))
    n_seeds = len(networks)

    # Pre-compute everything per seed; track the shared SD colour scale.
    per_seed = []
    sd_max = 0.0
    for net in networks:
        cols_round, sd_m, auto_m, phq9_m = _csd_matrices(net, interval, window)
        _, mean_phq9, assort = phq9_mean_and_assortativity(net, interval)
        per_seed.append(dict(net=net, cols_round=cols_round, sd=sd_m, ac=auto_m,
                             phq9=phq9_m, mean=mean_phq9, assort=assort))
        if np.isfinite(np.nanmax(sd_m)):
            sd_max = max(sd_max, float(np.nanmax(sd_m)))
    cols_round = per_seed[0]["cols_round"]
    n_cols = len(cols_round)
    cols = np.arange(n_cols)
    # every 100 rounds — the seed columns are too narrow for a label every 50.
    ticks = [c for c in cols if int(cols_round[c]) % 100 == 0] or list(cols[::10])
    tick_labels = [str(int(cols_round[c])) for c in ticks]

    # Small figure: ~1.05" per seed column + a sliver for the colourbars.
    fig = plt.figure(figsize=(1.05 * n_seeds + 0.7, 4.8))
    gs = gridspec.GridSpec(4, n_seeds + 1, figure=fig,
                           width_ratios=[1] * n_seeds + [0.06],
                           left=0.09, right=0.92, top=0.98, bottom=0.07,
                           wspace=0.12, hspace=0.18)

    # heatmap row spec: (row, key, cmap, vmin, vmax, label)
    hm_rows = [(1, "sd", "YlOrRd", 0, sd_max or None, "SD"),
               (2, "ac", "RdBu_r", -1, 1, "AC1"),
               (3, "phq9", "RdYlGn_r", 0, 27, "PHQ-9")]

    square_handles = []   # (ax, sc, n_cols) for post-layout sizing

    for c, d in enumerate(per_seed):
        n_agents = d["phq9"].shape[0]
        X = np.tile(cols, (n_agents, 1))
        Y = np.tile(np.arange(n_agents).reshape(-1, 1), (1, n_cols))

        # ── row 0: mean PHQ-9 + assortativity lines ──
        ax0 = fig.add_subplot(gs[0, c])
        ax0.plot(cols, d["mean"], color=COL_PHQ9, lw=1.2)
        ax0.set_ylim(0, 27)
        ax0.set_xlim(-0.5, n_cols - 0.5)
        ax0.set_xticks(ticks)
        ax0.set_xticklabels([])
        ax0.tick_params(labelsize=6)
        axr = ax0.twinx()
        axr.plot(cols, d["assort"], color=COL_CDS, lw=1.2)
        axr.axhline(0, color="0.6", lw=0.5, ls="--")
        axr.set_ylim(-1, 1)
        axr.tick_params(labelsize=6)
        if c == 0:
            ax0.set_ylabel("mean PHQ-9", color=COL_PHQ9, fontsize=7)
            ax0.tick_params(axis="y", labelcolor=COL_PHQ9, labelsize=6)
        else:
            ax0.set_yticklabels([])
        if c == n_seeds - 1:
            axr.set_ylabel("assortativity", color=COL_CDS, fontsize=7)
            axr.tick_params(axis="y", labelcolor=COL_CDS, labelsize=6)
        else:
            axr.set_yticklabels([])

        # ── rows 1-3: dot-grid heatmaps ──
        for (r, key, cmap, vmin, vmax, _lbl) in hm_rows:
            ax = fig.add_subplot(gs[r, c])
            mat = d[key]
            valid = ~np.isnan(mat)
            sc = ax.scatter(X[valid], Y[valid], c=mat[valid], cmap=cmap,
                            marker="s", linewidths=0, vmin=vmin, vmax=vmax)
            ax.set_yticks([])
            ax.set_ylim(-0.5, n_agents - 0.5)
            ax.set_xlim(-0.5, n_cols - 0.5)
            ax.set_xticks(ticks)
            if r == 3:
                ax.set_xticklabels(tick_labels, fontsize=6)
                ax.tick_params(axis="x", labelsize=6)
            else:
                ax.set_xticklabels([])
            square_handles.append((ax, sc, n_cols))
            if c == 0:
                ax.set_ylabel(_lbl, fontsize=7)
            d[f"_sc_{r}"] = sc   # last seed's handle is fine for the shared bar

    # one colourbar per heatmap row (shared scale across seeds)
    for (r, key, cmap, vmin, vmax, lbl) in hm_rows:
        cax = fig.add_subplot(gs[r, n_seeds])
        cbar = fig.colorbar(per_seed[-1][f"_sc_{r}"], cax=cax)
        cbar.set_label(lbl, fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    # size the squares to fill the column pitch (no whitespace) after layout.
    # GridSpec margins are fixed above, so skip _save's tight_layout (which would
    # reflow the axes and break the fit).
    fig.canvas.draw()
    for ax, sc, ncol in square_handles:
        w_pts = ax.get_window_extent().width * 72.0 / fig.dpi
        side = (w_pts / max(ncol, 1)) * 1.05
        sc.set_sizes([side ** 2])

    _save(fig, save, path, filename, f"param_grid_w{window}", show, do_tight=False)


# ──────────────────────────────────────────────────────────────────────────────
#  CDS validation  (does CDS rise with PHQ-9 *within this run*?)
# ──────────────────────────────────────────────────────────────────────────────

def cds_validation_summary(network, patterns=None, ngrams_path=NGRAMS_PATH,
                           verbose=True):
    """Validate the recomputed CDS signal against PHQ-9, per (agent, round) event.

    Mirrors ``tools/validate_cds.py`` but over the *simulation* rather than the
    training corpus: every active tweet is one event tagged with the agent's PHQ-9
    that round and whether the tweet contains CDS. Reports % CDS per PHQ-9 severity
    band and the point-biserial correlation, so you can confirm the in-run signal
    behaves like the validated training-data signal (CDS more likely at higher PHQ-9).

    Returns a dict with ``per_band`` (list of (band, n, pct)), ``r_pointbiserial``
    and overall ``pct_cds``.
    """
    if patterns is None:
        patterns = load_cds_patterns(ngrams_path)

    bands = [(0, 4, "minimal"), (5, 9, "mild"), (10, 14, "moderate"),
             (15, 19, "mod.severe"), (20, 27, "severe")]

    def band_of(score):
        for lo, hi, name in bands:
            if lo <= score <= hi:
                return name
        return "unknown"

    T = _min_T(network)
    phq9_vals, cds_vals = [], []
    for agent in network.all_agents:
        scores = agent.all_phq9_sumscores
        for r in range(min(T, len(scores))):
            tweet = agent.tweethistory[r]
            if tweet and tweet != FC.NO_CONTENT and scores[r] is not None:
                phq9_vals.append(float(scores[r]))
                cds_vals.append(1.0 if _is_cds(tweet, patterns) else 0.0)

    phq9_vals = np.array(phq9_vals)
    cds_vals = np.array(cds_vals)

    per_band = []
    for lo, hi, name in bands:
        m = np.array([band_of(p) == name for p in phq9_vals])
        n = int(m.sum())
        pct = 100.0 * cds_vals[m].mean() if n else float("nan")
        per_band.append((name, n, pct))

    if len(phq9_vals) > 1 and cds_vals.std() > 0:
        r = float(np.corrcoef(phq9_vals, cds_vals)[0, 1])
    else:
        r = float("nan")
    overall = 100.0 * cds_vals.mean() if len(cds_vals) else float("nan")

    if verbose:
        print("CDS validation over the run (active tweets only)")
        print("-" * 46)
        print(f"{'band':<12}{'n':>8}{'% CDS':>10}")
        for name, n, pct in per_band:
            print(f"{name:<12}{n:>8}{pct:>9.1f}%")
        print(f"{'all':<12}{len(cds_vals):>8}{overall:>9.1f}%")
        print(f"point-biserial r(PHQ-9, CDS) = {r:+.3f} "
              f"({'rises' if r > 0 else 'does not rise'} with PHQ-9)")

    return {"per_band": per_band, "r_pointbiserial": r, "pct_cds": overall}


# ──────────────────────────────────────────────────────────────────────────────
#  One-call orchestrator (drop-in for the notebook)
# ──────────────────────────────────────────────────────────────────────────────

def visualize_run(network, path, filename, *, phq9_interval=None, csd_window=8,
                  ngrams_path=NGRAMS_PATH, smooth_window=None,
                  save=True, show=True, validate=True, overwrite=False,
                  snapshots=True):
    """Produce the per-run figures (and the CDS validation table) for one run.

    Drop-in replacement for the network-evolution block in ``experiment.ipynb``::

        nev.visualize_run(network, plot_path, plot_filename,
                          phq9_interval=args.check_point, save=args.save)

    Writes three figures into ``path``: the CDS-evolution graph
    (``cds_evolution_*``), the critical-slowing-down dot grid
    (``csd_heatmaps_w*``) and — when ``snapshots`` — the 10-panel PHQ-9 network
    sequence (``network_snapshot_phq9_*``), identical to ``experiment.ipynb``'s
    ``vis.print_subnetworks_phq9``.

    ``phq9_interval`` defaults to a best-effort guess from the data; pass
    ``args.check_point`` when you have it for an exact cadence. With
    ``overwrite=False`` (the default) any figure whose PNG already exists is
    skipped individually, so re-running only fills in what's missing.
    """
    if phq9_interval is None:
        phq9_interval = infer_phq9_interval(network)
        print(f"[network_evolution] inferred PHQ-9 update interval = {phq9_interval} rounds")

    patterns = load_cds_patterns(ngrams_path)

    if validate:
        cds_validation_summary(network, patterns=patterns)

    plot_cds_evolution(network, patterns=patterns, phq9_interval=phq9_interval,
                       smooth_window=smooth_window, path=path, filename=filename,
                       save=save, show=show, overwrite=overwrite)
    plot_csd_heatmaps(network, phq9_interval=phq9_interval, window=csd_window,
                      path=path, filename=filename, save=save, show=show,
                      overwrite=overwrite)

    # The 10-panel PHQ-9 network sequence, exactly as in experiment.ipynb. Import
    # lazily so callers that only want the evolution figures don't pull in
    # visualization.py (seaborn etc.).
    if snapshots and not _skip_existing(path, filename, "network_snapshot_phq9",
                                        overwrite, save):
        from .visualization import print_subnetworks_phq9
        if save and path is not None:
            os.makedirs(str(path), exist_ok=True)
        print_subnetworks_phq9(network, path=path, filename=filename,
                               save=save, show_fig=show)
