"""Sensitivity analysis: cosine within-setting vs cross-setting, stratified by PHQ-9.

For each axis (neighbour, agent):
    - within-setting cosine = baseline of irreducible LLM stochasticity
      (3 unseeded replicates of the same (agent_seed, neighbor_seed)).
    - cross-setting cosine  = effect of varying the axis, with LLM noise
      mixed in.
If cross < within, the axis moves outputs more than LLM noise alone — the axis
matters. If cross ≈ within, the axis is undetectable above baseline noise.

Two comparison units, picked per axis:

    NEIGHBOUR axis: same 60 agents across all 4 settings. So for every
        (agent_id, round) anchor we have 4 × 3 = 12 post embeddings. Compute
        paired per-anchor cosine, then aggregate.

    AGENT axis: different 60 agents per setting. So no paired comparison.
        Aggregate each agent's 10 posts into a centroid (mean), then compare
        within-setting (same agent across reps) vs cross-setting (different
        agents in same PHQ-9 band).

Stratify both into the 5-band PHQ-9 buckets used elsewhere in the project
(Minimal 0-4 / Mild 5-9 / Moderate 10-14 / Mod. Severe 15-19 / Severe 20-27).

Outputs to ``data/sensitivity/plots/``:
    {neighbor,agent}_cosines.csv        - one row per pair
    {neighbor,agent}_summary.csv        - mean ± std per (band, within|cross)
    {neighbor,agent}_bar_scatter.png    - bars + jittered scatter per band
    {neighbor,agent}_heatmap.png        - 4×4 cosine matrix

Usage::

    PYTHONPATH=src python -m utils.sensitivity.sa_analyze
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# 5-band PHQ-9 split (matches visualization._phq9_severity_color).
PHQ9_BANDS = [
    (0,  4,  "Minimal",     "#2ecc71"),
    (5,  9,  "Mild",        "#f1c40f"),
    (10, 14, "Moderate",    "#e67e22"),
    (15, 19, "Mod. Severe", "#e74c3c"),
    (20, 27, "Severe",      "#8b0000"),
]
BAND_LABELS = [b[2] for b in PHQ9_BANDS]


def phq9_to_band(score: int) -> str:
    for lo, hi, label, _ in PHQ9_BANDS:
        if lo <= score <= hi:
            return label
    return "?"


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine sim. Each row is a vector; returns 1D array of cosines."""
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return (a_n * b_n).sum(axis=-1)


def load_axis_runs(root: str, axis: str) -> dict:
    """Return {(setting_seed, rep): {embeddings, agent_ids, rounds, phq9}}."""
    paths = sorted(glob.glob(os.path.join(root, axis, "setting_*", "rep_*", "embeddings.npz")))
    runs = {}
    for p in paths:
        parts = p.split(os.sep)
        setting = int(next(x for x in parts if x.startswith("setting_")).split("_")[1])
        rep = int(next(x for x in parts if x.startswith("rep_")).split("_")[1])
        data = np.load(p, allow_pickle=True)
        runs[(setting, rep)] = {
            "embeddings": data["embeddings"],
            "agent_ids":  data["agent_ids"],
            "rounds":     data["rounds"],
            "phq9":       data["phq9"],
        }
    return runs


# =====================================================================
# NEIGHBOUR AXIS — paired per-(agent, round) cosine
# =====================================================================

def neighbor_cosines(runs: dict) -> pd.DataFrame:
    """For each (agent_id, round) anchor, compute cosine between every pair of
    embeddings across the 4×3 = 12 (setting, rep) cells.
    """
    settings = sorted({s for (s, _) in runs.keys()})
    reps = sorted({r for (_, r) in runs.keys()})

    # Index each run by (agent_id, round) for paired lookup.
    indexed = {}
    for key, data in runs.items():
        idx = {(int(a), int(r)): i for i, (a, r) in
               enumerate(zip(data["agent_ids"], data["rounds"]))}
        indexed[key] = (data, idx)

    # Anchors present in EVERY run (should be all of them given fixed agents).
    common_keys = None
    for (data, idx) in indexed.values():
        keys = set(idx.keys())
        common_keys = keys if common_keys is None else common_keys & keys
    common_keys = sorted(common_keys or [])
    print(f"[neighbour] {len(common_keys)} common (agent, round) anchors "
          f"across {len(runs)} runs")

    rows = []
    for (aid, rd) in common_keys:
        # Pull the PHQ-9 from any run (it's persona-fixed).
        sample_key = next(iter(indexed.keys()))
        data, idx = indexed[sample_key]
        phq9 = int(data["phq9"][idx[(aid, rd)]])
        band = phq9_to_band(phq9)

        # All embeddings for this anchor.
        embs = {}
        for (s, r), (data, idx) in indexed.items():
            embs[(s, r)] = data["embeddings"][idx[(aid, rd)]]

        # WITHIN-SETTING pairs: same setting, different reps.
        for s in settings:
            for ra, rb in combinations(reps, 2):
                if (s, ra) in embs and (s, rb) in embs:
                    cs = float(cosine_rows(embs[(s, ra)][None, :],
                                           embs[(s, rb)][None, :])[0])
                    rows.append({"agent_id": aid, "round": rd, "phq9": phq9,
                                 "band": band, "pair_type": "within",
                                 "setting_a": s, "setting_b": s,
                                 "rep_a": ra, "rep_b": rb, "cosine": cs})

        # CROSS-SETTING pairs: different settings, all rep combinations.
        for sa, sb in combinations(settings, 2):
            for ra, rb in product(reps, reps):
                if (sa, ra) in embs and (sb, rb) in embs:
                    cs = float(cosine_rows(embs[(sa, ra)][None, :],
                                           embs[(sb, rb)][None, :])[0])
                    rows.append({"agent_id": aid, "round": rd, "phq9": phq9,
                                 "band": band, "pair_type": "cross",
                                 "setting_a": sa, "setting_b": sb,
                                 "rep_a": ra, "rep_b": rb, "cosine": cs})

    return pd.DataFrame(rows)


# =====================================================================
# AGENT AXIS — paired per-(slot, round) cosine
#
# Requires --stratify-phq9 at generation time so that slot i has the same
# PHQ-9 in every setting. The persona at slot i still differs across
# settings — that's precisely what the cross-setting cosine isolates,
# with neighbour-input and PHQ-9 held constant.
#
# Kept below as `agent_cosines_centroid` is the older per-agent-centroid +
# band-matched comparison, retained for non-stratified data or as a
# robustness check.
# =====================================================================

def agent_cosines(runs: dict) -> pd.DataFrame:
    """Paired per-(agent_id, round) cosine for the agent axis. Requires the
    runs to have been generated with --stratify-phq9 so that slot agent_id=i
    carries the same PHQ-9 in every setting (only the persona differs).

    Fails loudly if the PHQ-9 vector differs across settings at slot 0 — that
    means stratification wasn't applied at generation time, in which case
    pair-by-slot is meaningless and you should use `agent_cosines_centroid`.
    """
    # Sanity check: same (agent_id → phq9) mapping across all (setting, rep)?
    ref = None
    for key, data in runs.items():
        mapping = {int(a): int(p) for a, p in zip(data["agent_ids"], data["phq9"])}
        if ref is None:
            ref = mapping
            ref_key = key
            continue
        if mapping != ref:
            diffs = [k for k in mapping if mapping[k] != ref.get(k)]
            raise SystemExit(
                f"[agent] slot-level PHQ-9 vector differs between {ref_key} and {key} "
                f"(e.g. {diffs[:5]}). Re-generate the agent axis with --stratify-phq9, "
                "or call agent_cosines_centroid() instead."
            )
    return neighbor_cosines(runs)


# =====================================================================
# Legacy: centroid-based agent comparison (kept for non-stratified data)
# =====================================================================

def agent_cosines_centroid(runs: dict) -> pd.DataFrame:
    """For each agent compute a centroid (mean of its 10 posts) per (setting, rep).
    Within-setting: same agent across reps. Cross-setting: different agents in
    the same PHQ-9 band, all rep combinations.
    """
    settings = sorted({s for (s, _) in runs.keys()})
    reps = sorted({r for (_, r) in runs.keys()})

    # centroids[(s, r)] = {agent_id: (centroid_vec, phq9, band)}
    centroids = {}
    for (s, r), data in runs.items():
        per_agent = {}
        embs = data["embeddings"]
        aids = data["agent_ids"]
        phq9s = data["phq9"]
        for aid in np.unique(aids):
            mask = aids == aid
            cent = embs[mask].mean(axis=0).astype(np.float32)
            phq9 = int(phq9s[mask][0])
            per_agent[int(aid)] = (cent, phq9, phq9_to_band(phq9))
        centroids[(s, r)] = per_agent
    print(f"[agent] centroids computed across {len(runs)} runs; "
          f"agents/run = "
          f"{[len(v) for v in centroids.values()]}")

    rows = []

    # WITHIN-SETTING: same agent across reps within one setting.
    for s in settings:
        agents_in_s = set()
        for r in reps:
            if (s, r) in centroids:
                agents_in_s |= set(centroids[(s, r)].keys())
        for aid in agents_in_s:
            for ra, rb in combinations(reps, 2):
                if (s, ra) not in centroids or (s, rb) not in centroids:
                    continue
                if aid not in centroids[(s, ra)] or aid not in centroids[(s, rb)]:
                    continue
                ca, phq9, band = centroids[(s, ra)][aid]
                cb, *_ = centroids[(s, rb)][aid]
                cs = float(cosine_rows(ca[None, :], cb[None, :])[0])
                rows.append({"pair_type": "within",
                             "setting_a": s, "setting_b": s,
                             "rep_a": ra, "rep_b": rb,
                             "band": band, "agent_a": aid, "agent_b": aid,
                             "cosine": cs})

    # CROSS-SETTING: matched by PHQ-9 band across different settings.
    for sa, sb in combinations(settings, 2):
        for ra, rb in product(reps, reps):
            if (sa, ra) not in centroids or (sb, rb) not in centroids:
                continue
            # Group by band.
            band_a, band_b = {}, {}
            for aid, (cent, _, band) in centroids[(sa, ra)].items():
                band_a.setdefault(band, []).append((aid, cent))
            for aid, (cent, _, band) in centroids[(sb, rb)].items():
                band_b.setdefault(band, []).append((aid, cent))
            for band in band_a.keys() & band_b.keys():
                for (aid_a, ca), (aid_b, cb) in product(band_a[band], band_b[band]):
                    cs = float(cosine_rows(ca[None, :], cb[None, :])[0])
                    rows.append({"pair_type": "cross",
                                 "setting_a": sa, "setting_b": sb,
                                 "rep_a": ra, "rep_b": rb,
                                 "band": band, "agent_a": aid_a, "agent_b": aid_b,
                                 "cosine": cs})
    return pd.DataFrame(rows)


# =====================================================================
# PLOTTING
# =====================================================================

def plot_bar_scatter(df: pd.DataFrame, axis_name: str, out_path: str,
                     max_scatter_per_band: int = 200):
    """Bar chart with jittered per-pair scatter, two bars per PHQ-9 band."""
    fig, ax = plt.subplots(figsize=(11, 6))
    n_bands = len(BAND_LABELS)
    x = np.arange(n_bands)
    bar_w = 0.36
    rng = np.random.default_rng(0)
    colour_within = "#cccccc"
    colour_cross  = "#3498db"

    for ptype, colour, offset in [("within", colour_within, -bar_w / 2),
                                  ("cross",  colour_cross,  +bar_w / 2)]:
        means, errs = [], []
        for band in BAND_LABELS:
            sub = df[(df.band == band) & (df.pair_type == ptype)]
            means.append(sub.cosine.mean() if len(sub) else np.nan)
            errs.append(sub.cosine.std() if len(sub) > 1 else 0.0)
        ax.bar(x + offset, means, bar_w, yerr=errs,
               label=f"{ptype}-setting",
               color=colour, edgecolor="black", linewidth=0.5,
               error_kw={"elinewidth": 0.7, "capsize": 3})

        # Jittered scatter (subsample for readability).
        for j, band in enumerate(BAND_LABELS):
            sub = df[(df.band == band) & (df.pair_type == ptype)]
            if len(sub) == 0:
                continue
            if len(sub) > max_scatter_per_band:
                sub = sub.sample(max_scatter_per_band, random_state=int(j))
            jit = rng.uniform(-bar_w / 4, bar_w / 4, len(sub))
            ax.scatter(np.full(len(sub), x[j] + offset) + jit, sub.cosine,
                       s=6, alpha=0.25, color="black", linewidths=0)

    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS, rotation=10)
    ax.set_ylabel("Cosine similarity")
    ax.set_xlabel("PHQ-9 band")
    ax.set_title(f"{axis_name} sensitivity — within vs cross-setting cosine")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_heatmap(df: pd.DataFrame, axis_name: str, out_path: str):
    """4×4 cosine matrix: diagonal = within-setting, off-diagonal = cross."""
    settings = sorted(set(df.setting_a) | set(df.setting_b))
    n = len(settings)
    mat = np.full((n, n), np.nan)
    for i, sa in enumerate(settings):
        for j, sb in enumerate(settings):
            if sa == sb:
                sub = df[(df.pair_type == "within") & (df.setting_a == sa)]
            else:
                sub = df[(df.pair_type == "cross") &
                         (((df.setting_a == sa) & (df.setting_b == sb)) |
                          ((df.setting_a == sb) & (df.setting_b == sa)))]
            if len(sub):
                mat[i, j] = sub.cosine.mean()

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))
    pad = max(0.005, (vmax - vmin) * 0.05)
    im = ax.imshow(mat, cmap="Reds", vmin=vmin - pad, vmax=vmax + pad)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"seed {s}" for s in settings], rotation=30, ha="right")
    ax.set_yticklabels([f"seed {s}" for s in settings])
    threshold = vmin + (vmax - vmin) * 0.6
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                colour = "white" if mat[i, j] > threshold else "black"
                ax.text(j, i, f"{mat[i, j]:.3f}",
                        ha="center", va="center", color=colour, fontsize=10)
    ax.set_title(f"{axis_name} sensitivity — pairwise cosine across settings")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def _axis_heatmap_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Return (matrix, settings) for one axis: diagonal = within-setting mean,
    off-diagonal = cross-setting mean cosine."""
    settings = sorted(set(df.setting_a) | set(df.setting_b))
    n = len(settings)
    mat = np.full((n, n), np.nan)
    for i, sa in enumerate(settings):
        for j, sb in enumerate(settings):
            if sa == sb:
                sub = df[(df.pair_type == "within") & (df.setting_a == sa)]
            else:
                sub = df[(df.pair_type == "cross") &
                         (((df.setting_a == sa) & (df.setting_b == sb)) |
                          ((df.setting_a == sb) & (df.setting_b == sa)))]
            if len(sub):
                mat[i, j] = float(sub.cosine.mean())
    return mat, settings


def plot_combined_heatmaps(axis_dfs: dict, out_path: str,
                           cmap: str = "Blues", cell_size: float = 0.85):
    """All SA-axis heatmaps in one figure, sharing one colourbar and one colour
    scale. Style matches the prompt-SA heatmap in experiment.ipynb (Blues cmap,
    cell-size sizing, white linewidths, title beneath each panel).
    Y-axis seed labels render only on the first (leftmost) panel.
    """
    names = list(axis_dfs.keys())
    n_panels = len(names)
    if n_panels == 0:
        return

    matrices, settings_per = [], []
    for name in names:
        m, s = _axis_heatmap_matrix(axis_dfs[name])
        matrices.append(m); settings_per.append(s)

    # Shared colour scale across all panels.
    all_vals = np.concatenate([m[~np.isnan(m)] for m in matrices])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    pad = max(0.005, (vmax - vmin) * 0.05)
    vmin, vmax = vmin - pad, vmax + pad

    n_cells = max(len(s) for s in settings_per)
    panel_w = max(4.0, n_cells * cell_size)
    panel_h = max(4.0, n_cells * cell_size)
    fig_w = panel_w * n_panels + 1.5
    fig_h = panel_h + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, n_panels + 1,
                          width_ratios=[panel_w] * n_panels + [0.35],
                          wspace=0.15)
    cbar_ax = fig.add_subplot(gs[0, -1])

    for k, (name, mat, sets) in enumerate(zip(names, matrices, settings_per)):
        ax = fig.add_subplot(gs[0, k])
        show_cbar = (k == n_panels - 1)
        show_ytick = (k == 0)
        xlabels = [f"seed {s}" for s in sets]
        ylabels = [f"seed {s}" for s in sets] if show_ytick else False
        sns.heatmap(
            mat, ax=ax,
            xticklabels=xlabels, yticklabels=ylabels,
            vmin=vmin, vmax=vmax,
            annot=True, fmt=".3f", annot_kws={"fontsize": 9},
            cmap=cmap, linewidths=0.4, linecolor="white",
            cbar=show_cbar,
            cbar_ax=cbar_ax if show_cbar else None,
            cbar_kws={"label": "cosine similarity"} if show_cbar else None,
            square=True,
        )
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        # Title beneath the panel.
        ax.text(0.5, -0.18, name, transform=ax.transAxes,
                ha="center", va="top", fontsize=11)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# =====================================================================
# MAIN
# =====================================================================

# =====================================================================
# PHQ-9 conditioning: 5 band-settings, same agent + neighbour, vary PHQ-9
# =====================================================================

def phq9_conditioning_heatmap(root: str, out_dir: str) -> pd.DataFrame | None:
    """Build the band×band cosine matrix from the PHQ-9 conditioning runs.

    For each (agent_id, round) anchor that exists in all 5 band-settings,
    cosine every pair of the 5 embeddings. Bin each cosine by the
    (source_band, target_band) the two embeddings were generated under, and
    average. The result is a 5×5 matrix whose off-diagonal entries answer
    "how much do outputs change when the same persona/neighbour input is
    re-conditioned on a different PHQ-9 band?".
    """
    paths = sorted(glob.glob(os.path.join(root, "phq9", "*", "embeddings.npz")))
    if not paths:
        print(f"[phq9-cond] no embeddings under {root}/phq9/")
        return None

    # label_dir → (data, idx_by_anchor, dominant_band)
    settings: dict[str, tuple] = {}
    for p in paths:
        label = p.split(os.sep)[-2]                    # e.g. "minimal"
        data = np.load(p, allow_pickle=True)
        idx = {(int(a), int(r)): i for i, (a, r) in
               enumerate(zip(data["agent_ids"], data["rounds"]))}
        # Verify all rows in this setting fall in one band (defensive).
        bands_here = {phq9_to_band(int(p)) for p in data["phq9"]}
        if len(bands_here) != 1:
            print(f"[phq9-cond] WARNING: {label} spans bands {bands_here}; "
                  f"taking the most-common one for axis label.")
        dom_band = pd.Series([phq9_to_band(int(p)) for p in data["phq9"]]).mode().iloc[0]
        settings[label] = (data, idx, dom_band)

    # Order settings by their dominant band's position in BAND_LABELS.
    band_order = {b: i for i, b in enumerate(BAND_LABELS)}
    ordered = sorted(settings.items(), key=lambda kv: band_order.get(kv[1][2], 99))
    band_labels_ax = [b for _, (_, _, b) in ordered]
    keys = [k for k, _ in ordered]
    print(f"[phq9-cond] {len(ordered)} band settings: "
          f"{', '.join(f'{k}->{b}' for k, (_, _, b) in ordered)}")

    # Common anchors across all settings.
    common = None
    for _, (_, idx, _) in ordered:
        anchors = set(idx.keys())
        common = anchors if common is None else common & anchors
    common = sorted(common or [])
    print(f"[phq9-cond] {len(common)} common (agent, round) anchors")

    # Pairwise cosines binned by (source_band, target_band).
    cell_values: dict[tuple, list[float]] = defaultdict(list)
    for (aid, rd) in common:
        embs = {k: settings[k][0]["embeddings"][settings[k][1][(aid, rd)]] for k in keys}
        for ka, kb in product(keys, keys):
            if ka == kb:
                continue
            band_a = settings[ka][2]
            band_b = settings[kb][2]
            cs = float(cosine_rows(embs[ka][None, :], embs[kb][None, :])[0])
            cell_values[(band_a, band_b)].append(cs)

    # Build the N×N matrix in band_labels_ax order.
    n = len(band_labels_ax)
    mat = np.full((n, n), np.nan)
    for i, ba in enumerate(band_labels_ax):
        for j, bb in enumerate(band_labels_ax):
            if i == j:
                mat[i, j] = 1.0   # cosine with self
            else:
                vals = cell_values.get((ba, bb), [])
                mat[i, j] = float(np.mean(vals)) if vals else np.nan

    # Plot.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    off_diag = mat[~np.eye(n, dtype=bool)]
    vmin = float(np.nanmin(off_diag))
    im = ax.imshow(mat, cmap="Blues", vmin=vmin - 0.01, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(band_labels_ax, rotation=30, ha="right")
    ax.set_yticklabels(band_labels_ax)
    threshold = (vmin + 1.0) / 2
    for i in range(n):
        for j in range(n):
            if np.isnan(mat[i, j]):
                continue
            colour = "white" if mat[i, j] > threshold else "black"
            ax.text(j, i, f"{mat[i, j]:.3f}",
                    ha="center", va="center", color=colour, fontsize=10)
    ax.set_xlabel("PHQ-9 band (target)")
    ax.set_ylabel("PHQ-9 band (source)")
    ax.set_title("PHQ-9 conditioning — cosine between same-persona "
                 "posts under different PHQ-9 bands")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "phq9_conditioning_heatmap.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")

    df = pd.DataFrame(mat, index=band_labels_ax, columns=band_labels_ax)
    df.to_csv(os.path.join(out_dir, "phq9_conditioning_matrix.csv"))
    return df


# =====================================================================
# Three-axis comparison: per-anchor cosine drop (within − cross) per axis
# =====================================================================

def _per_anchor_drops(df: pd.DataFrame) -> pd.Series:
    """For each (agent_id, round) anchor, drop = mean within − mean cross.
    One value per anchor — the distribution this returns is what the box plot
    visualises."""
    grouped = df.groupby(["agent_id", "round"])
    drops = []
    for _, sub in grouped:
        within = sub.loc[sub.pair_type == "within", "cosine"]
        cross  = sub.loc[sub.pair_type == "cross",  "cosine"]
        if len(within) == 0 or len(cross) == 0:
            continue
        drops.append(float(within.mean() - cross.mean()))
    return pd.Series(drops)


def comparison_boxplot(axis_dfs: dict, out_path: str):
    """Box plot of per-anchor cosine drops, one box per axis.

    Median = typical effect, IQR = consistency, outliers = anchors where the
    factor really mattered. More informative than the bar+CI variant because
    it shows the full distribution; CI bands at this sample size are
    tiny and hide spread.
    """
    names = list(axis_dfs.keys())
    drops = [_per_anchor_drops(axis_dfs[n]).values for n in names]

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    colours = ["#3498db", "#e74c3c", "#9b59b6"][:len(names)]
    bp = ax.boxplot(drops, tick_labels=names, patch_artist=True,
                    widths=0.55, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black",
                                   markeredgecolor="black", markersize=5),
                    medianprops=dict(color="black", linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                    markeredgecolor="none", markerfacecolor="grey"))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")

    ax.axhline(0, color="black", linewidth=0.6, linestyle="-", alpha=0.5)
    ax.set_ylabel("Per-anchor cosine drop  (within − cross)")
    ax.set_title("Sensitivity comparison across factors\n"
                 "(higher = factor moves output more than LLM noise alone)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Annotate median values.
    medians = [float(np.median(d)) for d in drops]
    for i, m in enumerate(medians):
        ax.text(i + 1, m, f"  med={m:.3f}",
                va="center", ha="left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def _per_anchor_null_drops(df: pd.DataFrame, rng: np.random.Generator,
                           n_per_anchor: int = 20) -> np.ndarray:
    """For each anchor, compute the magnitude of a 'null drop' = |mean(half1) − mean(half2)|
    on random half-splits of that anchor's within-setting cosines. This is the cosine
    drop you'd see if the axis under test had NO effect — just LLM stochasticity.
    """
    nulls = []
    for _, sub in df[df.pair_type == "within"].groupby(["agent_id", "round"]):
        vals = sub.cosine.values
        if len(vals) < 4:
            continue
        half = len(vals) // 2
        diffs = []
        for _ in range(n_per_anchor):
            perm = rng.permutation(vals)
            diffs.append(abs(perm[:half].mean() - perm[half:2 * half].mean()))
        nulls.append(float(np.mean(diffs)))
    return np.asarray(nulls)


def comparison_combined(axis_dfs: dict, out_path: str,
                        n_bootstrap: int = 1000, seed: int = 0):
    """Side-by-side: box plot (left, full per-anchor distribution) + forest
    plot (right, mean within−cross with 95 % bootstrap CI). Adds an LLM-noise
    NULL reference: a 4th box of per-anchor null drops, and a horizontal line
    at the null median on both panels. Anything above the line exceeds
    irreducible LLM stochasticity."""
    rng = np.random.default_rng(seed)
    names = list(axis_dfs.keys())
    # Colour scheme tied to the agent_phq9_combined plot:
    #   Neighbour → dark brown (top of Oranges, matches heatmap diagonals)
    #   Agent     → blue        (within-setting bar)
    #   Joint     → bright orange (cross-setting bar)
    _COLOUR_BY_NAME = {
        "Neighbour": "#8d2c03",
        "Agent":     "#2e7ebc",
        "Joint":     "#d96907",
    }
    colours = [_COLOUR_BY_NAME.get(n, "#7f7f7f") for n in names]

    drops = [_per_anchor_drops(axis_dfs[n]).values for n in names]

    # Null = LLM-noise drops, computed from each axis's within-set then pooled.
    null_per_axis = [_per_anchor_null_drops(axis_dfs[n], rng) for n in names]
    null_pooled = np.concatenate(null_per_axis) if null_per_axis else np.asarray([])
    null_median = float(np.median(null_pooled)) if len(null_pooled) else np.nan

    # CLUSTER bootstrap on anchors (NOT individual pair rows). The 12-66 pair rows
    # per anchor are tightly correlated (same persona, same round), so resampling
    # rows treats correlated observations as independent and shrinks CIs
    # artificially. Resampling anchors gives CIs that reflect how much the mean
    # would shift if a different 60 agents had been drawn — the actual
    # generalisation question.
    means, los, his = [], [], []
    for name, anchor_drops in zip(names, drops):
        observed = float(anchor_drops.mean())
        boots = [
            float(rng.choice(anchor_drops, size=len(anchor_drops), replace=True).mean())
            for _ in range(n_bootstrap)
        ]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        means.append(observed); los.append(float(lo)); his.append(float(hi))

    fig, (ax_box, ax_forest) = plt.subplots(
        1, 2, figsize=(7.0, 3.3),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )

    # --- Box plot (left) ---
    # 3 factor boxes + 1 LLM-noise null box.
    box_data = drops + [null_pooled]
    box_labels = names + ["LLM noise"]
    box_colours = colours + ["#bdc3c7"]
    bp = ax_box.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                        widths=0.55, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="black",
                                       markeredgecolor="black", markersize=5),
                        medianprops=dict(color="black", linewidth=1.2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                        markeredgecolor="none",
                                        markerfacecolor="grey"))
    for patch, c in zip(bp["boxes"], box_colours):
        patch.set_facecolor(c); patch.set_alpha(0.65); patch.set_edgecolor("black")
    ax_box.axhline(0, color="black", linewidth=0.6, linestyle="-", alpha=0.5)
    if not np.isnan(null_median):
        ax_box.axhline(null_median, color="#555555", linewidth=1.0,
                       linestyle="--", alpha=0.8,
                       label="LLM-noise")
        ax_box.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_box.set_ylabel("Cosine drop")
    ax_box.grid(axis="y", linestyle=":", alpha=0.5)

    # --- Forest plot (right) ---
    y = np.arange(len(names))[::-1]
    err_low  = [m - lo for m, lo in zip(means, los)]
    err_high = [hi - m for m, hi in zip(means, his)]
    ax_forest.errorbar(means, y, xerr=[err_low, err_high],
                       fmt="none", color="black",
                       capsize=5, capthick=1.2, linewidth=1.2)
    for i, (m, c) in enumerate(zip(means, colours)):
        ax_forest.plot(m, y[i], "o", color=c, markersize=7,
                       markeredgecolor="black", markeredgewidth=0.8, zorder=3)
    ax_forest.set_yticks(y); ax_forest.set_yticklabels(names)
    ax_forest.set_xlabel("Cosine drop")
    ax_forest.axvline(0, color="black", linewidth=0.6, linestyle="-", alpha=0.5)
    if not np.isnan(null_median):
        ax_forest.axvline(null_median, color="#555555", linewidth=1.0,
                          linestyle="--", alpha=0.8)
    ax_forest.grid(axis="x", linestyle=":", alpha=0.5)
    for i, m in enumerate(means):
        ax_forest.text(m, y[i] + 0.18, f"{m:.3f}",
                       ha="center", va="bottom", fontsize=10)
    ax_forest.set_ylim(y.min() - 0.55, y.max() + 0.55)

    # Panel labels below each subplot. Same axis-coord y on both → same fig y
    # (both axes share the same row, so same y0 and height).
    _panel_y = -0.30
    ax_box.text(0.5, _panel_y, "(a) Distribution per anchor",
                transform=ax_box.transAxes, ha="center", va="top", fontsize=11)
    ax_forest.text(0.5, _panel_y, "(b) Mean ± 95 % CI",
                   transform=ax_forest.transAxes, ha="center", va="top", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def comparison_barplot(axis_dfs: dict, out_path: str,
                       n_bootstrap: int = 1000, seed: int = 0):
    """Bar plot of mean within−cross cosine drop per axis, bootstrap CIs.
    Kept as a single-number summary companion to the box plot."""
    rng = np.random.default_rng(seed)
    names = list(axis_dfs.keys())
    means, lo_err, hi_err = [], [], []
    for name in names:
        df = axis_dfs[name]
        within = df.loc[df.pair_type == "within", "cosine"].values
        cross = df.loc[df.pair_type == "cross", "cosine"].values
        if not len(within) or not len(cross):
            means.append(np.nan); lo_err.append(0); hi_err.append(0); continue
        observed = float(within.mean() - cross.mean())
        boots = []
        for _ in range(n_bootstrap):
            w = rng.choice(within, size=len(within), replace=True)
            c = rng.choice(cross,  size=len(cross),  replace=True)
            boots.append(w.mean() - c.mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        means.append(observed)
        lo_err.append(observed - lo)
        hi_err.append(hi - observed)

    colours = ["#3498db", "#e74c3c", "#9b59b6"][:len(names)]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=[lo_err, hi_err], color=colours,
           edgecolor="black", linewidth=0.5, capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean cosine drop (within − cross)")
    ax.set_title("Sensitivity comparison across factors\n(higher bar = larger output shift)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ymax = max([m + h for m, h in zip(means, hi_err) if not np.isnan(m)] or [0.05])
    ax.set_ylim(0, ymax * 1.18)
    for i, (m, lo, hi) in enumerate(zip(means, lo_err, hi_err)):
        if np.isnan(m):
            continue
        ax.text(x[i], m + hi + ymax * 0.02, f"{m:.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def phq9_distance_lineplot(matrix_df: pd.DataFrame, out_path: str):
    """Line plot of mean cosine vs PHQ-9 band-distance.

    Collapses the 5×5 PHQ-9 conditioning matrix into one curve. A smooth
    monotonic decline = continuous PHQ-9 encoding (the model treats PHQ-9 as
    a slider). A step pattern (e.g. all band-distances ≥1 land at the same
    cosine, no further decline) = banded encoding (the model treats PHQ-9 as
    a categorical severity classifier).
    """
    bands = list(matrix_df.index)
    n = len(bands)
    by_dist: dict[int, list[float]] = defaultdict(list)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue   # diagonal trivially 1.0; uninformative
            v = float(matrix_df.iloc[i, j])
            if not np.isnan(v):
                by_dist[abs(i - j)].append(v)

    xs = sorted(by_dist.keys())
    means = [float(np.mean(by_dist[d])) for d in xs]
    stds  = [float(np.std(by_dist[d]))  for d in xs]
    counts = [len(by_dist[d]) for d in xs]

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.errorbar(xs, means, yerr=stds, fmt="o-", color="#e74c3c",
                capsize=4, linewidth=1.8, markersize=7)
    for x, m, c in zip(xs, means, counts):
        ax.text(x, m + 0.005, f"  n={c}", fontsize=8, va="bottom", color="grey")
    ax.set_xticks(xs)
    ax.set_xlabel("PHQ-9 band-distance  (|source band index − target band index|)")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("PHQ-9 conditioning — output similarity vs PHQ-9 band-distance")
    ax.grid(linestyle=":", alpha=0.5)
    # A horizontal line at the within-band ceiling for visual reference
    # (diagonal of the matrix = 1.0 trivially; the off-diagonal max is more useful).
    ax.axhline(max(means), color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


# =====================================================================
# Combined: Agent-axis bars + PHQ-9-conditioning heatmap
# =====================================================================

def plot_agent_phq9_combined(agent_df: pd.DataFrame, phq9_matrix: pd.DataFrame,
                             out_path: str,
                             color_within: str = "#2e7ebc",
                             color_cross: str = "#d96907",
                             cmap: str = "Oranges",
                             max_scatter_per_band: int = 200,
                             y_floor: float = 0.80):
    """Side-by-side: agent-axis within/cross bars per PHQ-9 band (left) and
    PHQ-9 conditioning 5×5 cosine heatmap (right). Blue theme throughout;
    cross bars are project-orange (#d96907) by default."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=(8, 2.8))
    # 2 cols only — colorbar attaches directly to the heatmap below.
    # wspace now controls ONLY the bar↔heatmap gap.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.35)
    ax_bars = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])

    # ===== Left panel: per-band bars (within vs cross) for the agent axis =====
    rng = np.random.default_rng(0)
    bands = BAND_LABELS
    x = np.arange(len(bands))
    bar_w = 0.36
    for ptype, colour, offset in [("within", color_within, -bar_w / 2),
                                  ("cross",  color_cross,  +bar_w / 2)]:
        means, errs = [], []
        for band in bands:
            sub = agent_df[(agent_df.band == band) & (agent_df.pair_type == ptype)]
            means.append(sub.cosine.mean() if len(sub) else np.nan)
            errs.append(sub.cosine.std()  if len(sub) > 1 else 0.0)
        ax_bars.bar(x + offset, means, bar_w, yerr=errs,
                    label=f"{ptype}-setting",
                    color=colour, edgecolor="black", linewidth=0.5,
                    error_kw={"elinewidth": 0.7, "capsize": 3})
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(bands, rotation=30, fontsize=9)
    ax_bars.tick_params(axis="y", labelsize=9)
    ax_bars.set_ylabel("Cosine similarity")
    # ax_bars.set_xlabel("PHQ-9 band")
    ax_bars.set_ylim(y_floor, 0.95)
    ax_bars.grid(axis="y", linestyle=":", alpha=0.5)
    ax_bars.legend(loc="lower right")
    # Panel (a) label placed via fig.text below, aligned with (b).

    # Broken-axis "wiggle" — diagonal slashes at the bottom of the y-axis
    # indicating the y-axis is truncated at y_floor.
    kw = dict(transform=ax_bars.transAxes, color="black",
              clip_on=False, linewidth=1.0)
    d_x, d_y = 0.012, 0.020
    ax_bars.plot((-d_x, +d_x), (-d_y, +d_y), **kw)
    ax_bars.plot((-d_x, +d_x), (-d_y + 0.012, +d_y + 0.012), **kw)

    # ===== Right panel: PHQ-9 conditioning heatmap =====
    mat = phq9_matrix.values
    n = mat.shape[0]
    band_labels_ax = list(phq9_matrix.index)
    off_diag = mat[~np.eye(n, dtype=bool)]
    vmin = float(np.nanmin(off_diag))
    sns.heatmap(
        mat, ax=ax_heat,
        xticklabels=band_labels_ax, yticklabels=band_labels_ax,
        vmin=vmin - 0.01, vmax=1.0,
        annot=True, fmt=".3f", annot_kws={"fontsize": 9},
        cmap=cmap, linewidths=0.4, linecolor="white",
        cbar=False,
        square=True,
    )
    ax_heat.tick_params(axis="x", rotation=30, labelsize=9)
    ax_heat.tick_params(axis="y", rotation=0, labelsize=9)
    # Dock the colorbar directly to the heatmap. pad controls the gap.
    divider = make_axes_locatable(ax_heat)
    cbar_ax = divider.append_axes("right", size="4%", pad=0.08)
    sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin - 0.01, vmax=1.0),
        cmap=plt.get_cmap(cmap),
    )
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("cosine similarity")
    cbar_ax.tick_params(labelsize=9)
    # ax_heat.set_xlabel("PHQ-9 band (target)")
    # ax_heat.set_ylabel("PHQ-9 band (source)")
    # Panel labels in axis coords. Same y on both works now that both panels
    # share rotation=30 + labelsize=9 (so tick-label heights match).
    ax_bars.text(0.5, -0.30, "(a) Agent axis",
                 transform=ax_bars.transAxes, ha="center", va="top", fontsize=11)
    ax_heat.text(0.5, -0.30, "(b) PHQ-9 conditioning",
                 transform=ax_heat.transAxes, ha="center", va="top", fontsize=11)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/sensitivity",
                        help="Directory containing axis subdirs.")
    parser.add_argument("--out-dir", default="data/sensitivity/plots",
                        help="Where to write CSV + PNG outputs.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    axis_dfs: dict[str, pd.DataFrame] = {}
    for axis_dir, axis_label, cosine_fn in [
        ("neighbor", "Neighbour", neighbor_cosines),
        ("agent",    "Agent",     agent_cosines),
        ("joint",    "Joint",     neighbor_cosines),  # same paired logic; same anchors
    ]:
        full = os.path.join(args.root, axis_dir)
        if not os.path.isdir(full):
            print(f"[skip] {full} not present")
            continue
        print(f"\n=== {axis_label} axis ===")

        runs = load_axis_runs(args.root, axis_dir)
        print(f"  loaded {len(runs)} runs from {full}/")
        if not runs:
            continue

        df = cosine_fn(runs)
        df.to_csv(os.path.join(args.out_dir, f"{axis_dir}_cosines.csv"), index=False)

        summary = (df.groupby(["band", "pair_type"]).cosine
                   .agg(["mean", "std", "count"]).round(4))
        print("  per (band, pair_type):")
        print(summary)
        summary.to_csv(os.path.join(args.out_dir, f"{axis_dir}_summary.csv"))

        # Δ (cross − within) per band — useful one-glance signal.
        pivot = (df.groupby(["band", "pair_type"]).cosine.mean()
                 .unstack("pair_type"))
        if {"within", "cross"} <= set(pivot.columns):
            pivot["delta"] = pivot["cross"] - pivot["within"]
            print("  cross − within per band:")
            print(pivot[["within", "cross", "delta"]].round(4))

        plot_bar_scatter(df, axis_label,
                         os.path.join(args.out_dir, f"{axis_dir}_bar_scatter.png"))
        axis_dfs[axis_label] = df

    # All axis heatmaps combined into one figure, shared colourbar + scale.
    if len(axis_dfs) >= 1:
        plot_combined_heatmaps(axis_dfs,
                               os.path.join(args.out_dir, "axes_heatmaps.png"))

    # Cross-axis comparison: combined box (distribution) + forest (mean ± CI).
    if len(axis_dfs) >= 2:
        print("\n=== Cross-axis comparison ===")
        comparison_combined(axis_dfs,
                            os.path.join(args.out_dir, "axes_comparison.png"))

    # PHQ-9 conditioning: 5×5 cosine heatmap + line plot of cosine vs band-distance.
    print("\n=== PHQ-9 conditioning ===")
    phq9_matrix = phq9_conditioning_heatmap(args.root, args.out_dir)
    if phq9_matrix is not None:
        phq9_distance_lineplot(phq9_matrix,
                               os.path.join(args.out_dir, "phq9_distance_line.png"))
        # Combined: agent-axis bars (blue/orange) + PHQ-9 heatmap (Blues).
        if "Agent" in axis_dfs:
            plot_agent_phq9_combined(
                axis_dfs["Agent"], phq9_matrix,
                os.path.join(args.out_dir, "agent_phq9_combined.png"),
            )

    print(f"\n[done] CSVs + plots under {args.out_dir}/")


if __name__ == "__main__":
    main()
