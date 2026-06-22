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
# OPTIMIZATION RUN SUMMARIES
# =====================================================================

def _read_instruction(seed_dir: str, prompt_type: str) -> str:
    """Return the best saved instruction for a prompt-optimizer run dir.

    Strips the ``# val score: X | test score: Y`` metadata header that the
    tweet optimizer prepends before the first blank line.
    """
    candidates = (
        ["optimized_instruction.txt", "best_instruction.txt"]
        if prompt_type == "phq9"
        else ["optimized_instruction_tweet.txt", "best_instruction_tweet.txt"]
    )
    for fname in candidates:
        p = os.path.join(seed_dir, fname)
        if os.path.isfile(p):
            text = open(p).read().strip()
            if text.startswith("#"):
                _hdr, _, rest = text.partition("\n\n")
                if rest:
                    text = rest.strip()
            return text
    return ""


def _summarize_prompt_run(seed_dir: str, prompt_type: str,
                           best_is: str = "min") -> dict | None:
    """One-row summary dict for a single prompt-optimizer run dir.

    best_is="min" for PHQ-9 (MAE, lower better), "max" for tweet quality.
    Returns None if no trajectory CSV is found.
    """
    import pandas as _pd
    traj_path = os.path.join(seed_dir, "training_trajectory.csv")
    if not os.path.isfile(traj_path):
        return None
    traj = _pd.read_csv(traj_path)
    if traj.empty:
        return None
    val   = traj[traj.split == "val"]
    train = traj[traj.split == "train"]
    test  = traj[traj.split == "test"]
    if val.empty:
        return None
    idx = val["mean_score"].idxmin() if best_is == "min" else val["mean_score"].idxmax()
    best_val_row = val.loc[idx]
    best_step = int(best_val_row["step"])
    train_at_best = train[train["step"] == best_step]
    test_row = test.iloc[-1] if not test.empty else None
    return {
        "best_step":     best_step,
        "best_val":      float(best_val_row["mean_score"]),
        "best_val_std":  float(best_val_row["std_score"]),
        "train_at_best": float(train_at_best["mean_score"].iloc[0]) if not train_at_best.empty else float("nan"),
        "train_std":     float(train_at_best["std_score"].iloc[0])  if not train_at_best.empty else float("nan"),
        "test_mean":     float(test_row["mean_score"]) if test_row is not None else float("nan"),
        "test_std":      float(test_row["std_score"])  if test_row is not None else float("nan"),
        "test_n":        int(test_row["n_samples"])    if test_row is not None else 0,
        "batch_size":    int(train["n_samples"].iloc[0]) if not train.empty else None,
        "val_size":      int(best_val_row["n_samples"]),
        "num_steps":     int(traj["step"].max()),
        "instruction":   _read_instruction(seed_dir, prompt_type),
    }


def summarize_prompt_runs(base_dir: str, seeds: list, prompt_type: str,
                           best_is: str = "min",
                           model_short: str = "Qwen3.5-27B") -> "pd.DataFrame":
    """Aggregate per-seed prompt-optimizer summaries into a DataFrame."""
    rows = []
    for seed in seeds:
        d = os.path.join(base_dir, f"{model_short}_seed{seed}")
        s = _summarize_prompt_run(d, prompt_type, best_is=best_is)
        if s is not None:
            s["seed"] = seed
            rows.append(s)
    cols = ["seed", "best_step", "best_val", "best_val_std",
            "train_at_best", "train_std", "test_mean", "test_std",
            "test_n", "batch_size", "val_size", "num_steps", "instruction"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def print_prompt_summary(df: "pd.DataFrame", title: str) -> None:
    """Print the numeric table and each seed's optimised instruction."""
    import textwrap
    print(f"\n=== {title} ===\n")
    print(df.drop(columns=["instruction"]).round(3).to_string(index=False))
    print("\n--- Optimized instructions ---")
    for _, r in df.iterrows():
        wrapped = textwrap.fill(r["instruction"], width=100,
                                initial_indent="  ", subsequent_indent="  ")
        print(f"\n[seed {int(r['seed'])}]  best_step={int(r['best_step'])}  "
              f"val={r['best_val']:.2f}  test={r['test_mean']:.2f}\n{wrapped}")


def summarize_bert_runs(base_dir: str = "data/test_post/bert_regression",
                        model_short: str = "Qwen3.5-27B") -> "pd.DataFrame":
    """Read performance.json for each BERT seed dir; return a summary DataFrame."""
    import glob as _glob
    import json as _json
    rows = []
    for d in sorted(_glob.glob(os.path.join(base_dir, f"{model_short}_seed*"))):
        try:
            seed = int(d.rsplit("seed", 1)[-1])
        except ValueError:
            continue
        perf_path = os.path.join(d, "performance.json")
        if not os.path.isfile(perf_path):
            continue
        with open(perf_path) as fh:
            perf = _json.load(fh)
        epochs = perf.get("epochs", [])
        if not epochs:
            continue
        best = min(epochs, key=lambda e: e["val_mae"])
        rows.append({
            "seed":              seed,
            "best_epoch":        int(best["epoch"]),
            "best_val_mae":      float(best["val_mae"]),
            "best_val_std":      float(best["val_std"]),
            "train_at_best_mae": float(best["train_mae"]),
            "train_at_best_std": float(best["train_std"]),
            "test_mae":          float(perf["test_mae"]),
            "test_std":          float(perf["test_std"]),
            "n_train":           int(perf["n_train"]),
            "n_val":             int(perf["n_val"]),
            "n_test":            int(perf["n_test"]),
            "total_epochs":      len(epochs),
            "mental_bert":       bool(perf.get("mental_bert", True)),
        })
    return pd.DataFrame(rows)


# =====================================================================
# PROMPT-STRING ROBUSTNESS (prompt sensitivity analysis)
# =====================================================================

_PROMPT_TYPE_META = {
    "post_gen": {
        "base_dir": "data/test_post/optimized_tweets",
        "filename": "best_instruction_tweet.txt",
    },
    "phq9": {
        "base_dir": "data/test_post/optimized_phq9",
        "filename": "best_instruction.txt",
    },
}


def load_best_prompts(model_name: str, seeds: list,
                      prompt_type: str = "post_gen") -> list:
    """Read the best instruction file for each seed run."""
    if prompt_type not in _PROMPT_TYPE_META:
        raise ValueError(f"prompt_type must be one of {list(_PROMPT_TYPE_META)}")
    meta = _PROMPT_TYPE_META[prompt_type]
    prompts = []
    for seed in seeds:
        path = os.path.join(meta["base_dir"], f"{model_name}_seed{seed}", meta["filename"])
        with open(path) as f:
            prompts.append(f.read().strip())
    return prompts


def load_test_scores(model_name: str, seeds: list,
                     prompt_type: str = "post_gen") -> list:
    """Read the final test-split mean_score from training_trajectory.csv for each seed."""
    import csv
    if prompt_type not in _PROMPT_TYPE_META:
        raise ValueError(f"prompt_type must be one of {list(_PROMPT_TYPE_META)}")
    base_dir = _PROMPT_TYPE_META[prompt_type]["base_dir"]
    scores = []
    for seed in seeds:
        path = os.path.join(base_dir, f"{model_name}_seed{seed}", "training_trajectory.csv")
        test_score = None
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == "test":
                    test_score = float(row["mean_score"])
        if test_score is None:
            raise ValueError(f"No test split found in {path}")
        scores.append(test_score)
    return scores


def load_minimal_score(model_name: str, seeds: list,
                       prompt_type: str = "post_gen") -> float:
    """Average the first val-split mean_score across seeds (un-optimised baseline)."""
    import csv
    if prompt_type not in _PROMPT_TYPE_META:
        raise ValueError(f"prompt_type must be one of {list(_PROMPT_TYPE_META)}")
    base_dir = _PROMPT_TYPE_META[prompt_type]["base_dir"]
    firsts = []
    for seed in seeds:
        path = os.path.join(base_dir, f"{model_name}_seed{seed}", "training_trajectory.csv")
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == "val":
                    firsts.append(float(row["mean_score"]))
                    break
            else:
                raise ValueError(f"No val split found in {path}")
    return sum(firsts) / len(firsts)


def _weighted_mae_from_phq9_csv(path: str, metric_col: str = "avg_mae") -> float:
    """Sample-weighted mean of a per-PHQ-9 metric CSV (cols: phq9, <metric>, n_samples)."""
    import csv
    num = den = 0.0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n = float(row["n_samples"])
            num += float(row[metric_col]) * n
            den += n
    if den == 0:
        raise ValueError(f"no samples in {path}")
    return num / den


def load_heldout_phq9_scores(model_name: str, seeds: list,
                             eval_subdir: str = "eval_on_test_blocks_seed35",
                             base_dir: str = "data/test_post/optimized_phq9",
                             metric_col: str = "avg_mae") -> list:
    """Sample-weighted MAE per seed from the held-out BERT-testset re-scoring.

    Reads ``<base_dir>/<model>_seed<seed>/<eval_subdir>/test_scores_phq9.csv`` — the
    optimized PHQ-9 prompts scored on the SAME blocks the MentalBERT regressor was
    tested on (produced by run_phq9_on_bert_testset.sh). Use in place of
    ``load_test_scores`` to annotate the prompt landscape with held-out,
    apples-to-apples scores rather than each run's own in-distribution test split.
    """
    return [
        _weighted_mae_from_phq9_csv(
            os.path.join(base_dir, f"{model_name}_seed{seed}", eval_subdir, "test_scores_phq9.csv"),
            metric_col,
        )
        for seed in seeds
    ]


def load_heldout_minimal_score(model_name: str, minimal_seed: int = 23,
                               eval_subdir: str = "eval_on_test_blocks_seed35_minimal",
                               base_dir: str = "data/test_post/optimized_phq9",
                               metric_col: str = "avg_mae") -> float:
    """Sample-weighted MAE of the MINIMAL prompt on the same held-out blocks.

    Counterpart to ``load_heldout_phq9_scores`` for the un-optimized baseline.
    Requires the minimal re-run that writes to ``<eval_subdir>`` (a distinct
    posts-file stem so it does not overwrite the optimized seed-<minimal_seed> eval).
    """
    path = os.path.join(base_dir, f"{model_name}_seed{minimal_seed}",
                        eval_subdir, "test_scores_phq9.csv")
    return _weighted_mae_from_phq9_csv(path, metric_col)


def compute_prompt_robustness(prompts: list, test_scores: list,
                               baseline_prompt: str = None,
                               baseline_score: float = None,
                               seeds: list = None,
                               labels: list = None,
                               model_name: str = "all-MiniLM-L6-v2") -> dict:
    """Pairwise cosine-similarity matrix across optimised prompts.

    When `baseline_prompt` is provided it is appended as the final row/column
    and labelled "minimal". Returns a dict with keys: sim_matrix, labels,
    test_scores, baseline_score, has_baseline.
    """
    from sentence_transformers import SentenceTransformer
    all_prompts = list(prompts)
    if baseline_prompt is not None:
        all_prompts.append(baseline_prompt)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(all_prompts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-10)
    sim_matrix = (normed @ normed.T).astype(float)

    if labels is not None:
        if len(labels) != len(prompts):
            raise ValueError(f"labels length {len(labels)} != prompts length {len(prompts)}")
        resolved_labels = list(labels)
    elif seeds is not None and len(seeds) == len(prompts):
        resolved_labels = [f"seed {s}" for s in seeds]
    else:
        resolved_labels = [f"run {i+1}" for i in range(len(prompts))]
    if baseline_prompt is not None:
        resolved_labels.append("minimal")

    return {
        "sim_matrix": sim_matrix,
        "labels": resolved_labels,
        "test_scores": list(test_scores),
        "baseline_score": baseline_score,
        "has_baseline": baseline_prompt is not None,
    }


def _sort_by_score(prompts: list, scores: list, seeds: list,
                   ascending: bool) -> tuple:
    """Sort prompts/scores/seeds together; return (prompts, scores, seeds, labels)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i] if ascending else -scores[i])
    p = [prompts[i] for i in order]
    s = [scores[i]  for i in order]
    e = [seeds[i]   for i in order]
    labels = [f"{s[i]:.2f}" for i in range(len(order))]
    return p, s, e, labels


def _draw_prompt_sim_heatmap(ax, robustness: dict, caption: str,
                              vmin: float = None) -> None:
    """Draw a cosine-similarity heatmap panel onto `ax`.

    Cell colour = cosine similarity. Cell annotation = absolute test-score
    difference between the two runs. `caption` is placed under the panel as
    an x-axis label (e.g. "(a) PHQ-9 assessment prompt").
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    sim = np.array(robustness["sim_matrix"])
    labels = robustness["labels"]
    has_baseline = bool(robustness.get("has_baseline"))
    test_scores = list(robustness["test_scores"])
    n_total = sim.shape[0]
    if vmin is None:
        vmin = float(np.min(sim[sim < 1.0])) if (sim < 1.0).any() else 0.0
        vmin = max(0.0, vmin - 0.02)

    scores_with_base = list(test_scores)
    if has_baseline:
        scores_with_base.append(robustness.get("baseline_score"))

    annot = np.empty(sim.shape, dtype=object)
    for i in range(n_total):
        for j in range(n_total):
            si, sj = scores_with_base[i], scores_with_base[j]
            if si is None or sj is None:
                annot[i, j] = ""
            else:
                annot[i, j] = f"{np.abs(si - sj):.2f}"

    sns.heatmap(
        sim, ax=ax,
        xticklabels=labels, yticklabels=labels,
        vmin=vmin, vmax=1.0,
        annot=annot, fmt="",
        cmap="Blues", linewidths=0.4, linecolor="white",
        cbar=False, annot_kws={"fontsize": 8},
    )
    ax.set_aspect("equal")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    ax.set_xlabel(caption, fontsize=10, labelpad=12)


def plot_prompt_sensitivity_pair(robustness_phq9: dict, robustness_post_gen: dict,
                                  model_name: str, output_dir: str = None,
                                  cell: float = 0.55) -> str:
    """Render PHQ-9 and post-generation cosim heatmaps side by side and save.

    Both panels share ONE colour scale (a single shared ``vmin``), but each panel
    gets its own docked colourbar drawn via ``fig.colorbar`` so the left and right
    bars carry the same small black outline (matching
    ``plot_prompt_output_string_pair``).
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n_left = len(robustness_phq9["labels"])
    n_right = len(robustness_post_gen["labels"])
    panel_w = lambda n: max(3.0, n * 0.8 * cell)
    fig_w = panel_w(n_left) + panel_w(n_right) + 1.4
    fig_h = max(3.0, max(n_left, n_right) * cell * 0.8 + 1.0)

    sim_left = np.array(robustness_phq9["sim_matrix"])
    sim_right = np.array(robustness_post_gen["sim_matrix"])
    sim_all = np.concatenate([sim_left.ravel(), sim_right.ravel()])
    shared_vmin = float(np.min(sim_all[sim_all < 1.0])) if (sim_all < 1.0).any() else 0.0
    shared_vmin = max(0.0, shared_vmin - 0.02)

    fig, (ax_phq9, ax_post) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [panel_w(n_left), panel_w(n_right)],
                     "wspace": 0.55},
    )

    for ax, robustness, caption in (
            (ax_phq9, robustness_phq9, "(a) PHQ-9 assessment prompt"),
            (ax_post, robustness_post_gen, "(b) Post-generation prompt")):
        # Dock a colourbar to each panel via fig.colorbar (not seaborn's) so both
        # the left and right bars get the same small black outline.
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
        _draw_prompt_sim_heatmap(ax, robustness, caption=caption, vmin=shared_vmin)
        fig.colorbar(
            ScalarMappable(norm=Normalize(vmin=shared_vmin, vmax=1.0), cmap="Blues"),
            cax=cax)
        cax.set_ylabel("cosine similarity", fontsize=8)
        cax.tick_params(labelsize=7)

    if output_dir is None:
        output_dir = os.path.join(_PROMPT_TYPE_META["phq9"]["base_dir"],
                                  f"{model_name}_sensitivity")
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "prompt_sensitivity_pair.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pair sensitivity plot → {out}")
    return out


def draw_prompt_sim_heatmap(ax, robustness: dict, title: str,
                             cmap: str = "Blues",
                             annot_fmt: str = "{:.2f}") -> None:
    """Draw one cosine-similarity heatmap onto `ax` (annotation = mean test score).

    Cell annotation = mean test score of the two runs; baseline row/col blank.
    """
    sim = np.array(robustness["sim_matrix"])
    labels = robustness["labels"]
    has_baseline = bool(robustness.get("has_baseline"))
    test_scores = list(robustness["test_scores"])
    n_runs = len(test_scores)
    n_total = sim.shape[0]

    vmin = float(np.min(sim[sim < 1.0])) if (sim < 1.0).any() else 0.0
    vmin = max(0.0, vmin - 0.02)

    annot = np.empty(sim.shape, dtype=object)
    for i in range(n_total):
        for j in range(n_total):
            i_is_base = has_baseline and i == n_runs
            j_is_base = has_baseline and j == n_runs
            if i_is_base or j_is_base:
                annot[i, j] = ""
            else:
                annot[i, j] = annot_fmt.format((test_scores[i] + test_scores[j]) / 2.0)

    sns.heatmap(
        sim, ax=ax,
        xticklabels=labels, yticklabels=labels,
        vmin=vmin, vmax=1.0,
        annot=annot, fmt="",
        cmap=cmap, linewidths=0.4, linecolor="white",
        cbar_kws={"label": "cosine similarity", "shrink": 0.8},
    )
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)


# =====================================================================
# Generated-OUTPUT cosine companion to plot_prompt_sensitivity_pair
#   Same figure, but the LEFT panel is switched from the PHQ-9 prompt-string
#   heatmap to the within/cross cosine of the prompts' GENERATED OUTPUT.
# =====================================================================

def load_prompt_reps(root: str) -> dict:
    """``{(label, rep): {embeddings, agent_ids, rounds, phq9}}`` from
    ``<root>/<label>/rep_<N>/embeddings.npz`` (written by sa_embed)."""
    runs: dict = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "rep_*", "embeddings.npz"))):
        parts = p.split(os.sep)
        label, rep = parts[-3], int(parts[-2].split("_")[1])
        d = np.load(p, allow_pickle=True)
        runs[(label, rep)] = {k: d[k] for k in ("embeddings", "agent_ids", "rounds", "phq9")}
    return runs


def output_cosine_matrix(df: pd.DataFrame, labels: list,
                         agg: str = "mean") -> np.ndarray:
    """N×N generated-output cosine matrix from a prompt-reps ``neighbor_cosines``
    frame: diagonal = within-prompt (LLM-noise floor), off-diagonal = cross-prompt.

    ``agg`` is the reducer over the per-(agent, round) pairwise cosines in each
    cell — "mean" (default) or "median". The standalone prompt-reps heatmap CLI
    (``prompt_reps_main``) uses the median; the paired figure defaults to the mean
    so it matches the mean-based convention of the other experiment.ipynb SA
    figures. ``labels`` fixes the row/column order (and which prompts are included).
    """
    reducer = {"mean": np.mean, "median": np.median}[agg]
    within = {lab: float(reducer(g.cosine.values))
              for lab, g in df[df.pair_type == "within"].groupby("setting_a")}
    cross = {frozenset((a, b)): float(reducer(g.cosine.values))
             for (a, b), g in df[df.pair_type == "cross"].groupby(["setting_a", "setting_b"])}
    n = len(labels)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            mat[i, j] = within.get(a, np.nan) if i == j \
                else cross.get(frozenset((a, b)), np.nan)
    return mat


def _draw_output_sim_heatmap(ax, mat: np.ndarray, tick_labels: list, caption: str,
                             cbar_ax=None, cmap: str = "Blues",
                             vmin: float = None, vmax: float = None) -> None:
    """Draw a generated-output cosine heatmap (diag = within-prompt noise floor,
    off-diag = cross-prompt) onto ``ax``, styled like ``_draw_prompt_sim_heatmap``.

    Annotation = the cosine value itself (the quantity of interest here), unlike
    the prompt-string panel which annotates the test-score gap. ``cbar_ax`` (if
    given) receives this panel's own colourbar; the scale is tight on this
    panel's values so it stays readable next to the wider-ranged prompt panel.
    """
    vals = mat[~np.isnan(mat)]
    if vmin is None:
        vmin = max(0.0, float(vals.min()) - 0.005)
    if vmax is None:
        vmax = min(1.0, float(vals.max()) + 0.005)
    sns.heatmap(
        mat, ax=ax,
        xticklabels=tick_labels, yticklabels=tick_labels,
        vmin=vmin, vmax=vmax,
        annot=True, fmt=".3f", annot_kws={"fontsize": 8},
        cmap=cmap, linewidths=0.4, linecolor="white",
        cbar=cbar_ax is not None, cbar_ax=cbar_ax,
        square=True,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    ax.set_xlabel(caption, fontsize=10, labelpad=12)


def plot_prompt_output_string_pair(
        robustness_post_gen: dict,
        reps_labels: list,
        model_name: str = "Qwen3.5-27B",
        reps_root: str = "data/prompt_optimization_h/qwen27_baseline/prompt_sa_reps",
        agg: str = "mean",
        left_caption: str = "(a) Generated-output cosine",
        right_caption: str = "(b) Post-generation prompt",
        output_dir: str = None,
        cell: float = 0.55) -> str:
    """Companion to ``plot_prompt_sensitivity_pair`` with the LEFT panel switched.

    Left  panel: GENERATED-OUTPUT cosine for the post-gen prompts — within-prompt
                 (diagonal = LLM-noise floor) vs cross-prompt, aggregated with
                 ``agg`` ("mean" by default) over per-(agent, round) post pairs,
                 built from the replicate tree under ``reps_root``.
    Right panel: the existing post-generation PROMPT-STRING cosine heatmap,
                 unchanged (cosine of prompt embeddings; annotation = |Δ score|).

    Both panels show the SAME prompts in the SAME order, so the two similarity
    views line up row-for-row. ``reps_labels`` gives each non-baseline
    ``robustness_post_gen`` row its replicate-tree directory label (e.g.
    ``textgrad_seed24`` or ``iter_10`` — all optimized prompts are treated alike),
    in order; "minimal" is appended automatically when a baseline is present.
    Display tick labels come straight from ``robustness_post_gen["labels"]``.

    Each panel gets its OWN tight colour scale + colourbar: the output cosines
    (~0.86–0.92) and prompt-string cosines (~0.6–1.0) live on different ranges,
    so one shared bar would wash the left panel out.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    has_baseline = bool(robustness_post_gen.get("has_baseline"))
    tick_labels = list(robustness_post_gen["labels"])

    # Replicate-tree label per row (+ minimal baseline), matching the robustness rows.
    order = list(reps_labels)
    if has_baseline:
        order.append("minimal")
    if len(order) != len(tick_labels):
        raise ValueError(
            f"{len(order)} replicate labels but {len(tick_labels)} robustness "
            "labels — reps_labels must match robustness_post_gen (minus 'minimal').")

    runs = load_prompt_reps(reps_root)
    have = {lab for (lab, _) in runs}
    missing = [lab for lab in order if lab not in have]
    if missing:
        raise SystemExit(f"no replicate embeddings for {missing} under {reps_root} "
                         "(run sa_prompt_baseline_run.sh + sa_embed first).")
    df = neighbor_cosines(runs)
    mat = output_cosine_matrix(df, order, agg=agg)

    # ── Layout: two square panels, each with its own docked colourbar. ──
    n_left, n_right = len(order), len(tick_labels)
    panel_w = lambda n: max(3.0, n * 0.8 * cell)
    fig_w = panel_w(n_left) + panel_w(n_right) + 1.4
    fig_h = max(3.0, max(n_left, n_right) * cell * 0.8 + 1.0)
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [panel_w(n_left), panel_w(n_right)],
                     "wspace": 0.55},
    )

    # LEFT: generated-output cosine (own tight colourbar). Drawn via fig.colorbar
    # (not seaborn's) so it gets the same black outline as the right panel's bar.
    vals_l = mat[~np.isnan(mat)]
    vmin_l = max(0.0, float(vals_l.min()) - 0.005)
    vmax_l = min(1.0, float(vals_l.max()) + 0.005)
    cax_l = make_axes_locatable(ax_left).append_axes("right", size="5%", pad=0.08)
    _draw_output_sim_heatmap(ax_left, mat, tick_labels, left_caption,
                             vmin=vmin_l, vmax=vmax_l)
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin_l, vmax=vmax_l), cmap="Blues"),
                 cax=cax_l)
    cax_l.set_ylabel("cosine similarity", fontsize=8)
    cax_l.tick_params(labelsize=7)

    # RIGHT: existing post-gen prompt-string heatmap, unchanged styling.
    sim_r = np.array(robustness_post_gen["sim_matrix"])
    vmin_r = float(np.min(sim_r[sim_r < 1.0])) if (sim_r < 1.0).any() else 0.0
    vmin_r = max(0.0, vmin_r - 0.02)
    cax_r = make_axes_locatable(ax_right).append_axes("right", size="5%", pad=0.08)
    _draw_prompt_sim_heatmap(ax_right, robustness_post_gen, right_caption, vmin=vmin_r)
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin_r, vmax=1.0), cmap="Blues"),
                 cax=cax_r)
    cax_r.set_ylabel("cosine similarity", fontsize=8)
    cax_r.tick_params(labelsize=7)

    if output_dir is None:
        # These are the post-generation (tweet) prompts, so default under
        # optimized_tweets rather than optimized_phq9.
        output_dir = os.path.join(_PROMPT_TYPE_META["post_gen"]["base_dir"],
                                  f"{model_name}_sensitivity")
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "prompt_output_string_pair.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Output/string pair plot → {out}")
    return out


# =====================================================================
# Standalone prompt-reps heatmap (merged from the old sa_prompt_baseline.py):
# within/cross MEDIAN cosine heatmap + CLI. Each prompt is drawn several times
# (unseeded) with personas/neighbours/PHQ-9 fixed, so the diagonal is the real
# within-prompt LLM-noise floor and the off-diagonal is the cross-prompt median.
# Built by sa_prompt_baseline_run.sh (+ sa_embed); run via:
#   PYTHONPATH=src python -m utils.sensitivity.sa_analyze --prompt-reps --root <reps_root>
# =====================================================================

def plot_prompt_reps_heatmap(mat: np.ndarray, labels: list, out_path: str) -> None:
    """N×N heatmap with a TIGHT colour scale so the diagonal (within-prompt floor)
    and off-diagonals (cross-prompt) are visually comparable."""
    n = mat.shape[0]
    vals = mat[~np.isnan(mat)]
    vmin, vmax = float(vals.min()), float(vals.max())
    pad = max(0.003, (vmax - vmin) * 0.05)
    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * n + 2), max(4.5, 0.9 * n + 1.5)))
    sns.heatmap(mat, ax=ax, xticklabels=labels, yticklabels=labels,
                vmin=vmin - pad, vmax=vmax + pad, annot=True, fmt=".3f",
                annot_kws={"fontsize": 9}, cmap="Blues",
                linewidths=0.4, linecolor="white", square=True,
                cbar_kws={"label": "median per-anchor cosine"})
    ax.set_title("Prompt sensitivity — within-prompt (diagonal) vs cross-prompt\n"
                 "(median per-(agent, round) cosine; diagonal = LLM-noise floor)")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_path}")


def prompt_reps_main(argv=None) -> None:
    """CLI: per-prompt within/cross MEDIAN cosine heatmap from a replicate tree."""
    ap = argparse.ArgumentParser(
        description=prompt_reps_main.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--root",
        default="data/prompt_optimization_h/qwen27_baseline/prompt_sa_reps",
        help="Dir holding <label>/rep_*/embeddings.npz (run sa_embed first).")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write CSVs/PNG (default: <root>/prompt_cosine_reps).")
    args = ap.parse_args(argv)

    out_dir = args.out_dir or os.path.join(args.root, "prompt_cosine_reps")
    os.makedirs(out_dir, exist_ok=True)

    runs = load_prompt_reps(args.root)
    if not runs:
        raise SystemExit(
            f"no embeddings.npz under {args.root}/<label>/rep_* — "
            "run `python -m utils.sensitivity.sa_embed --root <root>` first.")
    labels = sorted({lab for (lab, _) in runs})
    reps_per = {lab: sorted(r for (l, r) in runs if l == lab) for lab in labels}
    print(f"[prompt-reps] {len(runs)} runs across {len(labels)} prompts")
    for lab in labels:
        print(f"  {lab}: reps {reps_per[lab]}")
    if max(len(v) for v in reps_per.values()) < 2:
        raise SystemExit("no prompt has >=2 reps — cannot compute a within-prompt floor.")

    df = neighbor_cosines(runs)
    df.to_csv(os.path.join(out_dir, "prompt_rep_cosines.csv"), index=False)

    within = df[df.pair_type == "within"]
    cross = df[df.pair_type == "cross"]
    print("\n[prompt-reps] WITHIN-prompt (LLM-noise floor) median cosine per prompt:")
    print(within.groupby("setting_a").cosine.median().round(4).to_string())
    print(f"  pooled within (floor) median = {within.cosine.median():.4f}")
    if not cross.empty:
        print(f"[prompt-reps] CROSS-prompt median cosine        = {cross.cosine.median():.4f}")

    mat = output_cosine_matrix(df, labels, agg="median")
    pd.DataFrame(mat, index=labels, columns=labels).to_csv(
        os.path.join(out_dir, "prompt_rep_heatmap_matrix.csv"))
    if len(labels) >= 2:
        plot_prompt_reps_heatmap(mat, labels, os.path.join(out_dir, "prompt_rep_heatmap.png"))
    else:
        print("[prompt-reps] <2 prompts — skipping heatmap (floor only).")

    print(f"\n[done] outputs under {out_dir}/")
    print("       diagonal = within-prompt noise floor; off-diagonal = cross-prompt median.")
    print("       Off-diagonal well below its row/col diagonal => prompt shifts output beyond noise.")


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
    import sys
    if "--prompt-reps" in sys.argv[1:]:
        prompt_reps_main([a for a in sys.argv[1:] if a != "--prompt-reps"])
    else:
        main()
