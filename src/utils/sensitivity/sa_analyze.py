"""Sensitivity analysis: cosine within-setting vs cross-setting, stratified by PHQ-9.

For each axis (neighbour, agent, joint, decoding):
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


def mean_pairwise_cos(embs: np.ndarray) -> float:
    """Mean cosine over all distinct pairs of rows in ``embs`` (a set's internal
    similarity). O(n) via the normalised-sum identity
    ``mean_{i≠j} x_i·x_j = (‖Σ x̂‖² − n) / (n(n−1))`` rather than the n² loop."""
    if embs.shape[0] < 2:
        return float("nan")
    X = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    n = X.shape[0]
    S = X.sum(axis=0)
    return float((S @ S - n) / (n * (n - 1)))


def _parse_setting(token: str):
    """Setting id from a 'setting_<x>' directory name.

    Numeric labels (neighbour/agent/joint axes: setting_11, ...) parse to int so
    their sort order and "seed N" display stay exactly as before. Non-numeric
    labels (decoding axis: setting_baseline, setting_temp_hi, ...) are kept as
    strings.
    """
    name = token[len("setting_"):] if token.startswith("setting_") else token
    return int(name) if name.lstrip("-").isdigit() else name


def _setting_label(s) -> str:
    """Heatmap tick label: 'seed N' for numeric settings (seeds), else the raw
    string label (the decoding axis's temp_hi / baseline / ...)."""
    return f"seed {s}" if isinstance(s, (int, np.integer)) else str(s)


def load_axis_runs(root: str, axis: str, emb_name: str = "embeddings.npz") -> dict:
    """Return {(setting, rep): {embeddings, agent_ids, rounds, phq9}}.

    ``setting`` is an int for seed-labelled axes, a str for the decoding axis.
    ``emb_name`` selects the encoder's .npz (``embeddings.npz`` = MentalBERT,
    ``embeddings_sbert.npz`` = SBERT for the content/topic axis).
    """
    paths = sorted(glob.glob(os.path.join(root, axis, "setting_*", "rep_*", emb_name)))
    runs = {}
    for p in paths:
        parts = p.split(os.sep)
        setting = _parse_setting(next(x for x in parts if x.startswith("setting_")))
        rep = int(next(x for x in parts if x.startswith("rep_")).split("_")[1])
        data = np.load(p, allow_pickle=True)
        runs[(setting, rep)] = {
            "embeddings": data["embeddings"],
            "agent_ids":  data["agent_ids"],
            "rounds":     data["rounds"],
            "phq9":       data["phq9"],
        }
    return runs


def load_phq9_runs(root: str, emb_name: str = "embeddings.npz") -> dict:
    """Return {(band, rep): {embeddings, agent_ids, rounds, phq9}} for the PHQ-9
    conditioning runs under ``root/phq9/<band>/rep_*/<emb_name>``.

    Keyed so it drops straight into ``neighbor_cosines`` with the band playing
    the role of "setting": within = same band, different reps (the LLM-noise
    floor); cross = different bands (the conditioning effect). Falls back to the
    legacy single-dir layout (``phq9/<band>/<emb_name>``, rep 1) when no rep_*
    dirs exist.
    """
    paths = sorted(glob.glob(os.path.join(root, "phq9", "*", "rep_*", emb_name)))
    legacy = not paths
    if legacy:
        paths = sorted(glob.glob(os.path.join(root, "phq9", "*", emb_name)))
    runs = {}
    for p in paths:
        parts = p.split(os.sep)
        band, rep = (parts[-2], 1) if legacy else \
            (parts[-3], int(parts[-2].split("_")[1]))
        data = np.load(p, allow_pickle=True)
        runs[(band, rep)] = {
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
# PHQ-9 conditioning: 5 band-settings, same agent + neighbour, vary PHQ-9
# =====================================================================

def phq9_conditioning_matrix(root: str, out_dir: str,
                             emb_name: str = "embeddings.npz",
                             diag_floor: dict | None = None) -> pd.DataFrame | None:
    """Build the band×band cosine matrix from the PHQ-9 conditioning runs.

    For each (agent_id, round) anchor that exists in all 5 band-settings,
    cosine every pair of the 5 embeddings. Bin each cosine by the
    (source_band, target_band) the two embeddings were generated under, and
    average. The result is a 5×5 matrix whose off-diagonal entries answer
    "how much do outputs change when the same persona/neighbour input is
    re-conditioned on a different PHQ-9 band?".

    ``diag_floor`` (optional ``{band: cosine}``) supplies the diagonal: the
    LLM-noise floor — the cosine you'd get by repeating the SAME band with
    everything fixed (only the seed changes). The PHQ-9 runs have no repeats, so
    this is borrowed from the agent axis's per-band within-setting cosine (same
    model + baseline decoding). Off-diagonal < diagonal ⇒ re-conditioning moves
    output beyond LLM noise. Falls back to each band's across-persona pairwise
    cosine when no floor is given.
    """
    # Rep-aware discovery: each band may have several rep_* runs (the native
    # within-band noise floor) or a single legacy run. Match both layouts.
    paths = sorted(glob.glob(os.path.join(root, "phq9", "*", "rep_*", emb_name)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "phq9", "*", emb_name)))
    if not paths:
        print(f"[phq9-cond] no {emb_name} under {root}/phq9/")
        return None

    # run_id (band_dir, rep) → (data, idx_by_anchor, dominant_band)
    runs: dict[tuple, tuple] = {}
    for p in paths:
        parts = p.split(os.sep)
        band_dir, rep = (parts[-3], parts[-2]) if parts[-2].startswith("rep_") \
            else (parts[-2], "rep_1")
        data = np.load(p, allow_pickle=True)
        idx = {(int(a), int(r)): i for i, (a, r) in
               enumerate(zip(data["agent_ids"], data["rounds"]))}
        bands_here = {phq9_to_band(int(s)) for s in data["phq9"]}
        if len(bands_here) != 1:
            print(f"[phq9-cond] WARNING: {band_dir}/{rep} spans bands {bands_here}; "
                  f"taking the most-common one for the axis label.")
        dom_band = pd.Series([phq9_to_band(int(s)) for s in data["phq9"]]).mode().iloc[0]
        runs[(band_dir, rep)] = (data, idx, dom_band)

    # Ordered band axis (unique dominant bands, BAND_LABELS order).
    band_order = {b: i for i, b in enumerate(BAND_LABELS)}
    band_labels_ax = sorted({b for (_, _, b) in runs.values()},
                            key=lambda b: band_order.get(b, 99))
    keys = sorted(runs.keys(),
                  key=lambda k: (band_order.get(runs[k][2], 99), k[1]))
    reps_per_band = pd.Series([runs[k][2] for k in keys]).value_counts().to_dict()
    print(f"[phq9-cond] {len(keys)} runs over {len(band_labels_ax)} bands "
          f"(reps/band: {reps_per_band})")

    # Common anchors across all runs.
    common = None
    for (_, idx, _) in runs.values():
        anchors = set(idx.keys())
        common = anchors if common is None else common & anchors
    common = sorted(common or [])
    print(f"[phq9-cond] {len(common)} common (agent, round) anchors")

    # Pairwise cosines binned by (source_band, target_band). Same-band pairs come
    # from DIFFERENT reps (when reps exist), so the diagonal cell collects the
    # within-band rep-to-rep cosine = the LLM-noise floor; cross-band pairs fill
    # the off-diagonal. All pairs are SAME anchor, so persona is held fixed.
    cell_values: dict[tuple, list[float]] = defaultdict(list)
    for (aid, rd) in common:
        embs = {k: runs[k][0]["embeddings"][runs[k][1][(aid, rd)]] for k in keys}
        for ka, kb in combinations(keys, 2):
            band_a, band_b = runs[ka][2], runs[kb][2]
            cs = float(cosine_rows(embs[ka][None, :], embs[kb][None, :])[0])
            cell_values[(band_a, band_b)].append(cs)
            if band_a != band_b:                       # keep matrix symmetric
                cell_values[(band_b, band_a)].append(cs)

    # Build the N×N matrix. Diagonal = native within-band rep-to-rep cosine when
    # ≥2 reps exist; otherwise fall back to the supplied agent-axis floor, then to
    # the band's across-persona pairwise cosine (so it is never a hard 1.0).
    n = len(band_labels_ax)
    mat = np.full((n, n), np.nan)
    for i, ba in enumerate(band_labels_ax):
        for j, bb in enumerate(band_labels_ax):
            vals = cell_values.get((ba, bb), [])
            if i == j and not vals:                    # single rep → no within pairs
                if diag_floor and np.isfinite(diag_floor.get(ba, np.nan)):
                    mat[i, j] = float(diag_floor[ba])
                else:
                    be = np.array([runs[k][0]["embeddings"][runs[k][1][a]]
                                   for k in keys if runs[k][2] == ba for a in common])
                    mat[i, j] = mean_pairwise_cos(be)
            else:
                mat[i, j] = float(np.mean(vals)) if vals else np.nan

    df = pd.DataFrame(mat, index=band_labels_ax, columns=band_labels_ax)
    df.to_csv(os.path.join(out_dir, "phq9_conditioning_matrix.csv"))
    return df


# =====================================================================
# Three-axis comparison: per-anchor cosine drop (within − cross) per axis
# =====================================================================

def _per_anchor_drops(df: pd.DataFrame, value_col: str = "cosine",
                      anchor_cols=("agent_id", "round"),
                      drop_sign: float = 1.0) -> pd.Series:
    """For each anchor (grouped by ``anchor_cols``), drop =
    ``drop_sign * (mean within − mean cross)`` of ``value_col``. One value per
    anchor — the distribution this returns is what the box plot visualises.

    Defaults give the cosine within−cross drop (axis moves output more than LLM
    noise ⇒ positive). The MentalBERT+MLP figure passes ``value_col='delta'``
    with ``drop_sign=-1`` so the box shows cross−within of |Δ predicted PHQ-9|
    (factor pushes predicted severity past noise ⇒ positive)."""
    grouped = df.groupby(list(anchor_cols))
    drops = []
    for _, sub in grouped:
        within = sub.loc[sub.pair_type == "within", value_col]
        cross  = sub.loc[sub.pair_type == "cross",  value_col]
        if len(within) == 0 or len(cross) == 0:
            continue
        drops.append(float(drop_sign * (within.mean() - cross.mean())))
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
                           n_per_anchor: int = 20, value_col: str = "cosine",
                           anchor_cols=("agent_id", "round")) -> np.ndarray:
    """For each anchor, compute the magnitude of a 'null drop' = |mean(half1) − mean(half2)|
    on random half-splits of that anchor's within-setting ``value_col`` values. This is the
    drop you'd see if the axis under test had NO effect — just LLM stochasticity.
    """
    nulls = []
    for _, sub in df[df.pair_type == "within"].groupby(list(anchor_cols)):
        vals = sub[value_col].values
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
                        n_bootstrap: int = 1000, seed: int = 0, *,
                        value_col: str = "cosine", ylabel: str = "Cosine drop",
                        anchor_cols=("agent_id", "round"), drop_sign: float = 1.0):
    """Side-by-side: box plot (left, full per-anchor distribution) + forest
    plot (right, mean within−cross with 95 % bootstrap CI). Adds an LLM-noise
    NULL reference: an extra box of per-anchor null drops, and a horizontal line
    at the null median on both panels. Anything above the line exceeds
    irreducible LLM stochasticity.

    ``value_col`` / ``anchor_cols`` / ``drop_sign`` / ``ylabel`` let the same
    figure render either the cosine within−cross drop (defaults) or the
    MentalBERT+MLP |Δ predicted PHQ-9| cross−within drop (sa_phq9 passes
    ``value_col='delta', ylabel='|Δ predicted PHQ-9|',
    anchor_cols=('agent_a',), drop_sign=-1``)."""
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
        "Decoding":  "#2e8b57",   # sea green — temperature/top_p axis
        "PHQ-9":     "#6a3d9a",   # purple — depression-severity conditioning axis
    }
    colours = [_COLOUR_BY_NAME.get(n, "#7f7f7f") for n in names]

    drops = [_per_anchor_drops(axis_dfs[n], value_col=value_col,
                               anchor_cols=anchor_cols, drop_sign=drop_sign).values
             for n in names]

    # Null = LLM-noise drops, computed from each axis's within-set then pooled.
    null_per_axis = [_per_anchor_null_drops(axis_dfs[n], rng, value_col=value_col,
                                            anchor_cols=anchor_cols) for n in names]
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
    # factor boxes (one per axis) + 1 LLM-noise null box.
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
    ax_box.set_ylabel(ylabel)
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
    ax_forest.set_xlabel(ylabel)
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


# =====================================================================
# DECODING axis — per-setting CENTROID-shift comparison.
#
# The within−cross noise-floor design used for the structural axes is INVALID
# here: temperature / top_p change the LLM's own rep-to-rep stochasticity, so a
# higher-temperature setting scatters more and drags cross cosine down even when
# the content has not shifted — there is no single, shared noise floor. Instead
# we average each setting's reps into a per-anchor centroid (cancelling per-run
# scatter) and compare baseline-centroid vs variant-centroid. The "no-shift"
# reference is a noise-matched permutation null (re-split of the SAME pooled
# reps), so the floor inherits each setting's own noise level.
# =====================================================================

# How the setting_* dir labels group into one box each. Each variant box pools
# its hi+lo settings; baseline (temp 0.7 / top_p 0.9) is the ~1 reference all
# variants are compared against (a centroid is identical to itself).
_DECODING_GROUPS = [
    ("temp",  "temp_",  "#2e7ebc"),   # temperature off default, top_p fixed
    ("top_p", "topp_",  "#d96907"),   # top_p off default, temperature fixed
    ("both",  "both_",  "#2e8b57"),   # both knobs off default
]


def _anchor_centroids(runs: dict):
    """Stack each setting's per-rep embeddings for every common anchor.

    Returns ``(settings, reps, per_anchor)`` where ``per_anchor`` maps
    ``(agent_id, round) -> {setting: ndarray (n_reps, dim)}``, keeping only
    anchors present in every run."""
    settings = sorted({s for (s, _) in runs.keys()})
    reps = sorted({r for (_, r) in runs.keys()})

    indexed = {}
    for key, data in runs.items():
        idx = {(int(a), int(r)): i for i, (a, r) in
               enumerate(zip(data["agent_ids"], data["rounds"]))}
        indexed[key] = (data, idx)

    common = None
    for (_, idx) in indexed.values():
        keys = set(idx.keys())
        common = keys if common is None else common & keys
    common = sorted(common or [])

    per_anchor = {}
    for (aid, rd) in common:
        by_setting = {}
        for s in settings:
            vecs = [indexed[(s, r)][0]["embeddings"][indexed[(s, r)][1][(aid, rd)]]
                    for r in reps if (s, r) in indexed]
            if vecs:
                by_setting[s] = np.vstack(vecs)
        per_anchor[(aid, rd)] = by_setting
    return settings, reps, per_anchor


def _centroid_cos(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cosine between the centroids (mean embedding) of two rep-groups."""
    ca = group_a.mean(axis=0)
    cb = group_b.mean(axis=0)
    return float(cosine_rows(ca[None, :], cb[None, :])[0])


def _perm_null_centroid_cos(base: np.ndarray, var: np.ndarray) -> float:
    """Noise-matched 'no content shift' reference for one anchor.

    Pool the baseline + variant reps, then average the centroid cosine over
    every split into baseline-sized vs variant-sized halves EXCEPT the true
    baseline|variant split. Because the pooled reps carry both settings'
    stochasticity, this floor inherits each setting's own noise level — the
    whole point of moving to centroids. If a variant only adds scatter (no
    content shift), its real centroid cosine ≈ this null; if it shifts content,
    the real cosine drops below it (pure-vs-pure separates the two contents,
    while mixed-vs-mixed blends them)."""
    pool = np.vstack([base, var])
    nb, n = base.shape[0], base.shape[0] + var.shape[0]
    true_a, true_b = frozenset(range(nb)), frozenset(range(nb, n))
    vals = []
    for combo in combinations(range(n), nb):
        a_set = frozenset(combo)
        if a_set == true_a or a_set == true_b:        # skip the real split
            continue
        b_idx = [i for i in range(n) if i not in a_set]
        vals.append(_centroid_cos(pool[list(combo)], pool[b_idx]))
    return float(np.mean(vals)) if vals else np.nan


def comparison_decoding_centroids(runs: dict, out_path: str,
                                  baseline: str = "baseline",
                                  n_bootstrap: int = 1000, seed: int = 0):
    """Box + forest plot of per-anchor CENTROID cosine for the decoding axis.

    One box per variant group (temp / top_p / both), each pooling its hi+lo
    settings: lower cosine(baseline-centroid, variant-centroid) = the TYPICAL
    output moved further from baseline (a systematic content shift, not just
    more scatter). The grey marker on every box/row is the noise-matched
    permutation floor (`_perm_null_centroid_cos`); a box clearly BELOW its floor
    shifts content beyond what that setting's stochasticity alone produces.
    See the section header for why within−cross is invalid on this axis.
    """
    rng = np.random.default_rng(seed)
    settings, reps, per_anchor = _anchor_centroids(runs)
    if baseline not in settings:
        raise SystemExit(
            f"[decoding-centroid] no '{baseline}' setting among {settings}.")
    print(f"[decoding-centroid] {len(per_anchor)} common anchors, "
          f"{len(settings)} settings, {len(reps)} reps")

    names, actual, nulls, colours = [], [], [], []
    for label, prefix, colour in _DECODING_GROUPS:
        variants = [s for s in settings if isinstance(s, str) and s.startswith(prefix)]
        if not variants:
            print(f"[decoding-centroid] no '{prefix}*' settings; skipping {label}")
            continue
        a_vals, n_vals = [], []
        for by_setting in per_anchor.values():
            if baseline not in by_setting:
                continue
            cb = by_setting[baseline]
            for v in variants:
                if v in by_setting:
                    a_vals.append(_centroid_cos(cb, by_setting[v]))
                    n_vals.append(_perm_null_centroid_cos(cb, by_setting[v]))
        if not a_vals:
            print(f"[decoding-centroid] no anchors for {label}; skipping")
            continue
        names.append(label); colours.append(colour)
        actual.append(np.asarray(a_vals)); nulls.append(np.asarray(n_vals))

    if not names:
        print("[decoding-centroid] no variant groups present; skipping plot")
        return None

    def _ci(x):
        obs = float(np.mean(x))
        boots = [float(rng.choice(x, size=len(x), replace=True).mean())
                 for _ in range(n_bootstrap)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return obs, float(lo), float(hi)

    means, los, his = map(list, zip(*[_ci(a) for a in actual]))
    null_means = [float(np.mean(n)) for n in nulls]
    null_medians = [float(np.median(n)) for n in nulls]

    fig, (ax_box, ax_forest) = plt.subplots(
        1, 2, figsize=(7.0, 3.3),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )

    # --- Box plot (left): actual centroid cosine; grey dash = noise floor. ---
    pos = np.arange(1, len(names) + 1)
    bp = ax_box.boxplot(actual, positions=pos, tick_labels=names,
                        patch_artist=True, widths=0.55, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="black",
                                       markeredgecolor="black", markersize=5),
                        medianprops=dict(color="black", linewidth=1.2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                        markeredgecolor="none",
                                        markerfacecolor="grey"))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.65); patch.set_edgecolor("black")
    ax_box.plot(pos, null_medians, "_", color="#333333", markersize=24,
                markeredgewidth=2.4, zorder=4, label="noise floor (no shift)")
    ax_box.axhline(1.0, color="black", linewidth=0.6, linestyle="-", alpha=0.4)
    ax_box.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax_box.set_ylabel("Centroid cosine (baseline ↔ setting)")
    ax_box.grid(axis="y", linestyle=":", alpha=0.5)

    # --- Forest plot (right): mean centroid cosine ± 95% CI + noise floor. ---
    y = np.arange(len(names))[::-1]
    err_low = [m - lo for m, lo in zip(means, los)]
    err_high = [hi - m for m, hi in zip(means, his)]
    ax_forest.errorbar(means, y, xerr=[err_low, err_high], fmt="none",
                       color="black", capsize=5, capthick=1.2, linewidth=1.2)
    for i, (m, c) in enumerate(zip(means, colours)):
        ax_forest.plot(m, y[i], "o", color=c, markersize=7,
                       markeredgecolor="black", markeredgewidth=0.8, zorder=3)
    ax_forest.plot(null_means, y, "|", color="#333333", markersize=14,
                   markeredgewidth=2.2, zorder=4, label="noise floor")
    ax_forest.set_yticks(y); ax_forest.set_yticklabels(names)
    ax_forest.set_xlabel("Centroid cosine")
    ax_forest.grid(axis="x", linestyle=":", alpha=0.5)
    for i, m in enumerate(means):
        ax_forest.text(m, y[i] + 0.18, f"{m:.3f}",
                       ha="center", va="bottom", fontsize=10)
    ax_forest.set_ylim(y.min() - 0.55, y.max() + 0.55)

    _panel_y = -0.30
    ax_box.text(0.5, _panel_y, "(a) Distribution per anchor",
                transform=ax_box.transAxes, ha="center", va="top", fontsize=11)
    ax_forest.text(0.5, _panel_y, "(b) Mean ± 95 % CI",
                   transform=ax_forest.transAxes, ha="center", va="top", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")

    # Per-group summary: shift_below_floor = noise_floor − actual (>0 ⇒ real
    # content shift beyond that setting's stochasticity).
    return pd.DataFrame({
        "group": names,
        "n_obs": [len(a) for a in actual],
        "mean_cos": means, "ci_lo": los, "ci_hi": his,
        "noise_floor_cos": null_means,
        "shift_below_floor": [nm - m for nm, m in zip(null_means, means)],
    })


def _order_decoding_settings(settings):
    """Order decoding settings baseline-first, then knob-grouped, and assign a
    colour per setting (one hue per knob). Returns (ordered_list, {setting:hex})."""
    # One hue per knob, distinguished light (lo) → dark (hi) so the lo/hi
    # settings no longer share an identical colour. 'both' uses the brown used
    # elsewhere in this module (the Neighbour axis) instead of sea green.
    knobs = [
        ("temp_", {"lo": "#86b8de", "hi": "#1f5c8a"}),   # blue:   light → dark
        ("topp_", {"lo": "#f0a868", "hi": "#a84e05"}),   # orange: light → dark
        ("both_", {"lo": "#c47a52", "hi": "#8d2c03"}),   # brown:  light → dark
    ]
    ordered, cmap = [], {}
    for s in settings:                       # baseline(s) first
        if s == "baseline":
            ordered.append(s); cmap[s] = "#7f7f7f"
    for prefix, shades in knobs:             # then lo→hi within each knob
        for suffix in ("lo", "hi"):
            name = f"{prefix}{suffix}"
            if name in settings:
                ordered.append(name); cmap[name] = shades[suffix]
    for s in settings:                       # anything unmatched, appended as-is
        if s not in cmap:
            ordered.append(s); cmap[s] = "#999999"
    return ordered, cmap


def comparison_decoding_diversity(runs: dict, out_path: str,
                                  n_bootstrap: int = 1000, seed: int = 0):
    """Box + forest of per-anchor within-setting DIVERSITY for the decoding axis.

    At each anchor (agent, round) a setting holds the prompt, persona and
    neighbour input fixed and varies only the LLM RNG across its reps, so the
    scatter of those reps is pure decoding stochasticity. Diversity at the
    anchor is ``1 − mean_pairwise_cosine`` of the setting's rep embeddings
    (higher ⇒ the same prompt produces more varied posts). One box per decoding
    setting; the dashed line marks the baseline (temp 0.7 / top_p 0.9) mean, so
    you read off directly whether raising a knob widens the output distribution.

    Contrast with `comparison_decoding_centroids`, which asks whether the MEAN
    output MOVES. This asks whether the SPREAD changes — the quantity decoding
    parameters are actually meant to control — so it needs no shared-noise floor.
    """
    rng = np.random.default_rng(seed)
    settings, reps, per_anchor = _anchor_centroids(runs)
    order, cmap = _order_decoding_settings(settings)

    per_setting = {s: [] for s in order}
    for by_setting in per_anchor.values():
        for s in order:
            grp = by_setting.get(s)
            if grp is not None and grp.shape[0] >= 2:
                per_setting[s].append(1.0 - mean_pairwise_cos(grp))
    order = [s for s in order if per_setting[s]]          # drop empty settings
    if not order:
        print("[decoding-diversity] no settings with >=2 reps; skipping plot")
        return None
    data = [np.asarray(per_setting[s]) for s in order]
    colours = [cmap[s] for s in order]
    print(f"[decoding-diversity] {len(per_anchor)} anchors, {len(reps)} reps; "
          f"settings={order}")

    def _ci(x):
        obs = float(np.mean(x))
        boots = [float(rng.choice(x, size=len(x), replace=True).mean())
                 for _ in range(n_bootstrap)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return obs, float(lo), float(hi)

    means, los, his = map(list, zip(*[_ci(d) for d in data]))
    base_mean = means[order.index("baseline")] if "baseline" in order else None

    fig, (ax_box, ax_forest) = plt.subplots(
        1, 2, figsize=(8.2, 3.7),
        gridspec_kw={"width_ratios": [1.7, 1.0]})

    # --- Box plot (left): per-anchor diversity distribution per setting. ---
    pos = np.arange(1, len(order) + 1)
    bp = ax_box.boxplot(data, positions=pos, tick_labels=order,
                        patch_artist=True, widths=0.6, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="black",
                                       markeredgecolor="black", markersize=5),
                        medianprops=dict(color="black", linewidth=1.2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.35,
                                        markeredgecolor="none",
                                        markerfacecolor="grey"))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.65); patch.set_edgecolor("black")
    if base_mean is not None:
        ax_box.axhline(base_mean, color="#333333", linestyle="--",
                       linewidth=1.1, alpha=0.85, label="baseline mean")
        ax_box.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_box.set_ylabel("Output diversity  (1 − within cosine)")
    ax_box.grid(axis="y", linestyle=":", alpha=0.5)
    ax_box.tick_params(axis="x", labelrotation=30)

    # --- Forest (right): mean diversity ± 95% CI, baseline line for reference. -
    y = np.arange(len(order))[::-1]
    err_low = [m - lo for m, lo in zip(means, los)]
    err_high = [hi - m for m, hi in zip(means, his)]
    ax_forest.errorbar(means, y, xerr=[err_low, err_high], fmt="none",
                       color="black", capsize=5, capthick=1.2, linewidth=1.2)
    for i, (m, c) in enumerate(zip(means, colours)):
        ax_forest.plot(m, y[i], "o", color=c, markersize=7,
                       markeredgecolor="black", markeredgewidth=0.8, zorder=3)
    if base_mean is not None:
        ax_forest.axvline(base_mean, color="#333333", linestyle="--",
                          linewidth=1.0, alpha=0.85)
    ax_forest.set_yticks(y); ax_forest.set_yticklabels(order)
    ax_forest.set_xlabel("Mean diversity")
    ax_forest.grid(axis="x", linestyle=":", alpha=0.5)
    for i, m in enumerate(means):
        ax_forest.text(m, y[i] + 0.18, f"{m:.3f}", ha="center", va="bottom",
                       fontsize=9)
    ax_forest.set_ylim(y.min() - 0.55, y.max() + 0.55)

    _panel_y = -0.42
    ax_box.text(0.5, _panel_y, "(a) Distribution per anchor",
                transform=ax_box.transAxes, ha="center", va="top", fontsize=11)
    ax_forest.text(0.5, _panel_y, "(b) Mean ± 95 % CI",
                   transform=ax_forest.transAxes, ha="center", va="top", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")

    return pd.DataFrame({
        "setting": order,
        "n_anchors": [len(d) for d in data],
        "mean_diversity": means, "ci_lo": los, "ci_hi": his,
        "delta_vs_baseline": [m - base_mean if base_mean is not None else np.nan
                              for m in means],
    })


def _phq9_band_probe(runs: dict, n_splits: int = 5):
    """Agent-grouped CV probe accuracy for the PHQ-9 band, per decoding setting.

    Returns ``{setting: ndarray}`` holding ONE balanced-accuracy value per rep
    (length = n_reps). Within a rep, a logistic probe is trained under GroupKFold
    by ``agent_id`` and its out-of-fold predictions are pooled into a single CV
    score, so every post is tested exactly once by a probe that never saw its
    agent. The rep — an independent non-deterministic generation — is the unit of
    replication: the 5 folds only partition one rep's data (not independent), so
    we collapse them per rep and let the caller take mean/SD ACROSS reps.
    GroupKFold blocks persona-identity leakage; ``class_weight='balanced'`` +
    balanced-accuracy counter the band imbalance. Chance = 0.20 (5 bands)."""
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score

    settings = sorted({s for (s, _) in runs})
    reps = sorted({r for (_, r) in runs})
    out = {}
    for s in settings:
        rep_accs = []
        for r in reps:
            if (s, r) not in runs:
                continue
            d = runs[(s, r)]
            X = d["embeddings"].astype(np.float64)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            y = np.array([phq9_to_band(int(p)) for p in d["phq9"]])
            g = np.asarray(d["agent_ids"])
            k = min(n_splits, len(np.unique(g)))
            if k < 2 or len(np.unique(y)) < 2:
                continue
            # Pool out-of-fold predictions into a single per-rep CV score.
            y_pred = np.empty_like(y)
            tested = np.zeros(len(y), dtype=bool)
            for tr, te in GroupKFold(n_splits=k).split(X, y, g):
                if len(np.unique(y[tr])) < 2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = LogisticRegression(max_iter=2000,
                                             class_weight="balanced")
                    clf.fit(X[tr], y[tr])
                    y_pred[te] = clf.predict(X[te])
                    tested[te] = True
            if tested.any():
                rep_accs.append(
                    balanced_accuracy_score(y[tested], y_pred[tested]))
        out[s] = np.asarray(rep_accs)
    return out


def comparison_decoding_phq9_probe(runs: dict, out_path: str, n_splits: int = 5):
    """Bar plot: PHQ-9 band DISTINGUISHABILITY per decoding setting.

    For each setting an agent-grouped CV logistic probe predicts the 5-class
    PHQ-9 band from post embeddings (see `_phq9_band_probe`), giving one CV score
    per rep. Bar height = mean balanced accuracy ACROSS the reps; the thin
    whisker = ±1 SD across reps (rep = an independent generation; drop `yerr=`
    for a plain bar). The y-axis starts at chance (0.20) — balanced accuracy of
    an uninformative classifier is 0.20 by construction, so bar height above the
    floor is the actual severity signal. The dotted line marks the baseline (temp
    0.7 / top_p 0.9); a setting below it blurs the severity classes — the knobs
    that raise diversity cost PHQ-9 signal.
    """
    acc = _phq9_band_probe(runs, n_splits=n_splits)
    order, cmap = _order_decoding_settings(sorted(acc.keys()))
    order = [s for s in order if len(acc.get(s, [])) > 0]
    if not order:
        print("[decoding-probe] no settings with probe estimates; skipping plot")
        return None
    colours = [cmap[s] for s in order]
    print(f"[decoding-probe] settings={order}; "
          f"reps/setting={[len(acc[s]) for s in order]}")

    means = [float(np.mean(acc[s])) for s in order]
    sds = [float(np.std(acc[s], ddof=1)) if len(acc[s]) > 1 else 0.0 for s in order]
    base_mean = means[order.index("baseline")] if "baseline" in order else None
    chance = 0.20

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(order))
    ax.bar(x, means, width=0.7, color=colours, edgecolor="black", linewidth=0.6,
           yerr=sds, capsize=4, error_kw=dict(elinewidth=1.0, alpha=0.6))
    ax.axhline(chance, color="#b22222", linestyle="--", linewidth=1.1, alpha=0.85,
               label=f"chance ({chance:.2f})")
    if base_mean is not None:
        ax.axhline(base_mean, color="#333333", linestyle=":", linewidth=1.3,
                   alpha=0.85, label="baseline")
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("PHQ-9 band probe — balanced accuracy")
    ax.set_ylim(chance - 0.02, max(m + s for m, s in zip(means, sds)) + 0.02)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")

    return pd.DataFrame({
        "setting": order,
        "n_reps": [len(acc[s]) for s in order],
        "mean_bal_acc": means, "sd_across_reps": sds,
        "delta_vs_baseline": [m - base_mean if base_mean is not None else np.nan
                              for m in means],
    })


def _phq9_band_linearity(runs: dict):
    """Per-setting cosine between ADJACENT PHQ-9 class centroids (severity order).

    For each (setting, rep) build one class centroid per PHQ-9 band — the mean of
    that band's L2-normalised post embeddings, renormalised — then take the
    cosine between every pair of NEIGHBOURING bands along the severity ladder
    (Minimal→Mild→…→Severe). This walks the ordinal axis one rung at a time: a
    high, flat curve means consecutive severities sit close and evenly spaced (a
    smooth/linear severity encoding), while a dip marks a sharp class boundary
    where the generated content jumps. The per-rep curves are averaged over the
    reps (the rep — an independent non-deterministic generation — is the unit of
    replication), with the across-rep SD as the spread.

    Returns ``(pair_labels, {setting: (mean_vec, sd_vec)})``; each vector has one
    entry per adjacent band pair (NaN where a band is empty in a rep). Cosine is
    only meaningful on SBERT embeddings (``emb_name='embeddings_sbert.npz'``);
    MentalBERT is anisotropic so its cosines collapse to a constant."""
    settings = sorted({s for (s, _) in runs})
    reps = sorted({r for (_, r) in runs})
    bands = BAND_LABELS
    pair_labels = [f"{bands[i]}→{bands[i + 1]}" for i in range(len(bands) - 1)]

    out = {}
    for s in settings:
        rep_curves = []
        for r in reps:
            if (s, r) not in runs:
                continue
            d = runs[(s, r)]
            X = d["embeddings"].astype(np.float64)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            y = np.array([phq9_to_band(int(p)) for p in d["phq9"]])
            cents = {}
            for b in bands:
                m = y == b
                if m.sum() >= 1:
                    c = X[m].mean(axis=0)
                    cents[b] = c / (np.linalg.norm(c) + 1e-12)
            curve = [float(cents[bands[i]] @ cents[bands[i + 1]])
                     if bands[i] in cents and bands[i + 1] in cents else np.nan
                     for i in range(len(bands) - 1)]
            rep_curves.append(curve)
        if rep_curves:
            arr = np.asarray(rep_curves)                      # (n_reps, n_pairs)
            mean_vec = np.nanmean(arr, axis=0)
            sd_vec = (np.nanstd(arr, axis=0, ddof=1)
                      if arr.shape[0] > 1 else np.zeros(arr.shape[1]))
            out[s] = (mean_vec, sd_vec)
    return pair_labels, out


def _decoding_diversity_stats(runs: dict, n_bootstrap: int = 1000, seed: int = 0):
    """Per-setting within-setting diversity stats for the forest panel — pure
    numbers, no plotting. Mirrors `comparison_decoding_diversity`'s computation
    so the combined figure can redraw its forest without rebuilding the box plot.

    Returns ``(order, colours, means, los, his, base_mean, n_anchors)`` or
    ``None`` when no setting has ≥2 reps per anchor."""
    rng = np.random.default_rng(seed)
    settings, reps, per_anchor = _anchor_centroids(runs)
    order, cmap = _order_decoding_settings(settings)
    per_setting = {s: [] for s in order}
    for by_setting in per_anchor.values():
        for s in order:
            grp = by_setting.get(s)
            if grp is not None and grp.shape[0] >= 2:
                per_setting[s].append(1.0 - mean_pairwise_cos(grp))
    order = [s for s in order if per_setting[s]]
    if not order:
        return None
    colours = [cmap[s] for s in order]
    data = [np.asarray(per_setting[s]) for s in order]

    def _ci(x):
        obs = float(np.mean(x))
        boots = [float(rng.choice(x, size=len(x), replace=True).mean())
                 for _ in range(n_bootstrap)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return obs, float(lo), float(hi)

    means, los, his = map(list, zip(*[_ci(d) for d in data]))
    base_mean = means[order.index("baseline")] if "baseline" in order else None
    return order, colours, means, los, his, base_mean, [len(d) for d in data]


# (temperature, top_p) actually swept per decoding setting (see sa_decoding_run.sh).
_DECODING_PARAMS = {
    "baseline": (0.7, 0.9),
    "temp_lo":  (0.4, 0.9),
    "temp_hi":  (1.0, 0.9),
    "topp_lo":  (0.7, 0.8),
    "topp_hi":  (0.7, 1.0),
    "both_lo":  (0.4, 0.8),
    "both_hi":  (1.0, 1.0),
}


def _decoding_tuple_label(s) -> str:
    """Display label '(temp, top_p)' for a decoding setting; falls back to the
    raw name when the setting isn't in the swept-parameter table."""
    tp = _DECODING_PARAMS.get(s)
    return f"({tp[0]}, {tp[1]})" if tp else str(s)


def comparison_decoding_linearity(runs: dict, out_path: str,
                                  n_bootstrap: int = 1000, seed: int = 0):
    """Two-panel decoding figure: PHQ-9 severity LINEARITY (left) + within-setting
    DIVERSITY forest (right). Both read SBERT embeddings (cosine-appropriate).

    (a) Linearity — for each setting, the cosine between adjacent PHQ-9 class
        centroids walked along the severity ladder (see `_phq9_band_linearity`).
        One line per decoding setting (baseline grey, knobs coloured); a high,
        flat curve = a smooth ordinal severity encoding, a dip = a sharp class
        boundary. Lets you read off how temp / top_p reshape the severity geometry.
    (b) Diversity — per-setting mean (1 − within-cosine) ± 95 % CI, identical to
        `comparison_decoding_diversity`'s forest panel — so the figure pairs
        "does decoding blur the severity ladder?" with "does it widen the spread?".
    """
    pair_labels, lin = _phq9_band_linearity(runs)
    order_lin, cmap = _order_decoding_settings(sorted(lin.keys()))
    order_lin = [s for s in order_lin if s in lin]
    if not order_lin:
        print("[decoding-linearity] no settings with band centroids; skipping plot")
        return None
    div = _decoding_diversity_stats(runs, n_bootstrap=n_bootstrap, seed=seed)
    print(f"[decoding-linearity] settings={order_lin}; "
          f"severity steps={pair_labels}")

    fig, (ax_lin, ax_for) = plt.subplots(
        1, 2, figsize=(7.0, 3.6), gridspec_kw={"width_ratios": [1.55, 1.0]})

    # --- (a) Linearity lines: cosine between adjacent class centroids. -------
    x = np.arange(len(pair_labels))
    for s in order_lin:
        mean_vec, _sd_vec = lin[s]
        ax_lin.plot(x, mean_vec, "o-", color=cmap[s], label=_decoding_tuple_label(s),
                    linewidth=2.0 if s == "baseline" else 1.6,
                    markersize=6, markeredgecolor="black", markeredgewidth=0.5,
                    zorder=4 if s == "baseline" else 3)
    ax_lin.set_xticks(x)
    disp_labels = [lab.replace("Moderate", "Mod.").replace("Severe", "Sev.")
                   for lab in pair_labels]
    ax_lin.set_xticklabels(disp_labels, rotation=20, ha="right", fontsize=9)
    ax_lin.set_ylabel("Cosine similarity")
    ax_lin.grid(axis="y", linestyle=":", alpha=0.5)
    ax_lin.legend(fontsize=8, ncol=2, framealpha=0.9, loc="lower right",
                  handlelength=1.6, handletextpad=0.6, columnspacing=1.3,
                  labelspacing=0.4, borderpad=0.5)

    # --- (b) Diversity forest: mean ± 95 % CI, baseline reference line. ------
    if div is not None:
        order, colours, means, los, his, base_mean, _n = div
        yy = np.arange(len(order))[::-1]
        err_low = [m - lo for m, lo in zip(means, los)]
        err_high = [hi - m for m, hi in zip(means, his)]
        ax_for.errorbar(means, yy, xerr=[err_low, err_high], fmt="none",
                        color="black", capsize=5, capthick=1.2, linewidth=1.2)
        for i, (m, c) in enumerate(zip(means, colours)):
            ax_for.plot(m, yy[i], "o", color=c, markersize=7,
                        markeredgecolor="black", markeredgewidth=0.8, zorder=3)
        if base_mean is not None:
            ax_for.axvline(base_mean, color="#333333", linestyle="--",
                           linewidth=1.0, alpha=0.85)
        ax_for.set_yticks(yy)
        ax_for.set_yticklabels([_decoding_tuple_label(s) for s in order])
        ax_for.set_xlabel("Mean diversity")
        ax_for.grid(axis="x", linestyle=":", alpha=0.5)
        ax_for.set_ylim(yy.min() - 0.55, yy.max() + 0.55)

    _py = -0.34
    ax_lin.text(0.5, _py, "(a) PHQ-9 severity linearity",
                transform=ax_lin.transAxes, ha="center", va="top", fontsize=11)
    ax_for.text(0.5, _py, "(b) Within-setting diversity",
                transform=ax_for.transAxes, ha="center", va="top", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")

    rows = []
    for s in order_lin:
        mean_vec, sd_vec = lin[s]
        for lab, m, sd in zip(pair_labels, mean_vec, sd_vec):
            rows.append({"setting": s, "severity_step": lab,
                         "mean_adjacent_cos": float(m),
                         "sd_across_reps": float(sd)})
    return pd.DataFrame(rows)


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
# Combined: PHQ-9 adjacent-band severity ladder + conditioning heatmap
# =====================================================================

def phq9_adjacent_band_ladder(root: str, emb_name: str = "embeddings_sbert.npz",
                              subdir: str = "phq9") -> pd.DataFrame | None:
    """The conditioning matrix's SUPER-DIAGONAL: same-persona cosine between each
    pair of CONSECUTIVE PHQ-9 bands, with the BETWEEN-RUN spread.

    The conditioning runs vary ONLY PHQ-9 (persona, neighbour and decoding held
    fixed), so the one-off-diagonal cell (band b, band b+1) is exactly "how much a
    persona's post moves when its PHQ-9 is bumped a single rung". Each band has
    several reps — independent regenerations that differ only in the LLM seed — and
    the rep is the unit of replication used throughout this module (cf.
    `_phq9_band_probe`, `_phq9_band_linearity`). So for each rep we pair that rep of
    band b with the same rep of band b+1, average the per-anchor cosine over the 600
    personas to get one cell estimate, and report the MEAN over reps with the SD
    ACROSS reps (run-to-run generation noise) — NOT the across-persona scatter,
    which is ~0.13 and reflects how much personas differ, not estimate uncertainty.

    A RISE toward the severe end means consecutive top bands barely differ: the scale
    saturates / merges at high severity, i.e. severe content dominates. SBERT-only
    (MentalBERT is anisotropic; see `_phq9_band_linearity`).

    ``subdir`` selects which conditioning tree to read: ``"phq9"`` (default, the
    iter_10 optimised prompt, 3 reps) or ``"phq9_minimal_prompt"`` (the minimal /
    un-optimised prompt baseline, 1 rep — so its SD is 0 and the line carries no
    error bars).

    Returns one row per severity step (BAND_LABELS order), or None when <2 bands
    carry the requested encoder (only rep_1 ships SBERT by default — run
    ``sa_embed --sbert`` on the extra reps).
    """
    paths = sorted(glob.glob(os.path.join(root, subdir, "*", "rep_*", emb_name)))
    if not paths:
        print(f"[phq9-ladder] no {emb_name} under {root}/{subdir}/*/rep_*")
        return None

    by_band: dict[str, list] = defaultdict(list)
    for p in paths:
        d = np.load(p, allow_pickle=True)
        idx = {(int(a), int(r)): i for i, (a, r) in
               enumerate(zip(d["agent_ids"], d["rounds"]))}
        dom = pd.Series([phq9_to_band(int(s)) for s in d["phq9"]]).mode().iloc[0]
        by_band[dom].append((idx, d["embeddings"]))   # rep-sorted (glob is sorted)
    bands = [b for b in BAND_LABELS if b in by_band]
    if len(bands) < 2:
        print(f"[phq9-ladder:{subdir}] <2 bands with {emb_name}; skipping.")
        return None

    # Anchors present in every run (same personas across all bands+reps).
    common = None
    for runs in by_band.values():
        for idx, _ in runs:
            s = set(idx)
            common = s if common is None else common & s
    common = sorted(common or [])
    n_rep = min(len(by_band[b]) for b in bands)        # rep-matched across bands
    print(f"[phq9-ladder:{subdir}] {sum(len(v) for v in by_band.values())} runs, "
          f"{len(common)} common anchors over {len(bands)} bands, {n_rep} reps")

    def _unit_rows(idx, emb):
        X = np.array([emb[idx[a]] for a in common], dtype=np.float64)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    rows = []
    for b0, b1 in zip(bands[:-1], bands[1:]):
        # One cell estimate per rep: pair rep r of b0 with rep r of b1, mean over
        # personas. SD over these rep estimates = run-to-run noise.
        cells = []
        for r in range(n_rep):
            A = _unit_rows(*by_band[b0][r])
            B = _unit_rows(*by_band[b1][r])
            cells.append(float((A * B).sum(axis=1).mean()))
        cells = np.asarray(cells)
        std = float(cells.std(ddof=1)) if cells.size > 1 else 0.0
        rows.append({"step": f"{b0} → {b1}", "from_band": b0, "to_band": b1,
                     "cos": float(cells.mean()), "std": std,
                     "sem": std / np.sqrt(cells.size), "n_reps": int(cells.size),
                     "n_anchors": len(common)})
    return pd.DataFrame(rows) if rows else None


def plot_agent_phq9_combined(ladder: pd.DataFrame, phq9_matrix: pd.DataFrame,
                             out_path: str,
                             color_cross: str = "#d96907",
                             cmap: str = "Oranges",
                             err: str = "std",
                             baseline: pd.DataFrame | None = None,
                             main_label: str = "Opt. prompt",
                             baseline_label: str = "Min. prompt",
                             baseline_color: str = "#6c6c6c"):
    """Side-by-side severity figure from the PHQ-9 conditioning runs:

    (a) the conditioning matrix's super-diagonal — same-persona cosine between
        consecutive PHQ-9 bands (`phq9_adjacent_band_ladder`) — with error bars
        (``err``: "std" = run-to-run SD across reps, or "sem"), mean printed at each
        step. A rise toward the severe end = consecutive high bands become
        near-indistinguishable: the severity classes saturate / merge at the top.
    (b) the PHQ-9 conditioning 5×5 cosine heatmap; panel (a) is its super-diagonal.

    ``baseline`` (optional) is a second adjacent-band ladder — the minimal /
    un-optimised prompt (`phq9_adjacent_band_ladder(..., subdir="phq9_minimal_prompt")`,
    1 rep so no SD) — drawn as a dashed reference line on panel (a). It is aligned
    to the main ladder's steps by (from_band, to_band), so the two curves stay
    rung-matched even if a band is missing on one side.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    _ABBR = {"Minimal": "Minimal", "Mild": "Mild", "Moderate": "Mod.",
             "Mod. Severe": "Mod. Sev", "Severe": "Sev"}

    fig = plt.figure(figsize=(6.8, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.35)
    ax_bars = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])

    # ===== Left panel: conditioning super-diagonal (adjacent-band cosine) =====
    x = np.arange(len(ladder))
    y = ladder["cos"].values
    yerr = ladder[err].values
    ax_bars.errorbar(x, y, yerr=yerr, fmt="o-", color=color_cross,
                     capsize=3, linewidth=1.8, markersize=6,
                     markeredgecolor="black", markeredgewidth=0.5, zorder=3,
                     label=main_label)

    # Track the spans both curves need to fit inside the y-limits.
    los = [float((y - yerr).min())]
    his = [float((y + yerr).max())]

    # Optional baseline (minimal prompt, 1 rep -> no SD): a dashed reference line.
    # Align to the main ladder's rungs by (from_band, to_band) so the two curves
    # stay step-matched even if a band is missing on one side.
    if baseline is not None and not baseline.empty:
        bmap = {(r.from_band, r.to_band): float(r.cos)
                for r in baseline.itertuples(index=False)}
        yb = np.array([bmap.get((a, b), np.nan)
                       for a, b in zip(ladder["from_band"], ladder["to_band"])])
        if np.isfinite(yb).any():
            ax_bars.plot(x, yb, ls="--", marker="s", color=baseline_color,
                         linewidth=1.5, markersize=5, markeredgecolor="black",
                         markeredgewidth=0.5, zorder=2, label=baseline_label)
            los.append(float(np.nanmin(yb)))
            his.append(float(np.nanmax(yb)))
        ax_bars.legend(loc="best", fontsize=7.5, framealpha=0.9)

    labels = [f"{_ABBR[a]} → {_ABBR[b]}"
              for a, b in zip(ladder["from_band"], ladder["to_band"])]
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax_bars.tick_params(axis="y", labelsize=8)
    ax_bars.set_ylabel("Cosine similarity", fontsize=9)
    lo = min(los)
    hi = max(his)
    span = max(hi - lo, 1e-3)
    ax_bars.set_ylim(lo - 0.08 * span, hi + 0.10 * span)
    ax_bars.set_xlim(-0.45, len(x) - 0.55)
    ax_bars.grid(axis="y", linestyle=":", alpha=0.5)

    # ===== Right panel: PHQ-9 conditioning heatmap =====
    mat = phq9_matrix.values
    n = mat.shape[0]
    band_labels_ax = list(phq9_matrix.index)
    # Span the whole matrix (diagonal now holds within-band cosine, not 1.0).
    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))
    sns.heatmap(
        mat, ax=ax_heat,
        xticklabels=band_labels_ax, yticklabels=band_labels_ax,
        vmin=vmin - 0.01, vmax=vmax + 0.01,
        annot=True, fmt=".3f", annot_kws={"fontsize": 7},
        cmap=cmap, linewidths=0.4, linecolor="white",
        cbar=False,
        square=True,
    )
    ax_heat.tick_params(axis="x", rotation=30, labelsize=7.5)
    ax_heat.tick_params(axis="y", rotation=0, labelsize=7.5)
    # Dock the colorbar directly to the heatmap. pad controls the gap.
    divider = make_axes_locatable(ax_heat)
    cbar_ax = divider.append_axes("right", size="4%", pad=0.08)
    sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin - 0.01, vmax=vmax + 0.01),
        cmap=plt.get_cmap(cmap),
    )
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("cosine similarity", fontsize=8)
    cbar_ax.tick_params(labelsize=7)
    # Panel labels in axis coords, dropped low enough to clear the rotated ticks.
    ax_bars.text(0.5, -0.42, "(a) Adjacent-band cosine",
                 transform=ax_bars.transAxes, ha="center", va="top", fontsize=10)
    ax_heat.text(0.5, -0.42, "(b) PHQ-9 conditioning",
                 transform=ax_heat.transAxes, ha="center", va="top", fontsize=10)

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
    parser.add_argument("--out-dir", default=None,
                        help="Where to write CSV + PNG outputs "
                             "(default: <root>/plots, or <root>/plots_sbert for --emb-name embeddings_sbert.npz).")
    parser.add_argument("--emb-name", default="embeddings.npz",
                        help="Which encoder's .npz to read: embeddings.npz (MentalBERT, default) "
                             "or embeddings_sbert.npz (SBERT, content/topic axis).")
    args = parser.parse_args()

    # Keep SBERT (content) outputs from clobbering the MentalBERT outputs.
    if args.out_dir is None:
        suffix = "" if args.emb_name == "embeddings.npz" else \
            "_" + args.emb_name.replace("embeddings_", "").replace(".npz", "")
        args.out_dir = os.path.join(args.root, "plots" + suffix)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[sa] reading {args.emb_name}  ->  out_dir = {args.out_dir}")

    axis_dfs: dict[str, pd.DataFrame] = {}
    for axis_dir, axis_label, cosine_fn in [
        ("neighbor", "Neighbour", neighbor_cosines),
        ("agent",    "Agent",     agent_cosines),
        ("joint",    "Joint",     neighbor_cosines),  # same paired logic; same anchors
        ("decoding", "Decoding",  neighbor_cosines),  # temp/top_p; same anchors (agents fixed)
    ]:
        full = os.path.join(args.root, axis_dir)
        if not os.path.isdir(full):
            print(f"[skip] {full} not present")
            continue
        print(f"\n=== {axis_label} axis ===")

        runs = load_axis_runs(args.root, axis_dir, emb_name=args.emb_name)
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

        axis_dfs[axis_label] = df

    # Cross-axis comparison: combined box (distribution) + forest (mean ± CI).
    # Decoding is excluded here — it gets its own per-setting figure below
    # (decoding_settings_comparison.png); this plot stays the structural axes.
    cross_axis_dfs = {k: v for k, v in axis_dfs.items() if k != "Decoding"}

    # PHQ-9 conditioning as a 4th box: bands play the role of "setting", so
    # neighbor_cosines gives within(=same band, diff rep) vs cross(=diff band).
    # Needs >=2 reps/band for the active encoder — true for SBERT, skipped for
    # the 1-rep MentalBERT cosine (whose anisotropy makes the metric meaningless
    # anyway; the MentalBERT story is told by sa_phq9's MLP figure instead).
    phq9_runs = load_phq9_runs(args.root, emb_name=args.emb_name)
    reps_per_band: dict = {}
    for (band, rep) in phq9_runs:
        reps_per_band.setdefault(band, set()).add(rep)
    if phq9_runs and min((len(v) for v in reps_per_band.values()), default=0) >= 2:
        print("\n=== PHQ-9 axis (for cross-axis comparison) ===")
        phq9_df = neighbor_cosines(phq9_runs)
        phq9_df.to_csv(os.path.join(args.out_dir, "phq9_cosines.csv"), index=False)
        cross_axis_dfs["PHQ-9"] = phq9_df
    else:
        print(f"[phq9] <2 reps/band with {args.emb_name}; PHQ-9 box omitted from "
              "the cosine comparison (expected for MentalBERT — use sa_phq9).")

    if len(cross_axis_dfs) >= 2:
        print("\n=== Cross-axis comparison ===")
        comparison_combined(cross_axis_dfs,
                            os.path.join(args.out_dir, "axes_comparison.png"))

    # Decoding axis: per-setting within-setting DIVERSITY (1 − within-cosine of
    # each setting's reps). Directly shows whether raising temperature / top_p
    # widens the output distribution, with temp 0.7 / top_p 0.9 (baseline) as the
    # reference line. (comparison_decoding_centroids — centroid shift, i.e. does
    # the MEAN move — is kept in the module as the complementary view.)
    if os.path.isdir(os.path.join(args.root, "decoding")):
        print("\n=== Decoding settings (within-setting diversity) ===")
        dec_runs = load_axis_runs(args.root, "decoding", emb_name=args.emb_name)
        if dec_runs:
            dec_summary = comparison_decoding_diversity(
                dec_runs,
                os.path.join(args.out_dir, "decoding_settings_comparison.png"))
            if dec_summary is not None:
                dec_summary.to_csv(
                    os.path.join(args.out_dir, "decoding_settings_summary.csv"),
                    index=False)
                print(dec_summary.round(4).to_string(index=False))

            # PHQ-9 distinguishability per setting: does the decoding choice blur
            # the severity classes? (agent-grouped CV probe; chance = 0.20)
            print("\n=== Decoding settings (PHQ-9 distinguishability) ===")
            probe_summary = comparison_decoding_phq9_probe(
                dec_runs,
                os.path.join(args.out_dir, "decoding_phq9_separability.png"))
            if probe_summary is not None:
                probe_summary.to_csv(
                    os.path.join(args.out_dir, "decoding_phq9_separability.csv"),
                    index=False)
                print(probe_summary.round(4).to_string(index=False))

            # PHQ-9 severity LINEARITY (cosine between adjacent class centroids)
            # paired with the diversity forest. Cosine-based, so SBERT only.
            if args.emb_name == "embeddings_sbert.npz":
                print("\n=== Decoding settings (PHQ-9 severity linearity) ===")
                lin_summary = comparison_decoding_linearity(
                    dec_runs,
                    os.path.join(args.out_dir, "decoding_phq9_linearity.png"))
                if lin_summary is not None:
                    lin_summary.to_csv(
                        os.path.join(args.out_dir, "decoding_phq9_linearity.csv"),
                        index=False)
                    print(lin_summary.round(4).to_string(index=False))

    # PHQ-9 conditioning: 5×5 cosine matrix → line plot of cosine vs band-distance
    # (+ the agent_phq9_combined figure below).
    # Diagonal = per-band LLM-noise floor, borrowed from the agent axis's
    # within-setting cosine (same model + baseline decoding; the PHQ-9 runs have
    # no repeats of their own). Without it the diagonal falls back to within-band.
    print("\n=== PHQ-9 conditioning ===")
    diag_floor = None
    if "Agent" in axis_dfs:
        w = axis_dfs["Agent"]
        diag_floor = (w[w.pair_type == "within"].groupby("band").cosine.mean()
                      .to_dict())
    phq9_matrix = phq9_conditioning_matrix(args.root, args.out_dir,
                                           emb_name=args.emb_name,
                                           diag_floor=diag_floor)
    if phq9_matrix is not None:
        phq9_distance_lineplot(phq9_matrix,
                               os.path.join(args.out_dir, "phq9_distance_line.png"))
        # Combined: (a) per-band within-rep floor vs cross-persona cosine +
        # (b) the conditioning heatmap. The left panel needs >=2 reps/band with the
        # active encoder; on SBERT run `sa_embed --sbert` on the extra reps first
        # (only rep_1 ships SBERT by default).
        ladder = phq9_adjacent_band_ladder(args.root, emb_name=args.emb_name)
        if ladder is not None and not ladder.empty:
            ladder.to_csv(
                os.path.join(args.out_dir, "phq9_adjacent_band_ladder.csv"),
                index=False)
            print("  adjacent-band same-persona cosine (conditioning super-diagonal):")
            print(ladder.round(4).to_string(index=False))

            # Minimal-prompt baseline (un-optimised prompt, 1 rep -> no SD): the
            # same adjacent-band ladder over data/sensitivity/phq9_minimal_prompt,
            # drawn as a dashed reference line on the left panel. Skipped (with a
            # note) if that tree wasn't embedded with the active encoder.
            baseline = phq9_adjacent_band_ladder(
                args.root, emb_name=args.emb_name, subdir="phq9_minimal_prompt")
            if baseline is not None and not baseline.empty:
                baseline.to_csv(
                    os.path.join(args.out_dir,
                                 "phq9_adjacent_band_ladder_minimal.csv"),
                    index=False)
                print("  adjacent-band cosine (minimal-prompt baseline):")
                print(baseline.round(4).to_string(index=False))
            else:
                print(f"  [baseline] no {args.emb_name} under "
                      f"{args.root}/phq9_minimal_prompt/*/rep_* — left panel "
                      "drawn without the minimal-prompt line. Embed it first, e.g. "
                      f"`sa_embed --root {args.root}/phq9_minimal_prompt"
                      f"{' --sbert' if args.emb_name != 'embeddings.npz' else ''}`.")

            plot_agent_phq9_combined(
                ladder, phq9_matrix,
                os.path.join(args.out_dir, "agent_phq9_combined.png"),
                baseline=baseline,
            )
        else:
            print(f"  [combined] skipped — need >=2 bands with {args.emb_name} "
                  "(run `sa_embed --sbert` on the extra reps).")

    print(f"\n[done] CSVs + plots under {args.out_dir}/")


if __name__ == "__main__":
    import sys
    if "--prompt-reps" in sys.argv[1:]:
        prompt_reps_main([a for a in sys.argv[1:] if a != "--prompt-reps"])
    else:
        main()
