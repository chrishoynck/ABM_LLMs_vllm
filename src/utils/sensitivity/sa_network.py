"""
Network topology Sensitivity Analysis: Sobol SA over 5 SDA parameters.

Parameters varied
-----------------
    alpha         [0.5, 5.0]   – connection-prob decay sharpness
    n_clusters    [2, 10]      – number of Gaussian clusters (integer)
    latent_weight [0.5, 20.0]  – scaling of latent dims
    dim           [2, 6]       – total dims, incl. age + phq9 slots (integer)
    age_weight    [0.5, 5.0]   – scaling of age dim

Metrics computed on the *initial* undirected graph (no LLM involved)
----------------------------------------------------------------------
    C            – average clustering coefficient
    age_assort   – numeric assortativity by age
    phq9_assort  – numeric assortativity by PHQ-9
    gamma        – power-law exponent of degree distribution
    ks           – KS goodness-of-fit to power law (lower = better fit)

Reference targets (Twitter / online social networks, McPherson et al.)
----------------------------------------------------------------------
    C          : 0.10 – 0.20   (ER baseline ≈ k/N ≈ 0.03)
    age_assort : 0.25           (point target)
    phq9_assort: 0.03           (point target)
    gamma      : 2.0 – 3.0     (observed only)
    ks         : < 0.10        (observed only)

Usage – CLI
-----------
    PYTHONPATH=src python -m utils.sensitivity.sa_network \\
        --well-being data/confidential/phq9.sav \\
        --n-sobol 512 --n-jobs -1 \\
        --out-dir data/sensitivity/network

Usage – notebook
----------------
    import sys; sys.path.insert(0, "src")
    import utils.tools.load_personas as lp
    from utils.sensitivity.sa_network import run_network_sa

    well_being = lp.load_phq9("data/confidential/phq9.sav", 200, seed=43)
    df, si_df, best = run_network_sa(well_being)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

try:
    import powerlaw as _powerlaw
    _POWERLAW_OK = True
except ImportError:
    _POWERLAW_OK = False
    warnings.warn("powerlaw not found; gamma/KS will be NaN", stacklevel=2)


# ──────────────────────────────────────────────────────────────────────────────
# Problem definition
# ──────────────────────────────────────────────────────────────────────────────

N_CLUSTERS_FIXED = 2   # SA shows S1 < 0.03 for calibration targets; fix at 2 (targets calibrated here)

PROBLEM = {
    "num_vars": 5,
    "names":    ["alpha", "n_clusters", "latent_weight", "dim", "age_weight"],
    "bounds":   [
        [0.5,  5.0],   # alpha
        [2.0, 10.0],   # n_clusters (integer)
        [0.5, 20.0],   # latent_weight
        [2.0,  6.0],   # dim  (rounded to int)
        [0.5,  5.0],   # age_weight
    ],
}

PARAM_NAMES = PROBLEM["names"]

# Calibration targets
TARGETS = {
    "C":           (0.10, 0.20),  # range; loss = range-distance
    "age_assort":  0.25,
    "phq9_assort": 0.03,
}

METRICS = ["C", "age_assort", "phq9_assort", "gamma", "ks"]

# For plot shading / annotations
REF_RANGES = {
    "C":           (0.10, 0.20),
    "age_assort":  (0.10, 0.40),
    "phq9_assort": (0.01, 0.09),
    "gamma":       (2.0,  3.0),
    "ks":          (None, 0.10),
}


# ──────────────────────────────────────────────────────────────────────────────
# Single-sample evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _eval_one(params, well_being, N, degree, seed, dist_type, src_path):
    """Build one SDA network and return topology metrics.

    Uses SocialDistanceAttachment directly (no code duplication). Stdout/stderr
    from the network build are suppressed so joblib progress stays clean.

    Parameters
    ----------
    params : array-like, length 5
        [alpha, n_clusters, latent_weight, dim, age_weight]
    well_being : list[dict]
        Real well-being data as returned by lp.load_phq9.
    src_path : str
        Absolute path to the ``src`` directory; added to sys.path in the
        worker so SocialDistanceAttachment is importable without venv hacks.
    """
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from classes.network import SocialDistanceAttachment  # noqa: PLC0415

    alpha, n_clusters, latent_weight, dim, age_weight = params
    n_clusters = int(round(n_clusters))
    dim        = int(round(dim))

    nan_row = {m: np.nan for m in METRICS}

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        try:
            net = SocialDistanceAttachment(
                alpha=alpha,
                dim=dim,
                degree=degree,
                num_agents=N,
                seed=seed,
                dist_type=dist_type,
                n_clusters=n_clusters,
                latent_weight=latent_weight,
                age_weight=age_weight,
                well_being=list(well_being),
            )
        except Exception:
            return nan_row

    # Build networkx graph from the initialised network
    g = nx.Graph()
    for agent in net.all_agents:
        wb = agent.well_being or {}
        g.add_node(agent.ID,
                   age=float(wb.get("age", 0)),
                   phq9=float(wb.get("phq9_sumscore", 0)))
    for conn in net.connections:
        g.add_edge(conn[0].ID, conn[1].ID)

    C = nx.average_clustering(g)

    try:
        age_assort = nx.numeric_assortativity_coefficient(g, "age")
    except Exception:
        age_assort = np.nan

    try:
        phq9_assort = nx.numeric_assortativity_coefficient(g, "phq9")
    except Exception:
        phq9_assort = np.nan

    gamma = np.nan
    ks    = np.nan
    if _POWERLAW_OK:
        degrees = [d for _, d in g.degree()]
        if max(degrees, default=0) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit   = _powerlaw.Fit(degrees, verbose=False)
                gamma = float(fit.power_law.alpha)
                ks    = float(fit.power_law.KS())

    return {"C": C, "age_assort": age_assort, "phq9_assort": phq9_assort,
            "gamma": gamma, "ks": ks}


# ──────────────────────────────────────────────────────────────────────────────
# Sobol SA
# ──────────────────────────────────────────────────────────────────────────────

def run_sobol_sa(well_being: list[dict],
                 N: int = 200,
                 degree: int = 6,
                 seed: int = 43,
                 n_sobol: int = 512,
                 n_jobs: int = -1,
                 dist_type: str = "gaussian_clusters",
                 out_dir: str = "data/sensitivity/network",
                 samples=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Sobol SA for one seed; return (samples_df, sobol_indices_df).

    Total evaluations = n_sobol × (2 × num_vars + 2) = n_sobol × 12.
    For n_sobol=512 that is 6 144 graph builds (no LLM).

    Parameters
    ----------
    samples : optional pre-computed Sobol sample matrix (shape n×5).
        Pass the same array across multiple seeds to share the parameter
        grid — differences in results then reflect only network realization
        noise, not different parameter coverage.
    """
    os.makedirs(out_dir, exist_ok=True)

    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    if samples is None:
        samples = saltelli.sample(PROBLEM, n_sobol, calc_second_order=False)
    n_total = len(samples)
    print(f"[sa_network] {n_total} evaluations  "
          f"(n_sobol={n_sobol} × {2*PROBLEM['num_vars']+2})")

    results = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(_eval_one)(p, well_being, N, degree, seed, dist_type, src_path)
        for p in samples
    )

    df = pd.DataFrame(samples, columns=PARAM_NAMES)
    df["n_clusters"] = df["n_clusters"].round().astype(int)
    df["dim"]        = df["dim"].round().astype(int)
    for m in METRICS:
        df[m] = [r[m] for r in results]

    n_err = df[METRICS].isna().all(axis=1).sum()
    print(f"[sa_network] {n_err}/{n_total} evaluations fully NaN (bisection failed)")
    df.to_csv(os.path.join(out_dir, "sobol_samples.csv"), index=False)

    # Sobol indices
    si_rows = []
    for metric in METRICS:
        Y     = df[metric].values
        valid = ~np.isnan(Y)
        if valid.mean() < 0.80:
            print(f"[sa_network] {metric}: <80% valid → skipping Sobol indices")
            continue
        Y_filled = np.where(np.isnan(Y), np.nanmean(Y), Y)
        si = sobol_analyze.analyze(PROBLEM, Y_filled,
                                   calc_second_order=False, print_to_console=False)
        for i, name in enumerate(PARAM_NAMES):
            si_rows.append({
                "metric":  metric, "param":  name,
                "S1":      float(si["S1"][i]),
                "S1_conf": float(si["S1_conf"][i]),
                "ST":      float(si["ST"][i]),
                "ST_conf": float(si["ST_conf"][i]),
            })

    si_df = pd.DataFrame(si_rows)
    si_df.to_csv(os.path.join(out_dir, "sobol_indices.csv"), index=False)
    print(f"[sa_network] Sobol indices → {out_dir}/sobol_indices.csv")
    return df, si_df


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_sobol_indices(si_df: pd.DataFrame, out_dir: str) -> None:
    """Grouped bar chart: S1 and ST per (metric × parameter)."""
    metrics = [m for m in METRICS if m in si_df["metric"].unique()]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = si_df[si_df["metric"] == metric].reset_index(drop=True)
        x, w = np.arange(len(sub)), 0.38
        ax.bar(x - w/2, sub["S1"], w, yerr=sub["S1_conf"], label="S1 (first-order)",
               color="#3498db", edgecolor="black", linewidth=0.5, capsize=4)
        ax.bar(x + w/2, sub["ST"], w, yerr=sub["ST_conf"], label="ST (total-order)",
               color="#e74c3c", edgecolor="black", linewidth=0.5, capsize=4, alpha=0.80)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["param"], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Sobol index")
        ax.set_title(metric)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        if ax is axes[0]:
            ax.legend(fontsize=9)
    fig.suptitle("Sobol sensitivity indices — network topology SA", fontsize=12)
    fig.tight_layout()
    out = os.path.join(out_dir, "sobol_indices.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[sa_network] → {out}")


def plot_scatter_grid(df: pd.DataFrame, out_dir: str,
                      filename: str = "scatter_grid.png",
                      params: list[str] | None = None) -> None:
    """rows = metrics, cols = parameters; green band / line = target.

    Parameters
    ----------
    params : which columns to use as parameters. Defaults to all DataFrame
        columns that are not metric columns — so old CSVs with n_clusters
        will include it automatically even after it was removed from PROBLEM.
    """
    _non_metric = [c for c in df.columns if c not in METRICS]
    params = params if params is not None else _non_metric

    fig, axes = plt.subplots(len(METRICS), len(params),
                             figsize=(1.5 * len(params), 1.4 * len(METRICS)),
                             sharex="col", sharey="row", squeeze=False)
    for row, metric in enumerate(METRICS):
        for col, param in enumerate(params):
            ax    = axes[row][col]
            valid = df[metric].notna()
            ax.scatter(df.loc[valid, param], df.loc[valid, metric],
                       s=3, alpha=0.25, color="#2c3e50", linewidths=0, rasterized=True)
            ax.set_xlabel(param if row == len(METRICS) - 1 else "", fontsize=8)
            ax.set_ylabel(metric if col == 0 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(linestyle=":", alpha=0.4)
            ref = REF_RANGES.get(metric)
            if ref:
                lo, hi = ref
                if lo is not None and hi is not None and lo != hi:
                    ax.axhspan(lo, hi, color="#d96907", alpha=0.25)
                elif lo is not None and lo == hi:
                    ax.axhline(lo, color="#d96907", linewidth=1.0, linestyle="--", alpha=0.8)
                elif hi is not None:
                    ax.axhline(hi, color="#d96907", linewidth=1.0, linestyle="--", alpha=0.8)
    fig.tight_layout()
    out = os.path.join(out_dir, filename)
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[sa_network] → {out}")


def plot_scatter_grid_averaged(seed_dfs: dict[int, pd.DataFrame],
                               out_dir: str) -> None:
    """Scatter grid with metrics averaged across seeds per Sobol sample.

    Since all seeds share the same parameter combinations, averaging the
    metric values reduces per-realization noise and shows the true
    parameter-metric relationship more cleanly than any single seed.
    Saved as scatter_grid_averaged.png in out_dir.
    """
    if not seed_dfs:
        return
    seeds = sorted(seed_dfs.keys())

    # Parameter columns are identical across seeds — take from first.
    # Use all non-metric columns so removed params (e.g. n_clusters) still appear.
    first = seed_dfs[seeds[0]]
    param_cols = [c for c in first.columns if c not in METRICS]
    avg_df = first[param_cols].copy()

    # Average each metric across seeds row-wise
    for m in METRICS:
        cols = [seed_dfs[s][m].values for s in seeds if m in seed_dfs[s].columns]
        if cols:
            avg_df[m] = np.nanmean(np.stack(cols, axis=1), axis=1)

    plot_scatter_grid(avg_df, out_dir, filename="scatter_grid_averaged.png")


def plot_parallel_coords(df: pd.DataFrame, out_dir: str, top_n: int = 50) -> None:
    """Parallel coordinates of the top-N best-fitting samples."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    losses = _compute_losses(df)
    top    = df.assign(_loss=losses).nsmallest(top_n, "_loss")
    if top.empty:
        return
    cols = PARAM_NAMES + ["C", "age_assort", "phq9_assort"]
    norm = Normalize(top["_loss"].min(), top["_loss"].max())
    cmap = plt.cm.viridis_r

    # Normalise each column for display
    normed = top[cols].copy()
    for c in cols:
        lo, hi = normed[c].min(), normed[c].max()
        normed[c] = (normed[c] - lo) / (hi - lo + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 4))
    for (_, row), (_, nrow) in zip(top.iterrows(), normed.iterrows()):
        ax.plot(range(len(cols)), nrow.values,
                color=cmap(norm(row["_loss"])), alpha=0.5, linewidth=0.8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Normalised value")
    ax.set_title(f"Top-{top_n} samples by calibration loss  (dark = best)")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="Loss", fraction=0.03, pad=0.02)
    fig.tight_layout()
    out = os.path.join(out_dir, "parallel_coords.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[sa_network] → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

def _range_dist(v: float, lo: float, hi: float) -> float:
    if lo <= v <= hi:
        return 0.0
    return min(abs(v - lo), abs(v - hi))


def _compute_losses(df: pd.DataFrame,
                    targets: dict | None = None) -> np.ndarray:
    """Scalar calibration loss per row (NaN if any key metric is NaN).

    C           : range-distance to target range, normalised by width
    age_assort  : |v - target| / 0.15
    phq9_assort : |v - target| / 0.03

    Parameters
    ----------
    targets : override the module-level TARGETS dict.
        Pass e.g. {"C": (0.12, 0.18), "age_assort": 0.30, "phq9_assort": 0.05}
        to recalibrate without re-running the Sobol evaluation.
    """
    t = targets if targets is not None else TARGETS
    C_target  = t["C"]
    age_t     = t["age_assort"]
    phq_t     = t["phq9_assort"]
    C_lo, C_hi = (C_target if isinstance(C_target, tuple)
                  else (C_target, C_target))
    C_width = max(C_hi - C_lo, 1e-6)

    losses = np.full(len(df), np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        if any(np.isnan([row["C"], row["age_assort"], row["phq9_assort"]])):
            continue
        losses[i] = (
            (_range_dist(row["C"],           C_lo, C_hi)   / C_width) ** 2 +
            (abs(row["age_assort"]  - age_t)               / 0.15)    ** 2 +
            (abs(row["phq9_assort"] - phq_t)               / 0.03)    ** 2
        )
    return losses


def calibrate_from_samples(df: pd.DataFrame, out_dir: str,
                            targets: dict | None = None) -> pd.Series:
    """Find the best row in the Sobol sample set; print and save summary.

    Parameters
    ----------
    targets : optional target override — same format as module-level TARGETS.
        Lets you recalibrate with new targets without re-running the SA.
    """
    losses = _compute_losses(df, targets=targets)
    valid  = ~np.isnan(losses)
    if not valid.any():
        print("[sa_network] No valid rows for calibration.")
        return df.iloc[0]

    best_idx  = int(np.nanargmin(losses))
    best_row  = df.loc[best_idx]
    best_loss = float(losses[best_idx])

    top10_valid_idx = np.where(valid)[0][np.argsort(losses[valid])[:10]]
    top10_df = df.loc[top10_valid_idx].copy()
    top10_df["loss"] = losses[top10_valid_idx]
    top10_df = top10_df.reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"[sa_network] Best fit  (loss = {best_loss:.4f})")
    print(f"{'='*60}")
    for p in PARAM_NAMES:
        print(f"  {p:<16} = {best_row[p]:.4g}")
    print()
    for m in METRICS:
        ref = REF_RANGES.get(m, (None, None))
        lo, hi = ref
        if lo is not None and hi is not None and lo != hi:
            tag = f"  [target {lo}–{hi}]"
        elif lo is not None and lo == hi:
            tag = f"  [target {lo}]"
        elif hi is not None:
            tag = f"  [target < {hi}]"
        else:
            tag = "  (observe)"
        print(f"  {m:<16} = {best_row[m]:.4f}{tag}")

    print(f"\n[sa_network] Top-10 parameter sets:")
    print(top10_df[PARAM_NAMES + METRICS + ["loss"]].round(4).to_string(index=False))

    top10_df.to_csv(os.path.join(out_dir, "top10_params.csv"), index=False)
    best_row.to_frame().T.reset_index(drop=True).to_csv(
        os.path.join(out_dir, "best_params.csv"), index=False)
    print(f"\n[sa_network] Saved → {out_dir}/best_params.csv  top10_params.csv")
    return best_row


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_network_sa(well_being: list[dict],
                   N: int = 200,
                   degree: int = 6,
                   seed: int = 43,
                   n_sobol: int = 512,
                   n_jobs: int = -1,
                   dist_type: str = "gaussian_clusters",
                   out_dir: str = "data/sensitivity/network") -> tuple:
    """Full pipeline: Sobol SA → plots → calibration.

    Returns (samples_df, sobol_indices_df, best_row).
    """
    df, si_df = run_sobol_sa(well_being, N=N, degree=degree, seed=seed,
                              n_sobol=n_sobol, n_jobs=n_jobs,
                              dist_type=dist_type, out_dir=out_dir)
    plot_sobol_indices(si_df, out_dir)
    plot_scatter_grid(df, out_dir)
    plot_parallel_coords(df, out_dir)
    best_row = calibrate_from_samples(df, out_dir)
    return df, si_df, best_row


# ──────────────────────────────────────────────────────────────────────────────
# Fast recalibration from saved CSV (no Sobol re-run)
# ──────────────────────────────────────────────────────────────────────────────

def recalibrate_from_csv(out_dir: str, targets: dict | None = None) -> None:
    """Re-run calibration on already-saved sobol_samples.csv files.

    Useful when you want to try different target values without paying the
    cost of the Sobol evaluation again.  Overwrites best_params.csv and
    top10_params.csv in each seed subdirectory (or the root if no seed
    subdirs exist).

    Parameters
    ----------
    out_dir : root output directory (same as passed to run_network_sa[_multi_seed]).
    targets : new targets dict, e.g.
        {"C": (0.12, 0.18), "age_assort": 0.30, "phq9_assort": 0.05}
        Defaults to the module-level TARGETS if not provided.
    """
    t = targets if targets is not None else TARGETS
    print(f"[sa_network] Recalibrating with targets: {t}")

    # Find all sobol_samples.csv files under out_dir
    csv_paths = []
    for root, _, files in os.walk(out_dir):
        if "sobol_samples.csv" in files:
            csv_paths.append(os.path.join(root, "sobol_samples.csv"))

    if not csv_paths:
        print(f"[sa_network] No sobol_samples.csv found under {out_dir}")
        return

    for csv_path in sorted(csv_paths):
        sub_dir = os.path.dirname(csv_path)
        print(f"\n[sa_network] Recalibrating {sub_dir}")
        df = pd.read_csv(csv_path)
        calibrate_from_samples(df, sub_dir, targets=t)

    print(f"\n[sa_network] Recalibration done.  best_params.csv updated in each subdir.")


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed stability
# ──────────────────────────────────────────────────────────────────────────────

_STABILITY_PANEL_LABELS = {
    "C":           "clustering coeff.",
    "age_assort":  "age assort.",
    "phq9_assort": "PHQ-9 assort.",
}

# Project colour scheme (mirrors sa_analyze.py)
_COL_S1 = "#2e7ebc"   # blue  — first-order
_COL_ST = "#d96907"   # orange — total-order


def plot_stability_indices(seed_si: dict[int, pd.DataFrame], out_dir: str) -> None:
    """S1 and ST grouped side-by-side per parameter, one panel per metric.

    Blue bars = S1 (first-order), orange bars = ST (total-order).
    Error caps = mean within-sample SALib CI across seeds.
    Grey dots = individual seed estimates.
    Legend in first panel only.
    """
    cal_metrics = [m for m in ["C", "age_assort", "phq9_assort"]
                   if any(m in df["metric"].values for df in seed_si.values())]
    if not cal_metrics:
        return

    seeds   = sorted(seed_si.keys())
    n_met   = len(cal_metrics)
    w       = 0.35

    fig, axes = plt.subplots(1, n_met,
                             figsize=(2.3 * n_met, 2.2),
                             sharey=True)
    if n_met == 1:
        axes = [axes]

    for k, (ax, metric) in enumerate(zip(axes, cal_metrics)):
        x = np.arange(len(PARAM_NAMES))

        for idx, color, offset, label in [
            ("S1", _COL_S1, -w / 2, "S1"),
            ("ST", _COL_ST, +w / 2, "ST"),
        ]:
            conf_key = f"{idx}_conf"
            val_mat  = np.full((len(seeds), len(PARAM_NAMES)), np.nan)
            conf_mat = np.full((len(seeds), len(PARAM_NAMES)), np.nan)
            for si, seed in enumerate(seeds):
                sub = seed_si[seed]
                sub = sub[sub["metric"] == metric].set_index("param")
                for pi, param in enumerate(PARAM_NAMES):
                    if param in sub.index:
                        val_mat[si, pi]  = sub.loc[param, idx]
                        if conf_key in sub.columns:
                            conf_mat[si, pi] = sub.loc[param, conf_key]

            means     = np.nanmean(val_mat,  axis=0)
            mean_conf = np.nanmean(conf_mat, axis=0)

            ax.bar(x + offset, means, w, yerr=mean_conf,
                   color=color, edgecolor="black", linewidth=0.5,
                   capsize=3, error_kw={"elinewidth": 0.7},
                   alpha=0.80, zorder=2,
                   label=label if k == 0 else None)

            for si in range(len(seeds)):
                ax.scatter(x + offset, val_mat[si], color="#7f7f7f",
                           s=12, zorder=4, alpha=0.7, linewidths=0)

        ax.set_xticks(x)
        ax.set_xticklabels(PARAM_NAMES, rotation=30, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

        panel_label = f"({'abcde'[k]}) {_STABILITY_PANEL_LABELS.get(metric, metric)}"
        ax.text(0.5, -0.38, panel_label,
                transform=ax.transAxes, ha="center", va="top", fontsize=10)

    axes[0].set_ylabel("Sobol index", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.85)

    out = os.path.join(out_dir, "stability_indices.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[sa_network] → {out}")


def replot_stability(out_dir: str) -> None:
    """Reload saved CSVs and regenerate all stability plots — no network builds.

    Reads ``seed_*/sobol_indices.csv`` for the S1/ST plots and
    ``seed_*/sobol_samples.csv`` for the rank-correlation plot.
    """
    seed_si  = {}
    seed_dfs = {}
    for entry in sorted(os.listdir(out_dir)):
        if not entry.startswith("seed_"):
            continue
        seed = int(entry.split("_")[1])
        si_path  = os.path.join(out_dir, entry, "sobol_indices.csv")
        smp_path = os.path.join(out_dir, entry, "sobol_samples.csv")
        if os.path.isfile(si_path):
            seed_si[seed]  = pd.read_csv(si_path)
        if os.path.isfile(smp_path):
            seed_dfs[seed] = pd.read_csv(smp_path)

    if not seed_si:
        print(f"[sa_network] No seed_*/sobol_indices.csv found under {out_dir}")
        return

    print(f"[sa_network] Replotting stability for seeds: {sorted(seed_si)}")
    plot_stability_indices(seed_si, out_dir)
    if seed_dfs:
        plot_loss_rank_correlation(seed_dfs, out_dir)
        plot_scatter_grid_averaged(seed_dfs, out_dir)
        for seed, df in seed_dfs.items():
            seed_dir = os.path.join(out_dir, f"seed_{seed}")
            plot_scatter_grid(df, seed_dir)
            plot_sobol_indices(seed_si[seed], seed_dir)
    pick_best_across_seeds(out_dir)


def plot_loss_rank_correlation(seed_dfs: dict[int, pd.DataFrame],
                               out_dir: str) -> pd.DataFrame:
    """Pairwise Spearman rank correlation of calibration loss vectors.

    Because all seeds share the same Sobol sample matrix, each seed produces
    a loss value for the exact same parameter combinations.  Spearman
    correlation between those loss vectors answers: does the same parameter
    region rank well regardless of which seed was used?

    rho ≈ 1  → the landscape is stable; seed choice doesn't matter
    rho ≈ 0  → network realization noise dominates; results can't be trusted

    Outputs
    -------
    stability_rank_corr.png   — annotated heatmap
    stability_rank_corr.csv   — correlation matrix
    """
    from scipy.stats import spearmanr

    seeds = sorted(seed_dfs.keys())
    n = len(seeds)

    # Compute loss vectors; replace NaN with max loss so they rank at the bottom
    loss_vecs = {}
    for seed, df in seed_dfs.items():
        losses = _compute_losses(df)
        finite = losses[~np.isnan(losses)]
        fill   = float(finite.max()) if len(finite) else 1.0
        loss_vecs[seed] = np.where(np.isnan(losses), fill, losses)

    # Pairwise Spearman
    mat = np.full((n, n), np.nan)
    for i, si in enumerate(seeds):
        for j, sj in enumerate(seeds):
            if i == j:
                mat[i, j] = 1.0
            elif i < j:
                rho, _ = spearmanr(loss_vecs[si], loss_vecs[sj])
                mat[i, j] = mat[j, i] = float(rho)

    labels = [f"seed {s}" for s in seeds]
    corr_df = pd.DataFrame(mat, index=labels, columns=labels)
    corr_df.to_csv(os.path.join(out_dir, "stability_rank_corr.csv"))

    # Plot
    fig, ax = plt.subplots(figsize=(0.9 * n + 2.5, 0.9 * n + 2.0))
    vmin = max(0.0, float(np.nanmin(mat[mat < 1.0])) - 0.05) if (mat < 1.0).any() else 0.0
    im = ax.imshow(mat, cmap="RdYlGn", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    threshold = vmin + (1.0 - vmin) * 0.6
    for i in range(n):
        for j in range(n):
            colour = "white" if mat[i, j] < threshold else "black"
            ax.text(j, i, f"{mat[i, j]:.3f}",
                    ha="center", va="center", fontsize=10, color=colour)
    ax.set_title("Spearman rank correlation of calibration loss\n"
                 "(rho ≈ 1 = same parameter region ranks well across seeds)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    fig.tight_layout()
    out = os.path.join(out_dir, "stability_rank_corr.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[sa_network] → {out}")

    # Print summary
    off_diag = mat[~np.eye(n, dtype=bool)]
    print(f"[sa_network] Loss rank correlation — "
          f"mean rho = {off_diag.mean():.3f},  min = {off_diag.min():.3f}")
    return corr_df


def _apply_fixes(df: pd.DataFrame, fixes: dict[str, float]) -> pd.DataFrame:
    """Filter df to rows where each fixed parameter matches its target value.

    Integer columns (n_clusters, dim): exact match after rounding.
    Continuous columns: within ±5 % of the parameter's total range.
    Warns if a fix column is missing or leaves no rows.
    """
    _INT_PARAMS = {"n_clusters", "dim"}
    _BOUNDS = {name: (lo, hi) for name, (lo, hi) in
               zip(PROBLEM["names"], PROBLEM["bounds"])}

    mask = pd.Series(True, index=df.index)
    for param, value in fixes.items():
        if param not in df.columns:
            print(f"[sa_network] WARNING: --fix {param} ignored "
                  f"(column not in CSV — was it in this SA run?)")
            continue
        if param in _INT_PARAMS:
            col_rounded = df[param].round().astype(int)
            sub = col_rounded == int(round(value))
        else:
            lo, hi = _BOUNDS.get(param, (df[param].min(), df[param].max()))
            tol = (hi - lo) * 0.05
            sub = (df[param] - value).abs() <= tol
        n_match = sub.sum()
        if n_match == 0:
            print(f"[sa_network] WARNING: --fix {param}={value} matched 0 rows "
                  f"(value outside sampled range {_BOUNDS.get(param, '?')}). "
                  f"Rerun the SA with this value fixed to get valid results.")
        else:
            print(f"[sa_network] --fix {param}={value}: {n_match} rows retained")
        mask &= sub
    filtered = df[mask]
    if filtered.empty:
        print("[sa_network] WARNING: no rows survive all fixes — returning unfiltered")
        return df
    return filtered


def pick_best_across_seeds(out_dir: str,
                           targets: dict | None = None,
                           fixes: dict[str, float] | None = None) -> pd.DataFrame:
    """Pick the best parameter combination by averaging loss across all seeds.

    Because all seeds share the same Sobol sample matrix (same row = same
    parameter combination), we can compute a loss for each seed and average
    them. The combination with the lowest *mean* loss is robust to network
    realization noise. The std tells you how confident you can be.

    Reads existing ``seed_*/sobol_samples.csv`` files — no network builds.

    Returns a DataFrame of the top-10 combinations ranked by mean loss,
    with columns: parameters, per-seed metrics (mean across seeds),
    mean_loss, std_loss.  Also saves ``averaged_best.csv``.
    """
    t = targets if targets is not None else TARGETS

    # Load all seed CSVs
    seed_dfs = {}
    for entry in sorted(os.listdir(out_dir)):
        if not entry.startswith("seed_"):
            continue
        csv_path = os.path.join(out_dir, entry, "sobol_samples.csv")
        if os.path.isfile(csv_path):
            seed = int(entry.split("_")[1])
            seed_dfs[seed] = pd.read_csv(csv_path)

    if not seed_dfs:
        print(f"[sa_network] No seed_*/sobol_samples.csv found under {out_dir}")
        return pd.DataFrame()

    seeds = sorted(seed_dfs.keys())

    # Apply parameter fixes — filter each seed's DataFrame identically
    if fixes:
        seed_dfs = {s: _apply_fixes(df, fixes) for s, df in seed_dfs.items()}

    print(f"[sa_network] Averaging loss across {len(seeds)} seeds "
          f"({len(seed_dfs[seeds[0]])} rows after filtering)")

    # Compute loss per seed — shape (n_samples,) per seed
    loss_matrix = np.stack(
        [_compute_losses(seed_dfs[s], targets=t) for s in seeds],
        axis=1
    )  # shape (n_samples, n_seeds)

    mean_loss = np.nanmean(loss_matrix, axis=1)
    std_loss  = np.nanstd( loss_matrix, axis=1)

    # Use parameter columns from first seed (identical across seeds)
    base_df = seed_dfs[seeds[0]][PARAM_NAMES].copy()

    # Average each metric across seeds for context
    for m in ["C", "age_assort", "phq9_assort", "gamma", "ks"]:
        cols = [seed_dfs[s][m].values for s in seeds if m in seed_dfs[s].columns]
        if cols:
            base_df[m] = np.nanmean(np.stack(cols, axis=1), axis=1)

    base_df["mean_loss"] = mean_loss
    base_df["std_loss"]  = std_loss

    valid = base_df["mean_loss"].notna()
    top10 = base_df[valid].nsmallest(10, "mean_loss").reset_index(drop=True)

    best = top10.iloc[0]
    print(f"\n{'='*60}")
    print(f"[sa_network] Best parameters (mean loss = {best['mean_loss']:.4f} "
          f"± {best['std_loss']:.4f} across {len(seeds)} seeds)")
    print(f"{'='*60}")
    for p in PARAM_NAMES:
        print(f"  {p:<16} = {best[p]:.4g}")
    print()
    for m in ["C", "age_assort", "phq9_assort"]:
        ref = REF_RANGES.get(m, (None, None))
        lo, hi = ref
        if lo is not None and hi is not None and lo != hi:
            tag = f"  [target {lo}–{hi}]"
        elif lo is not None and lo == hi:
            tag = f"  [target {lo}]"
        else:
            tag = ""
        print(f"  {m:<16} = {best[m]:.4f}{tag}")

    print(f"\n[sa_network] Top-10 by mean loss:")
    display_cols = PARAM_NAMES + ["C", "age_assort", "phq9_assort", "mean_loss", "std_loss"]
    print(top10[display_cols].round(4).to_string(index=False))

    out_path = os.path.join(out_dir, "averaged_best.csv")
    top10.to_csv(out_path, index=False)
    print(f"\n[sa_network] Saved → {out_path}")
    return top10


def _consensus_best(seed_best: dict[int, pd.Series], out_dir: str) -> pd.DataFrame:
    """Average best-fit parameters across seeds; save consensus_best.csv."""
    rows = pd.DataFrame({s: b[PARAM_NAMES] for s, b in seed_best.items()}).T
    rows.index.name = "seed"
    summary = rows.agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    print(f"\n[sa_network] Consensus best parameters across {len(seed_best)} seeds:")
    print(summary.round(4).to_string())
    rows.reset_index().to_csv(os.path.join(out_dir, "per_seed_best.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "consensus_best.csv"))
    print(f"[sa_network] Saved → {out_dir}/per_seed_best.csv  consensus_best.csv")
    return summary


def run_network_sa_multi_seed(well_being: list[dict],
                               seeds: list[int],
                               N: int = 200,
                               degree: int = 6,
                               n_sobol: int = 512,
                               n_jobs: int = -1,
                               dist_type: str = "gaussian_clusters",
                               out_dir: str = "data/sensitivity/network") -> dict:
    """Run the full SA pipeline for each seed; aggregate stability metrics.

    Sobol samples are generated once and shared across all seeds so that
    differences in S1/ST reflect network realization noise, not different
    parameter coverage.

    Per-seed outputs go to  ``out_dir/seed_{s}/``.
    Stability outputs (S1 plot, consensus best) go to ``out_dir/``.

    Returns
    -------
    dict mapping seed → (samples_df, si_df, best_row)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Generate the shared sample matrix once
    shared_samples = saltelli.sample(PROBLEM, n_sobol, calc_second_order=False)
    print(f"[sa_network] {len(shared_samples)} shared Sobol samples × "
          f"{len(seeds)} seeds = {len(shared_samples)*len(seeds)} total builds")

    results   = {}
    seed_dfs  = {}
    seed_si   = {}
    seed_best = {}

    for seed in seeds:
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        print(f"\n{'='*60}")
        print(f"[sa_network] Seed {seed}  →  {seed_dir}")
        print(f"{'='*60}")

        df, si_df = run_sobol_sa(
            well_being, N=N, degree=degree, seed=seed,
            n_sobol=n_sobol, n_jobs=n_jobs,
            dist_type=dist_type, out_dir=seed_dir,
            samples=shared_samples,
        )
        plot_sobol_indices(si_df, seed_dir)
        plot_scatter_grid(df, seed_dir)
        plot_parallel_coords(df, seed_dir)
        best = calibrate_from_samples(df, seed_dir)

        results[seed]   = (df, si_df, best)
        seed_dfs[seed]  = df
        seed_si[seed]   = si_df
        seed_best[seed] = best

    # Stability summary across seeds
    plot_stability_indices(seed_si, out_dir)
    plot_loss_rank_correlation(seed_dfs, out_dir)
    plot_scatter_grid_averaged(seed_dfs, out_dir)
    _consensus_best(seed_best, out_dir)
    pick_best_across_seeds(out_dir)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--well-being",      default="data/confidential/phq9.sav")
    p.add_argument("--n-sobol",         type=int, default=512,
                   help="Sobol base N; total evals = N×12")
    p.add_argument("--n-agents",        type=int, default=200)
    p.add_argument("--degree",          type=int, default=6)
    p.add_argument("--seeds",           type=int, nargs="+", default=[43],
                   help="One or more network construction seeds. "
                        "Multiple seeds → multi-seed stability run.")
    p.add_argument("--n-jobs",          type=int, default=-1)
    p.add_argument("--dist-type",       default="gaussian_clusters",
                   choices=["gaussian_clusters", "lognormal"])
    p.add_argument("--out-dir",         default="data/sensitivity/network")
    # ── Recalibration (skips the expensive Sobol evaluation) ──────────────────
    p.add_argument("--replot",          action="store_true",
                   help="Reload saved CSVs and regenerate stability plots only. "
                        "No network builds.")
    p.add_argument("--pick-best",       action="store_true",
                   help="Average loss across saved seed runs and print the "
                        "best parameter combination.  No network builds.")
    p.add_argument("--recalibrate",     action="store_true",
                   help="Re-run calibration only on saved sobol_samples.csv "
                        "files.  No network builds.  Use with --*-target flags "
                        "to try different targets.")
    p.add_argument("--c-target",        type=float, nargs=2,
                   metavar=("LO", "HI"), default=None,
                   help="C target range, e.g. --c-target 0.10 0.20")
    p.add_argument("--age-target",      type=float, default=None,
                   help="age_assort point target, e.g. --age-target 0.30")
    p.add_argument("--phq9-target",     type=float, default=None,
                   help="phq9_assort point target, e.g. --phq9-target 0.05")
    p.add_argument("--fix",             action="append", nargs=2,
                   metavar=("PARAM", "VALUE"), default=[],
                   help="Fix a parameter to a value during --pick-best, "
                        "filtering samples to those matching that value. "
                        "Can be repeated. E.g. --fix n_clusters 2 --fix dim 4")
    args = p.parse_args()

    # Build targets dict from CLI overrides (falls back to module defaults)
    cli_targets = {}
    if args.c_target is not None:
        cli_targets["C"] = tuple(args.c_target)
    if args.age_target is not None:
        cli_targets["age_assort"] = args.age_target
    if args.phq9_target is not None:
        cli_targets["phq9_assort"] = args.phq9_target
    targets = {**TARGETS, **cli_targets} if cli_targets else None

    fixes = {param: float(value) for param, value in args.fix} if args.fix else None

    if args.replot:
        replot_stability(args.out_dir)
        return

    if args.pick_best:
        pick_best_across_seeds(args.out_dir, targets=targets, fixes=fixes)
        return

    if args.recalibrate:
        recalibrate_from_csv(args.out_dir, targets=targets)
        return

    sys.path.insert(0, "src")
    import utils.tools.load_personas as lp
    well_being = lp.load_phq9(args.well_being, args.n_agents, seed=args.seeds[0])

    if len(args.seeds) == 1:
        run_network_sa(
            well_being,
            N=args.n_agents, degree=args.degree, seed=args.seeds[0],
            n_sobol=args.n_sobol, n_jobs=args.n_jobs,
            dist_type=args.dist_type, out_dir=args.out_dir,
        )
    else:
        run_network_sa_multi_seed(
            well_being,
            seeds=args.seeds,
            N=args.n_agents, degree=args.degree,
            n_sobol=args.n_sobol, n_jobs=args.n_jobs,
            dist_type=args.dist_type, out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
