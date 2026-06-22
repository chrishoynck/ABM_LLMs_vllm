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
    lcc          – fraction of nodes in the largest connected component
                   (detects fragmented topologies; observed only)

Reference targets (Twitter / online social networks, McPherson et al.)
----------------------------------------------------------------------
    C          : 0.10 – 0.20   (ER baseline ≈ k/N ≈ 0.03)
    age_assort : 0.25           (point target)
    phq9_assort: 0.03           (point target)
    gamma      : 2.0 – 3.0     (observed only)
    ks         : < 0.10        (observed only)

This module covers two network modes (select with ``--net``):
    sda  – SocialDistanceAttachment (sdc=False): calibrate C + assortativities.
    sdc  – SDA + stub matching (sdc=True): scale-free degree sequence; calibrate
           the degree distribution (gamma band, KS < 0.10, mean degree ≈ goal)
           together with clustering C (wide range) and PHQ-9 assortativity
           (soft/informative wide band). Swept: alpha, stub_gamma, degree, dim,
           latent_weight; n_clusters and age_weight are fixed (S1≈0 on targets).
           The degree sweep brackets the goal so the realized mean (which falls
           below the target fed in, due to the stub shortfall) straddles it.

Usage – CLI
-----------
    # SDA (default)
    PYTHONPATH=src python -m utils.sensitivity.sa_network \\
        --well-being data/confidential/phq9.sav \\
        --n-sobol 512 --n-jobs -1 \\
        --out-dir data/sensitivity/network

    # SDC
    PYTHONPATH=src python -m utils.sensitivity.sa_network --net sdc \\
        --well-being data/confidential/phq9.sav \\
        --n-sobol 512 --n-jobs -1 \\
        --out-dir data/sensitivity/network_sdc

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
# Problem definitions — two network modes share this module
#   "sda" : SocialDistanceAttachment (sdc=False) — calibrate C + assortativities
#   "sdc" : SDA with stub-matching   (sdc=True)  — calibrate degree distribution
#
# Mode is selected with set_mode(net) (CLI: --net). It rebinds the module-level
# PROBLEM / PARAM_NAMES / TARGETS / METRICS / REF_RANGES so every function below
# reads the active config with no further plumbing. _eval_one and _compute_losses
# branch on the mode explicitly because they must also run inside joblib workers,
# which re-import this module fresh and therefore see only the default mode.
# ──────────────────────────────────────────────────────────────────────────────

N_CLUSTERS_FIXED = 2       # SDC: recommended fix once the SA shows S1≈0 on every target
AGE_WEIGHT_FIXED = 2.3149  # SDC: recommended fix once the SA shows S1≈0 (SDA-calibrated value)

GOAL_DEGREE = 4.5      # SDC: realized mean-degree target (accepted within ±DEGREE_TOL)
DEGREE_TOL  = 0.25     # SDC: half-width of the mean-degree acceptance band
LCC_FLOOR   = 0.90     # SDC: realized largest-component fraction must stay ≥ this

# ── SDA: Social Distance Attachment ───────────────────────────────────────────
_PROBLEM_SDA = {
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
_TARGETS_SDA = {
    "C":           (0.10, 0.20),  # range; loss = range-distance
    "age_assort":  0.25,
    "phq9_assort": 0.03,
}
_METRICS_SDA = ["C", "age_assort", "phq9_assort", "gamma", "ks", "lcc"]
_REF_SDA = {
    "C":           (0.10, 0.20),
    "age_assort":  (0.10, 0.40),
    "phq9_assort": (0.01, 0.09),
    "gamma":       (2.0,  3.0),
    "ks":          (None, 0.10),
    "lcc":         (0.90, None),   # ≥ 0.90 → connected; below = fragmented
}

# ── SDC: SDA + stub matching (scale-free degree sequence) ─────────────────────
# stub_gamma is the power-law exponent fed to generate_stub_list. It is NOT the
# realized exponent — that is re-estimated off the built graph as the `gamma`
# metric. degree is swept across a band that brackets the goal: the realized mean
# falls below the target fed in (network_powerlaw leaves stubs unmatched), so the
# sweep spans above and below to straddle the goal.
#
# All 7 params are swept so the Sobol indices document why n_clusters and
# age_weight are safe to fix afterwards (their S1≈0 on every calibration target).
# Recommended post-SA fix values: N_CLUSTERS_FIXED / AGE_WEIGHT_FIXED.
_PROBLEM_SDC = {
    "num_vars": 7,
    "names":    ["alpha", "stub_gamma", "degree", "dim",
                 "n_clusters", "latent_weight", "age_weight"],
    "bounds":   [
        [0.5,  5.0],   # alpha          (drives C)
        [1.5,  3.0],   # stub_gamma     (power-law exponent; bounded sampler, floored at 1.5)
        [4.5,  8.5],   # degree         (target mean fed in; brackets goal=4.5)
        [2.0,  6.0],   # dim            (rounded to int; drives phq9_assort)
        [2.0, 10.0],   # n_clusters     (integer; swept to justify dropping it)
        [0.5, 20.0],   # latent_weight  (secondary on phq9_assort)
        [0.5,  5.0],   # age_weight     (swept to justify dropping it)
    ],
}
_TARGETS_SDC = {
    "gamma":       (2.0, 3.0),     # match: realized power-law exponent band
    "ks":          0.10,           # constraint: KS goodness-of-fit < 0.10
    "mean_degree": (GOAL_DEGREE - DEGREE_TOL, GOAL_DEGREE + DEGREE_TOL),  # match: range ±tol
    "C":           (0.03, 0.20),   # match: clustering range (looser floor; upper 0.20)
    "phq9_assort": (0.0, 0.40),    # soft/informative: wide band so realistic homophily is free
    "lcc":         LCC_FLOOR,      # constraint: largest-component fraction ≥ floor (connectivity)
}
_METRICS_SDC = ["mean_degree", "gamma", "ks", "C", "age_assort", "phq9_assort", "lcc"]
_REF_SDC = {
    "mean_degree": (GOAL_DEGREE - DEGREE_TOL, GOAL_DEGREE + DEGREE_TOL),   # range ±tol
    "gamma":       (2.0,  3.0),
    "ks":          (None, 0.10),
    "C":           (0.03, 0.20),          # match: clustering band (looser floor; upper 0.20)
    "phq9_assort": (0.0,  0.40),          # soft/informative band (wide)
    "age_assort":  (0.10, 0.40),          # observed-only (no longer calibrated)
    "lcc":         (0.90, None),
}

# SDC loss normalisation scales (1 unit = one "acceptable" deviation)
_SDC_KS_SCALE   = 0.10   # KS units above the 0.10 limit
_SDC_DEG_SCALE  = 2.0    # degree units outside the ±tol band
_SDC_PHQ9_SCALE = 0.20   # phq9_assort units outside its band — loose (informative, not a hard goal)
_SDC_LCC_SCALE  = 0.10   # largest-component fraction below the floor (one-sided)
# (C uses its band width for normalisation, like gamma — see _compute_losses)

# Calibration metrics that enter the loss, per mode (others are observed-only)
_LOSS_METRICS = {
    "sda": ["C", "age_assort", "phq9_assort"],
    "sdc": ["mean_degree", "gamma", "ks", "C", "phq9_assort", "lcc"],
}

# Names (metrics OR params) dropped from the SCATTER GRID only — declutter.
# Everything is still computed, saved, and shown in the Sobol-index and stability
# plots (so the low-sensitivity argument for dropping these is still visible there).
_SCATTER_EXCLUDE = {
    "sda": set(),
    "sdc": {"age_assort", "n_clusters", "age_weight"},
}

# Params held fixed when SEARCHING for the optimum, per mode. They are still swept
# in the SA (so the Sobol indices justify the fix), but the calibration / pick-best
# step restricts to these values. CLI --fix overrides/extends this.
_SEARCH_FIXES = {
    "sda": {},
    "sdc": {"n_clusters": N_CLUSTERS_FIXED, "age_weight": AGE_WEIGHT_FIXED},
}

# Every metric _eval_one can return — used to build NaN rows in joblib workers,
# which otherwise see only the default mode's (shorter) METRICS list.
_ALL_METRICS = ["mean_degree", "C", "age_assort", "phq9_assort", "gamma", "ks", "lcc"]

LCC_WARN = LCC_FLOOR   # warn when a selected configuration falls below the connectivity floor

# ── Active config (rebound by set_mode; default = SDA for backward compat) ────
_NET        = "sda"
PROBLEM     = _PROBLEM_SDA
PARAM_NAMES = PROBLEM["names"]
TARGETS     = _TARGETS_SDA
METRICS     = _METRICS_SDA
REF_RANGES  = _REF_SDA


def set_mode(net: str) -> None:
    """Rebind the module-level config globals to the chosen network mode.

    Call once before run_*; idempotent. joblib workers do NOT inherit this —
    _eval_one and _compute_losses take/branch on the mode explicitly.
    """
    global _NET, PROBLEM, PARAM_NAMES, TARGETS, METRICS, REF_RANGES
    if net == "sda":
        _NET, PROBLEM, TARGETS, METRICS, REF_RANGES = (
            "sda", _PROBLEM_SDA, _TARGETS_SDA, _METRICS_SDA, _REF_SDA)
    elif net == "sdc":
        _NET, PROBLEM, TARGETS, METRICS, REF_RANGES = (
            "sdc", _PROBLEM_SDC, _TARGETS_SDC, _METRICS_SDC, _REF_SDC)
    else:
        raise ValueError(f"net must be 'sda' or 'sdc', got {net!r}")
    PARAM_NAMES = PROBLEM["names"]


def _loss_metrics() -> list[str]:
    """Calibration metrics (subset of METRICS) for the active mode."""
    return _LOSS_METRICS[_NET]


def _scatter_exclude() -> set:
    """Names (metrics or params) to drop from the scatter grid for the active mode."""
    return _SCATTER_EXCLUDE.get(_NET, set())


def _search_fixes() -> dict:
    """Params held fixed when searching for the optimum (still swept in the SA)."""
    return dict(_SEARCH_FIXES.get(_NET, {}))


# ──────────────────────────────────────────────────────────────────────────────
# Single-sample evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _eval_one(params, well_being, N, degree, seed, dist_type, src_path, net="sda"):
    """Build one network (SDA or SDC) and return topology metrics.

    Uses SocialDistanceAttachment directly (no code duplication). Stdout/stderr
    from the network build are suppressed so joblib progress stays clean.

    Parameters
    ----------
    params : array-like
        SDA: [alpha, n_clusters, latent_weight, dim, age_weight]
        SDC: [alpha, stub_gamma, degree, dim, n_clusters, latent_weight, age_weight]
    degree : int
        Fixed target degree for SDA. Ignored for SDC (read from ``params``).
    net : {"sda", "sdc"}
        Build mode. SDC sets sdc=True and feeds stub_gamma as the Zipf exponent.
    well_being : list[dict]
        Real well-being data as returned by lp.load_phq9.
    src_path : str
        Absolute path to the ``src`` directory; added to sys.path in the
        worker so SocialDistanceAttachment is importable without venv hacks.

    The returned dict always contains every key in _ALL_METRICS so that workers
    (which see only the default mode's METRICS) still emit well-formed rows.
    """
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from classes.network import SocialDistanceAttachment  # noqa: PLC0415

    nan_row = {m: np.nan for m in _ALL_METRICS}

    if net == "sdc":
        alpha, stub_gamma, degree_t, dim, n_clusters, latent_weight, age_weight = params
        build_kwargs = dict(sdc=True, gamma=float(stub_gamma), degree=float(degree_t))
    else:
        alpha, n_clusters, latent_weight, dim, age_weight = params
        build_kwargs = dict(sdc=False, degree=degree)
    n_clusters = int(round(n_clusters))
    dim        = int(round(dim))

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        try:
            net_obj = SocialDistanceAttachment(
                alpha=alpha,
                dim=dim,
                num_agents=N,
                seed=seed,
                dist_type=dist_type,
                n_clusters=n_clusters,
                latent_weight=latent_weight,
                age_weight=age_weight,
                well_being=list(well_being),
                **build_kwargs,
            )
        except Exception:
            return nan_row

    # Build networkx graph from the initialised network
    g = nx.Graph()
    for agent in net_obj.all_agents:
        wb = agent.well_being or {}
        g.add_node(agent.ID,
                   age=float(wb.get("age", 0)),
                   phq9=float(wb.get("phq9_sumscore", 0)))
    for conn in net_obj.connections:
        g.add_edge(conn[0].ID, conn[1].ID)

    n_nodes     = g.number_of_nodes()
    mean_degree = (2 * g.number_of_edges() / n_nodes) if n_nodes else np.nan
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

    lcc = (len(max(nx.connected_components(g), key=len)) / n_nodes
           if n_nodes else np.nan)

    return {"mean_degree": mean_degree, "C": C, "age_assort": age_assort,
            "phq9_assort": phq9_assort, "gamma": gamma, "ks": ks, "lcc": lcc}


def _out_clustering(g: "nx.DiGraph") -> float:
    """Mean local *out*-clustering coefficient of a directed graph (Fagiolo 2007).

    For every node: the number of ordered out-neighbour pairs (j, h) that are
    themselves linked by an edge j→h, divided by d_out·(d_out−1). Only the
    out-degree enters the denominator — the in-edges of a node never count — so
    this is the clustering "calculated solely with out degree". Nodes with
    out-degree < 2 contribute 0, matching ``nx.average_clustering``'s default
    (count_zeros=True) so the directed value is comparable to the undirected one.
    """
    coeffs = []
    for node in g.nodes():
        succ = set(g.successors(node))
        succ.discard(node)                       # ignore any self-loop
        k = len(succ)
        if k < 2:
            coeffs.append(0.0)
            continue
        links = sum(1 for u in succ for v in succ
                    if u != v and g.has_edge(u, v))
        coeffs.append(links / (k * (k - 1)))
    return float(np.mean(coeffs)) if coeffs else np.nan


def directed_metrics(params, well_being, N, degree, seed, dist_type, src_path,
                     net="sda"):
    """Build the DIRECTED counterpart of one configuration and measure every metric.

    Identical construction to :func:`_eval_one` (same params, same seed) but with
    ``directed=True``: each ordered edge i→j is sampled independently instead of
    being symmetrised, so the resulting ``nx.DiGraph`` — and every metric below —
    genuinely differs from the undirected build. Degree-based metrics use the
    *out*-degree throughout (matching the out-clustering), which keeps the mean on
    the same scale as the undirected panel (out-edges per node ≈ undirected
    neighbours, since each mirrored edge becomes ~one out-edge):

        mean_degree  – mean out-degree (edges / N)
        C            – mean local out-clustering (Fagiolo 2007; see _out_clustering)
        age_assort   – numeric assortativity by age over the directed edges
        phq9_assort  – numeric assortativity by PHQ-9 over the directed edges
        gamma, ks    – power-law fit of the out-degree sequence
        lcc          – largest *weakly* connected component fraction

    Returns a dict with every key in _ALL_METRICS (``nan`` on build failure or
    where a fit is undefined), mirroring :func:`_eval_one`.
    """
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from classes.network import SocialDistanceAttachment  # noqa: PLC0415

    nan_row = {m: np.nan for m in _ALL_METRICS}

    if net == "sdc":
        alpha, stub_gamma, degree_t, dim, n_clusters, latent_weight, age_weight = params
        build_kwargs = dict(sdc=True, gamma=float(stub_gamma), degree=float(degree_t))
    else:
        alpha, n_clusters, latent_weight, dim, age_weight = params
        build_kwargs = dict(sdc=False, degree=degree)
    n_clusters = int(round(n_clusters))
    dim        = int(round(dim))

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        try:
            net_obj = SocialDistanceAttachment(
                alpha=alpha,
                dim=dim,
                num_agents=N,
                seed=seed,
                dist_type=dist_type,
                n_clusters=n_clusters,
                latent_weight=latent_weight,
                age_weight=age_weight,
                well_being=list(well_being),
                directed=True,
                **build_kwargs,
            )
        except Exception:
            return nan_row

    g = nx.DiGraph()
    for agent in net_obj.all_agents:
        wb = agent.well_being or {}
        g.add_node(agent.ID,
                   age=float(wb.get("age", 0)),
                   phq9=float(wb.get("phq9_sumscore", 0)))
    for conn in net_obj.connections:
        if conn[0].ID != conn[1].ID:
            g.add_edge(conn[0].ID, conn[1].ID)

    n_nodes     = g.number_of_nodes()
    mean_degree = (g.number_of_edges() / n_nodes) if n_nodes else np.nan   # out-degree mean
    C = _out_clustering(g)

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
        out_degrees = [d for _, d in g.out_degree()]
        if max(out_degrees, default=0) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit   = _powerlaw.Fit(out_degrees, verbose=False)
                gamma = float(fit.power_law.alpha)
                ks    = float(fit.power_law.KS())

    lcc = (len(max(nx.weakly_connected_components(g), key=len)) / n_nodes
           if n_nodes else np.nan)

    return {"mean_degree": mean_degree, "C": C, "age_assort": age_assort,
            "phq9_assort": phq9_assort, "gamma": gamma, "ks": ks, "lcc": lcc}


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
                 samples=None,
                 net: str = "sda") -> tuple[pd.DataFrame, pd.DataFrame]:
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
    set_mode(net)
    os.makedirs(out_dir, exist_ok=True)

    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    if samples is None:
        samples = saltelli.sample(PROBLEM, n_sobol, calc_second_order=False)
    n_total = len(samples)
    print(f"[sa_network] net={net}  {n_total} evaluations  "
          f"(n_sobol={n_sobol} × {2*PROBLEM['num_vars']+2})")

    results = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(_eval_one)(p, well_being, N, degree, seed, dist_type, src_path, net)
        for p in samples
    )

    df = pd.DataFrame(samples, columns=PARAM_NAMES)
    for _int_col in ("n_clusters", "dim"):
        if _int_col in df.columns:
            df[_int_col] = df[_int_col].round().astype(int)
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
    params : which columns to use as parameters. Defaults to all non-metric
        columns minus the per-mode scatter exclusions (_SCATTER_EXCLUDE), so
        low-sensitivity params (e.g. n_clusters, age_weight) are dropped here
        for clarity while still appearing in the Sobol-index/stability plots.
    """
    excl = _scatter_exclude()
    _non_metric = [c for c in df.columns if c not in METRICS and c not in excl]
    params = params if params is not None else _non_metric
    metrics = [m for m in METRICS if m in df.columns and m not in excl]  # old CSVs may lack lcc

    fig, axes = plt.subplots(len(metrics), len(params),
                             figsize=(1.5 * len(params), 1.4 * len(metrics)),
                             sharex="col", sharey="row", squeeze=False)
    for row, metric in enumerate(metrics):
        for col, param in enumerate(params):
            ax    = axes[row][col]
            valid = df[metric].notna()
            ax.scatter(df.loc[valid, param], df.loc[valid, metric],
                       s=3, alpha=0.25, color="#2c3e50", linewidths=0, rasterized=True)
            ax.set_xlabel(param if row == len(metrics) - 1 else "", fontsize=8)
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
                elif lo is not None or hi is not None:
                    ax.axhline(lo if lo is not None else hi, color="#d96907",
                               linewidth=1.0, linestyle="--", alpha=0.8)
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
    cols = PARAM_NAMES + _loss_metrics()
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

    # ── SDC: degree-distribution + geometry loss ──────────────────────────────
    #   gamma  → squared range-distance to the target band (match)
    #   ks     → squared excess above the limit only (constraint, one-sided)
    #   degree → squared range-distance to the ±tol band (match)
    #   C      → squared range-distance to the clustering band (match)
    #   phq9   → squared range-distance to a wide band, loosely scaled
    #            (soft/informative — nudges, does not drive the optimisation)
    #   lcc    → squared shortfall below the connectivity floor (constraint, one-sided)
    if _NET == "sdc":
        g_lo, g_hi     = t["gamma"]
        g_width        = max(g_hi - g_lo, 1e-6)
        ks_max         = t["ks"]
        deg_lo, deg_hi = t["mean_degree"]    # range ±tol
        C_lo, C_hi     = t["C"]
        C_width        = max(C_hi - C_lo, 1e-6)
        phq_lo, phq_hi = t["phq9_assort"]
        lcc_min        = t["lcc"]            # connectivity floor (one-sided)
        losses = np.full(len(df), np.nan)
        for i, (_, row) in enumerate(df.iterrows()):
            if any(np.isnan([row["gamma"], row["ks"], row["mean_degree"],
                             row["C"], row["phq9_assort"], row["lcc"]])):
                continue
            losses[i] = (
                (_range_dist(row["gamma"], g_lo, g_hi) / g_width)                 ** 2 +
                (max(0.0, row["ks"]          - ks_max)   / _SDC_KS_SCALE)           ** 2 +
                (_range_dist(row["mean_degree"], deg_lo, deg_hi) / _SDC_DEG_SCALE)  ** 2 +
                (_range_dist(row["C"], C_lo, C_hi)       / C_width)                 ** 2 +
                (_range_dist(row["phq9_assort"], phq_lo, phq_hi) / _SDC_PHQ9_SCALE) ** 2 +
                (max(0.0, lcc_min - row["lcc"]) / _SDC_LCC_SCALE)                   ** 2
            )
        return losses

    # ── SDA: clustering + assortativity loss ──────────────────────────────────
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

    The optimum is searched with the mode's _SEARCH_FIXES applied (e.g. SDC holds
    n_clusters/age_weight fixed), so the full SA stays in the saved CSV/plots but
    the *best pick* respects the fix. Does not mutate the caller's df.
    """
    sf = _search_fixes()
    if sf:
        df = _apply_fixes(df, sf).reset_index(drop=True)
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
    metrics = [m for m in METRICS if m in best_row.index]   # old CSVs may lack lcc
    for m in metrics:
        ref = REF_RANGES.get(m, (None, None))
        lo, hi = ref
        if lo is not None and hi is not None and lo != hi:
            tag = f"  [target {lo}–{hi}]"
        elif lo is not None and lo == hi:
            tag = f"  [target {lo}]"
        elif lo is not None:
            tag = f"  [target ≥ {lo}]"
        elif hi is not None:
            tag = f"  [target < {hi}]"
        else:
            tag = "  (observe)"
        print(f"  {m:<16} = {best_row[m]:.4f}{tag}")

    if "lcc" in best_row.index and best_row["lcc"] < LCC_WARN:
        print(f"\n[sa_network] WARNING: best fit is fragmented "
              f"(lcc = {best_row['lcc']:.2f} < {LCC_WARN})")

    print(f"\n[sa_network] Top-10 parameter sets:")
    print(top10_df[PARAM_NAMES + metrics + ["loss"]].round(4).to_string(index=False))

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
                   out_dir: str = "data/sensitivity/network",
                   net: str = "sda") -> tuple:
    """Full pipeline: Sobol SA → plots → calibration.

    Returns (samples_df, sobol_indices_df, best_row).
    """
    set_mode(net)
    df, si_df = run_sobol_sa(well_being, N=N, degree=degree, seed=seed,
                              n_sobol=n_sobol, n_jobs=n_jobs,
                              dist_type=dist_type, out_dir=out_dir, net=net)
    plot_sobol_indices(si_df, out_dir)
    plot_scatter_grid(df, out_dir)
    plot_parallel_coords(df, out_dir)
    best_row = calibrate_from_samples(df, out_dir)
    return df, si_df, best_row


# ──────────────────────────────────────────────────────────────────────────────
# Fast recalibration from saved CSV (no Sobol re-run)
# ──────────────────────────────────────────────────────────────────────────────

def recalibrate_from_csv(out_dir: str, targets: dict | None = None,
                         net: str | None = None) -> None:
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
    if net is not None:
        set_mode(net)
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
    "mean_degree": "mean degree",
    "gamma":       "power-law gamma",
    "ks":          "KS fit",
    "lcc":         "largest comp.",
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
    cal_metrics = [m for m in _loss_metrics()
                   if any(m in df["metric"].values for df in seed_si.values())]
    if not cal_metrics:
        return

    seeds   = sorted(seed_si.keys())
    n_met   = len(cal_metrics)
    w       = 0.35

    ncols = 3
    nrows = max(1, -(-n_met // ncols))   # ceil → 2×3 for the 6 SDC loss metrics
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.6 * ncols, 2.6 * nrows),
                             sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_met:]:              # hide unused cells
        ax.set_visible(False)

    for k, metric in enumerate(cal_metrics):
        ax = axes[k]
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

        panel_label = f"({chr(97 + k)}) {_STABILITY_PANEL_LABELS.get(metric, metric)}"
        ax.text(0.5, -0.50, panel_label,
                transform=ax.transAxes, ha="center", va="top", fontsize=10)

    for r in range(nrows):              # y-label on the left column of each row
        axes[r * ncols].set_ylabel("Sobol index", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.85)

    fig.subplots_adjust(hspace=0.72)    # room for x-tick labels + panel label between rows
    out = os.path.join(out_dir, "stability_indices.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[sa_network] → {out}")


def replot_stability(out_dir: str, net: str | None = None) -> None:
    """Reload saved CSVs and regenerate all stability plots — no network builds.

    Reads ``seed_*/sobol_indices.csv`` for the S1/ST plots and
    ``seed_*/sobol_samples.csv`` for the rank-correlation plot.
    """
    if net is not None:
        set_mode(net)
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
                           fixes: dict[str, float] | None = None,
                           net: str | None = None) -> pd.DataFrame:
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
    if net is not None:
        set_mode(net)
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

    # Apply parameter fixes — the mode's _SEARCH_FIXES (e.g. SDC: n_clusters,
    # age_weight) plus any CLI --fix (which overrides) — filtered identically.
    eff_fixes = _search_fixes()
    if fixes:
        eff_fixes.update(fixes)
    if eff_fixes:
        seed_dfs = {s: _apply_fixes(df, eff_fixes) for s, df in seed_dfs.items()}

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
    for m in METRICS:
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
    disp_metrics = list(dict.fromkeys(_loss_metrics() + ["lcc"]))  # lcc once even if in loss
    for m in disp_metrics:
        if m not in best.index:
            continue
        ref = REF_RANGES.get(m, (None, None))
        lo, hi = ref
        if lo is not None and hi is not None and lo != hi:
            tag = f"  [target {lo}–{hi}]"
        elif lo is not None and lo == hi:
            tag = f"  [target {lo}]"
        elif lo is not None:
            tag = f"  [target ≥ {lo}]"
        else:
            tag = ""
        print(f"  {m:<16} = {best[m]:.4f}{tag}")

    if "lcc" in best.index and best["lcc"] < LCC_WARN:
        print(f"\n[sa_network] WARNING: best combination is fragmented on average "
              f"(lcc = {best['lcc']:.2f} < {LCC_WARN}) — check per-seed components")

    print(f"\n[sa_network] Top-10 by mean loss:")
    display_cols = PARAM_NAMES + [m for m in disp_metrics
                                  if m in top10.columns] + ["mean_loss", "std_loss"]
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
                               out_dir: str = "data/sensitivity/network",
                               net: str = "sda") -> dict:
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
    set_mode(net)
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
            samples=shared_samples, net=net,
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
    p.add_argument("--net",             default="sda", choices=["sda", "sdc"],
                   help="sda: calibrate C + assortativities (degree fixed). "
                        "sdc: stub-matched scale-free; calibrate gamma/KS/degree "
                        "(degree, stub_gamma swept).")
    p.add_argument("--well-being",      default="data/confidential/phq9.sav")
    p.add_argument("--n-sobol",         type=int, default=512,
                   help="Sobol base N; total evals = N×(2·num_vars+2) "
                        "(sda: ×12, sdc: ×16)")
    p.add_argument("--n-agents",        type=int, default=200)
    p.add_argument("--degree",          type=int, default=6,
                   help="SDA only: fixed target degree. Ignored for sdc "
                        "(degree is swept; set its goal with --degree-goal).")
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
    # ── SDC target overrides ──────────────────────────────────────────────────
    p.add_argument("--gamma-target",    type=float, nargs=2,
                   metavar=("LO", "HI"), default=None,
                   help="SDC gamma band, e.g. --gamma-target 2.0 3.0")
    p.add_argument("--ks-max",          type=float, default=None,
                   help="SDC KS upper-bound constraint, e.g. --ks-max 0.10")
    p.add_argument("--degree-goal",     type=float, default=None,
                   help="SDC realized mean-degree point target, e.g. --degree-goal 4.5")
    p.add_argument("--fix",             action="append", nargs=2,
                   metavar=("PARAM", "VALUE"), default=[],
                   help="Fix a parameter to a value during --pick-best, "
                        "filtering samples to those matching that value. "
                        "Can be repeated. E.g. --fix n_clusters 2 --fix dim 4")
    args = p.parse_args()

    # Activate the chosen mode so TARGETS/PROBLEM/etc. below are mode-correct.
    set_mode(args.net)

    # Build targets dict from CLI overrides (falls back to module defaults).
    cli_targets = {}
    if args.net == "sda":
        if args.c_target is not None:
            cli_targets["C"] = tuple(args.c_target)
        if args.age_target is not None:
            cli_targets["age_assort"] = args.age_target
        if args.phq9_target is not None:
            cli_targets["phq9_assort"] = args.phq9_target
    else:  # sdc
        if args.gamma_target is not None:
            cli_targets["gamma"] = tuple(args.gamma_target)
        if args.ks_max is not None:
            cli_targets["ks"] = args.ks_max
        if args.degree_goal is not None:      # point goal → ±DEGREE_TOL band
            cli_targets["mean_degree"] = (args.degree_goal - DEGREE_TOL,
                                          args.degree_goal + DEGREE_TOL)
        if args.c_target is not None:          # SDC now calibrates C too (range)
            cli_targets["C"] = tuple(args.c_target)
        # phq9_assort band is soft/informative; tune it in _TARGETS_SDC if needed.
    targets = {**TARGETS, **cli_targets} if cli_targets else None

    fixes = {param: float(value) for param, value in args.fix} if args.fix else None

    if args.replot:
        replot_stability(args.out_dir, net=args.net)
        return

    if args.pick_best:
        pick_best_across_seeds(args.out_dir, targets=targets, fixes=fixes, net=args.net)
        return

    if args.recalibrate:
        recalibrate_from_csv(args.out_dir, targets=targets, net=args.net)
        return

    sys.path.insert(0, "src")
    import utils.tools.load_personas as lp
    well_being = lp.load_phq9(args.well_being, args.n_agents, seed=args.seeds[0])

    if len(args.seeds) == 1:
        run_network_sa(
            well_being,
            N=args.n_agents, degree=args.degree, seed=args.seeds[0],
            n_sobol=args.n_sobol, n_jobs=args.n_jobs,
            dist_type=args.dist_type, out_dir=args.out_dir, net=args.net,
        )
    else:
        run_network_sa_multi_seed(
            well_being,
            seeds=args.seeds,
            N=args.n_agents, degree=args.degree,
            n_sobol=args.n_sobol, n_jobs=args.n_jobs,
            dist_type=args.dist_type, out_dir=args.out_dir, net=args.net,
        )


if __name__ == "__main__":
    main()
