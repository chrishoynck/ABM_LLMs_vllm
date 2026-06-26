"""Plot the simulated network configurations against the calibration target ranges.

One panel per topology metric — clustering coefficient, age assortativity,
initial PHQ-9 assortativity, and mean degree (observed-only, no target) — laid
out on a 2×2 grid. Shaded bands show the empirical target ranges (REF_RANGES in
sa_network); the mean-degree panel has no band because degree is fixed by
construction, not calibrated. Grey dots are individual network realizations,
open markers flag fragmented realizations (lcc < 0.9), and the coloured marker
gives the mean ± SD per configuration. Every panel also carries a deep-brown
overlay of each config's directed counterpart — the same network built with
``directed=True`` and measured on the directed graph (clustering from out-degree
only, Fagiolo 2007; the degree panel uses the out-degree). The directed graph is
sampled asymmetrically, so all of these differ from the undirected (blue) values.

Data sources
------------
    SDA low degree k=4.5 – rebuilt from the calibrated parameter set at degree 4.5
                           (~30 s, no LLM; deterministic given the construction seeds)
    SDA low C k=4.5      – same set at degree 4.5 with alpha lowered (2.17→1.17)
                           to push clustering below the target band
    SDA low C k=3        – same low-alpha set at degree 3: the strongest joint
                           density + clustering reduction (often drops C below the
                           band and/or fragments — open markers flag this)
    SDA calibrated k=6   – initial graphs of the saved simulation runs at degree 6
                           (old_debiased/2_1655_d6_dim5/meta.json). Debiasing only
                           affects the LLM dynamics, so the initial-graph topology
                           is identical to the non-debiased run at the same seeds.
    SDA high PHQ-9       – calibrated set rebuilt at degree 4.5 with dim 5→3 and
                           latent_weight 7.98→1.0 so the PHQ-9 axis dominates the
                           geometry; pushes PHQ-9 assortativity above its band.

Config labels follow the canonical naming table (e.g. "low C", "calibrated",
"high PHQ-9$_\\rho$"), not the degree — degree is its own panel now, and it varies
between configs (read it off panel (d)).

Usage
-----
    PYTHONPATH=src python -m utils.sensitivity.plot_network_targets \\
        --out data/sensitivity/network_target_ranges.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.sensitivity.sa_network import REF_RANGES, LCC_WARN

_COL_MEAN = "#2e7ebc"   # blue   — mean ± SD
_COL_BAND = "#d96907"   # orange — target range
_COL_DIR  = "#8d2c03"   # deep brown — directed counterpart (out-degree clustering)

_PANEL_METRICS = {
    "C":           "clustering coeff.",
    "age_assort":  "age assort.",
    "phq9_assort": "initial PHQ-9 assort.",
    "mean_degree": "mean degree",          # observed-only — no target band
}

# Two degree-4.5 configurations, both rebuilt from the calibrated parameter set:
#   _LOWK_COMBO       — low density only (calibrated alpha; C stays in range)
#   _LOWK_COMBO_LOW_C — low density + low clustering: alpha dropped 2.17→1.17 so the
#                       flatter distance decay wires more long-range edges, pushing
#                       C below the target band (other params unchanged).
# At degree 3 the network typically loses target-range clustering or fragments, so 4.5
# is the strongest density reduction that still preserves connectivity. The low-alpha
# combo is also plotted at degree 3 (_LOWK_DEGREE_3) to show that breakdown directly.
_LOWK_COMBO       = [2.1655, 2, 7.9839, 5, 2.3149]   # alpha, n_clusters, latent_w, dim, age_w
_LOWK_COMBO_LOW_C = [1.1655, 2, 7.9839, 5, 2.3149]   # as above, alpha lowered for low C
_LOWK_DEGREE   = 4.5
_LOWK_DEGREE_3 = 3.0
_LOWK_SEEDS  = [14, 15, 16, 17, 18]
_WELL_BEING  = "data/confidential/phq9.sav"

# High-PHQ-9-assortativity probe. The PHQ-9 axis carries a fixed unit weight, so it
# dominates when the latent dims are few and weak: drop dim 5→3 (one latent dim
# instead of three) and latent_weight 7.98→1.0. This lifts initial PHQ-9
# assortativity well above its target band — a knob check, not a calibrated config.
_HIGHPHQ_COMBO  = [2.1655, 2, 1.0, 3, 2.3149]   # alpha, n_clusters, latent_w↓, dim↓, age_w
_HIGHPHQ_DEGREE = 4.5

_RUNS_BASE  = "data/networks_post/basis/sda/undirected"
_K6_RUNS    = {"high\ndegree": "old_debiased/2_1655_d6_dim5"}   # saved k=6 run (C in band; renamed from debiased/)

# meta.json topology keys per panel metric
_META_KEYS = {"C": "clustering", "age_assort": "age_assort",
              "phq9_assort": "phq9_assort_initial", "mean_degree": "mean_degree"}


def _lowk_points(combo: list[float],
                 degree: float = _LOWK_DEGREE) -> dict[str, list[tuple[float, bool]]]:
    """Rebuild a low-degree configuration from ``combo`` at ``degree`` and measure it.

    No LLM involved. One realization per seed; each carries its own fragmentation
    flag (lcc < LCC_WARN) so the open-marker overlay reflects that realization only.
    """
    import utils.tools.load_personas as lp
    from utils.sensitivity.sa_network import _eval_one

    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    wb = lp.load_phq9(_WELL_BEING, 100, seed=_LOWK_SEEDS[0])
    points = {m: [] for m in _PANEL_METRICS}
    for seed in _LOWK_SEEDS:
        r = _eval_one(combo, wb, 100, degree, seed, "gaussian_clusters", src_path)
        fragmented = r["lcc"] < LCC_WARN
        for m in _PANEL_METRICS:
            points[m].append((float(r[m]), fragmented))
    return points


def _lowk_directed_points(combo: list[float],
                          degree: float = _LOWK_DEGREE
                          ) -> dict[str, list[tuple[float, bool]]]:
    """Per-metric (value, fragmented) points for the DIRECTED counterpart.

    Same configuration as :func:`_lowk_points` (same combo, degree and seeds),
    rebuilt with ``directed=True``; every metric is measured on the directed graph
    (clustering = out-clustering, degree-based metrics use the out-degree — see
    ``directed_metrics``). The flag is weak-component fragmentation (lcc < LCC_WARN).
    Used for the saved k=6 config too — those runs are undirected-only, so its
    directed counterpart is rebuilt from the calibrated parameters at degree 6.
    """
    import utils.tools.load_personas as lp
    from utils.sensitivity.sa_network import directed_metrics

    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    wb = lp.load_phq9(_WELL_BEING, 100, seed=_LOWK_SEEDS[0])
    points = {m: [] for m in _PANEL_METRICS}
    for seed in _LOWK_SEEDS:
        r = directed_metrics(combo, wb, 100, degree, seed,
                             "gaussian_clusters", src_path)
        fragmented = r["lcc"] < LCC_WARN
        for m in _PANEL_METRICS:
            points[m].append((float(r[m]), fragmented))
    return points


def _k6_points(run: str) -> dict[str, list[tuple[float, bool]]]:
    """Per-metric (value, fragmented) tuples from the saved simulation runs."""
    points = {m: [] for m in _PANEL_METRICS}
    for path in sorted(glob.glob(f"{_RUNS_BASE}/{run}/rounds300_N100/seed_*/meta.json")):
        with open(path, encoding="utf-8") as f:
            topo = json.load(f)["topology"]
        fragmented = topo.get("lcc_frac", 1.0) < LCC_WARN
        for m, key in _META_KEYS.items():
            points[m].append((float(topo[key]), fragmented))
    return points


def plot_network_targets(out: str) -> None:
    # (label, points). Canonical config names (table); degree is panel (d) and
    # varies between configs. The two low-alpha configs differ by degree: k=3 is
    # "low degree (low C)", k=4.5 is "low C".
    configs = [
        ("low degree\n(low C)", _lowk_points(_LOWK_COMBO_LOW_C, degree=_LOWK_DEGREE_3)),  # k=3
        ("low C",      _lowk_points(_LOWK_COMBO_LOW_C)),                     # k=4.5
        ("calibrated", _lowk_points(_LOWK_COMBO)),                          # k=4.5
    ]
    for label, run in _K6_RUNS.items():
        configs.append((label, _k6_points(run)))                            # high degree, k=6
    configs.append(("high\nPHQ-9$_\\rho$",
                    _lowk_points(_HIGHPHQ_COMBO, degree=_HIGHPHQ_DEGREE)))   # k=4.5

    # Directed counterpart of each config, aligned to ``configs`` — measured on the
    # directed graph (out-degree throughout) and overlaid in every panel. The k=6
    # entry has no saved directed graph, so it is rebuilt from the calibrated combo
    # at degree 6.
    directed = [
        _lowk_directed_points(_LOWK_COMBO_LOW_C, degree=_LOWK_DEGREE_3),  # low degree (low C), k=3
        _lowk_directed_points(_LOWK_COMBO_LOW_C),                         # low C, k=4.5
        _lowk_directed_points(_LOWK_COMBO),                              # calibrated, k=4.5
        _lowk_directed_points(_LOWK_COMBO, degree=6.0),                  # high degree, k=6 (rebuilt)
        _lowk_directed_points(_HIGHPHQ_COMBO, degree=_HIGHPHQ_DEGREE),   # high PHQ-9, k=4.5
    ]

    ncols = 2
    nrows = -(-len(_PANEL_METRICS) // ncols)        # ceil → 2×2 for four panels
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 1.75 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(_PANEL_METRICS):]:           # hide any unused cells
        ax.set_visible(False)

    frag_seen = band_labeled = False
    for k, (ax, (metric, panel_label)) in enumerate(zip(axes, _PANEL_METRICS.items())):
        ref = REF_RANGES.get(metric)                # mean_degree has none → no band
        if ref and ref[0] is not None and ref[1] is not None:
            ax.axhspan(ref[0], ref[1], color=_COL_BAND, alpha=0.25, zorder=1,
                       label="target range" if not band_labeled else None)
            band_labeled = True

        for x, (label, points) in enumerate(configs):
            vals  = np.array([v for v, _ in points[metric]], dtype=float)
            frag  = np.array([f for _, f in points[metric]], dtype=bool)
            if vals.size == 0:
                continue
            ax.scatter(np.full(vals[~frag].shape, x), vals[~frag], s=9, color="#7f7f7f",
                       alpha=0.8, linewidths=0, zorder=3,
                       label="realization" if (k == 0 and x == 0) else None)
            if frag.any():
                ax.scatter(np.full(vals[frag].shape, x), vals[frag], s=12, facecolors="none",
                           edgecolors="#7f7f7f", linewidths=0.9, zorder=3,
                           label=None if frag_seen else f"lcc $<$ {LCC_WARN}")
                frag_seen = True
            ax.errorbar(x, vals.mean(), yerr=vals.std(), fmt="D", markersize=4,
                        color=_COL_MEAN, capsize=3, elinewidth=1.0, zorder=4,
                        label="mean $\\pm$ SD" if (k == 0 and x == 0) else None)

        for x, dpts in enumerate(directed):          # directed counterpart (out-degree)
            dvals = np.array([v for v, _ in dpts[metric]], dtype=float)
            dvals = dvals[~np.isnan(dvals)]
            if dvals.size == 0:
                continue
            dx   = 0.30                               # offset from the undirected marker
            ax.scatter(np.full(dvals.shape, x + dx), dvals, s=9, color=_COL_DIR,
                       alpha=0.8, linewidths=0, zorder=3)
            ax.errorbar(x + dx, dvals.mean(), yerr=dvals.std(), fmt="D",
                        markersize=4, color=_COL_DIR, capsize=3, elinewidth=1.0,
                        zorder=4,
                        label="directed (out-deg.)" if (k == 0 and x == 0) else None)

        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([lbl for lbl, _ in configs], fontsize=6)
        ax.set_xlim(-0.5, len(configs) - 0.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        ax.text(0.5, -0.36, f"({'abcdef'[k]}) {panel_label}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h; labels += l
    fig.legend(handles, labels, ncol=len(labels), loc="upper center",
               bbox_to_anchor=(0.5, 1.06), fontsize=6.5, framealpha=0.85)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_network_targets] → {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/sensitivity/network_target_ranges.png")
    args = p.parse_args()
    plot_network_targets(args.out)


if __name__ == "__main__":
    main()
