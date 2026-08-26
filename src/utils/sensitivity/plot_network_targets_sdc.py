"""Plot SDC (stub-matched, scale-free) network configs against their target ranges.

Companion to ``plot_network_targets.py`` (which covers the plain-SDA mode). SDC
adds stub matching to produce a scale-free degree sequence, so the calibration
targets are different: the degree *distribution* (power-law gamma band, KS fit,
realized mean degree) plus clustering and PHQ-9 assortativity. Age assortativity
is shown alongside as an observed-only metric (no target band).

Panels (3×2 grid)
-----------------
    (a) power-law gamma   target 2.0–3.0
    (b) KS fit            target < 0.10        (acceptable region shaded 0–0.10)
    (c) mean degree       target 4.25–4.75     (GOAL_DEGREE ± DEGREE_TOL)
    (d) clustering coeff. target 0.03–0.20
    (e) PHQ-9 assort.     target 0.0–0.10      (see _BAND_OVERRIDE below)
    (f) age assort.       observed only — no band

Every panel also carries a deep-brown overlay of the DIRECTED counterpart of each
config (the same network built with ``directed=True``), measured on the directed
graph: clustering is the out-clustering (Fagiolo 2007) and the degree-based panels
(a)–(c) use the out-degree. The directed graph is sampled asymmetrically, so all of
these differ from the undirected (blue) values.

A key SDC quirk: the ``degree`` parameter fed to the stub sampler is NOT the
realized mean degree. Stub matching leaves some stubs unpaired, so the realized
mean (panel c) lands well below the target degree fed in — e.g. the saved config
feeds degree=8.25 but realizes ≈4.7. The mean-degree band makes that shortfall
visible.

Configurations (rebuilt from parameters with _eval_one; no LLM)
---------------------------------------------------------------
Labels follow the canonical naming table (not "saved"/arrows), mirroring the
plain-SDA plot.

    calibrated      – the parameter set used for the saved simulation runs
                      (data/networks_post/basis/sdc/.../non_debiased/4_9429_d8_2539_dim3),
                      rebuilt so every metric is measured identically to the
                      variations below; PHQ-9 assortativity lands in the 0–0.10
                      band.
    high degree     – same set with the degree parameter raised (8.25→10) to see
                      how realized degree / clustering / fragmentation respond
                      (exploratory only — not retargeted).
    high PHQ-9$_\rho$ – higher alpha, lower dim (3→2, drops the latent slot) and
                      lower latent_weight so the fixed-weight PHQ-9 axis drives
                      the geometry, lifting PHQ-9 assortativity (ρ) above its band.

These three are first-pass probes ("see what we end up with") before searching
for properly calibrated combinations. Five master seeds (14–18), each reseeding
both the population and the network wiring (matching the saved runs). Read panel (a)
with care: SDC power-law gamma is unstable per realization — powerlaw.Fit
occasionally latches onto a short tail and returns a large exponent (gamma ≈ 14–34),
sometimes on fragmented graphs (lcc < 0.9, open markers) but also on connected ones.
It is a fit artifact, not a real change in the degree distribution.

Usage
-----
    PYTHONPATH=src python -m utils.sensitivity.plot_network_targets_sdc \\
        --out data/sensitivity/network_target_ranges_sdc.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import utils.sensitivity.sa_network as san

_COL_MEAN = "#2e7ebc"   # blue   — mean ± SD
_COL_BAND = "#d96907"   # orange — target range
_COL_DIR  = "#8d2c03"   # deep brown — directed counterpart (out-degree clustering)

# Panel order → 3 columns × 2 rows. Goal metrics carry a target band; age_assort
# is shown alongside (observed only, no band — see _GOAL below).
_PANEL_METRICS = {
    "gamma":       "power-law $\\gamma$",
    "ks":          "KS fit",
    "mean_degree": "mean degree",
    "C":           "clustering coeff.",
    "phq9_assort": "initial PHQ-9 assort.",
    "age_assort":  "age assort.",            # observed-only — no target band
}
_GOAL = {"gamma", "ks", "mean_degree", "C", "phq9_assort"}   # metrics that get a band

# Per-figure band overrides — these do NOT touch the SA loss globals in sa_network.
# The Sobol search used a too-wide 0–0.40 PHQ-9 band (which let the optimiser drift
# to high homophily); 0–0.10 is the reference these configs should be shown against.
_BAND_OVERRIDE = {"phq9_assort": (0.0, 0.10)}

# SDC parameter order for _eval_one(net="sdc"):
#   [alpha, stub_gamma, degree, dim, n_clusters, latent_weight, age_weight]
# Saved set: data/networks_post/basis/sdc/undirected/non_debiased/4_9429_d8_2539_dim3
# (scripts/simulation/run_simulation_sdc.sh / network_sdc/averaged_best.csv).
_SDC_SAVED     = [4.9429, 1.6187, 8.2539, 3, 2, 18.3813, 2.2095]
_SDC_HIGH_DEG  = [4.9429, 1.6187, 10.0,   3, 2, 18.3813, 2.2095]   # degree param ↑ (realizes k≈6)
_SDC_HIGH_PHQ9 = [8.0,    1.6187, 8.2539, 2, 2,  2.0,    2.2095]   # alpha↑ dim↓ latent_w↓

_SDC_SEEDS  = [14, 15, 16, 17, 18]   # per-realization master seeds: each seed reseeds BOTH
                                     # the population and the network wiring, matching the
                                     # saved SDC simulation runs (common.seed = 14..18).
_N_AGENTS   = 100
_WELL_BEING = "data/confidential/phq9.sav"


def _sdc_points(combo: list[float]) -> dict[str, list[tuple[float, bool]]]:
    """Rebuild one SDC configuration across all seeds and measure it.

    No LLM involved. One realization per seed; each carries its own fragmentation
    flag (lcc < LCC_WARN). The degree fed in lives in ``combo`` (index 2); the
    ``degree`` argument to _eval_one is ignored in SDC mode.
    """
    import utils.tools.load_personas as lp

    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    points = {m: [] for m in _PANEL_METRICS}
    for seed in _SDC_SEEDS:
        # Reseed the population per seed (coupled to the network seed) so each
        # realization is a full reseed, exactly like the saved simulation runs.
        wb = lp.load_phq9(_WELL_BEING, _N_AGENTS, seed=seed)
        r = san._eval_one(combo, wb, _N_AGENTS, 0, seed,
                          "gaussian_clusters", src_path, net="sdc")
        fragmented = r["lcc"] < san.LCC_WARN
        for m in _PANEL_METRICS:
            points[m].append((float(r[m]), fragmented))
    return points


def _sdc_directed_points(combo: list[float]) -> dict[str, list[tuple[float, bool]]]:
    """Per-metric (value, fragmented) points for the DIRECTED counterpart.

    Same SDC configuration as :func:`_sdc_points` (reseeding both population and
    wiring per seed), rebuilt with ``directed=True``; every metric is measured on
    the directed graph (clustering = out-clustering, degree-based metrics use the
    out-degree — see ``san.directed_metrics``). The flag is weak-component
    fragmentation (lcc < LCC_WARN).
    """
    import utils.tools.load_personas as lp

    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    points = {m: [] for m in _PANEL_METRICS}
    for seed in _SDC_SEEDS:
        wb = lp.load_phq9(_WELL_BEING, _N_AGENTS, seed=seed)
        r = san.directed_metrics(combo, wb, _N_AGENTS, 0, seed,
                                 "gaussian_clusters", src_path, net="sdc")
        fragmented = r["lcc"] < san.LCC_WARN
        for m in _PANEL_METRICS:
            points[m].append((float(r[m]), fragmented))
    return points


def plot_network_targets_sdc(out: str) -> None:
    san.set_mode("sdc")   # bind san.REF_RANGES to the SDC reference bands

    # (label, points). Canonical config names (table): the saved set is the
    # "calibrated" config, the raised-degree probe is "high degree", and the
    # alpha/dim probe is "high PHQ-9$_\rho$" (ρ = PHQ-9 assortativity).
    configs = [
        ("calibrated",        _sdc_points(_SDC_SAVED)),
        ("high\ndegree",      _sdc_points(_SDC_HIGH_DEG)),
        ("high\nPHQ-9$_\\rho$", _sdc_points(_SDC_HIGH_PHQ9)),
    ]
    # Directed counterpart of each config, aligned to ``configs`` — measured on the
    # directed graph (out-degree throughout) and overlaid in every panel.
    directed = [_sdc_directed_points(combo)
                for combo in (_SDC_SAVED, _SDC_HIGH_DEG, _SDC_HIGH_PHQ9)]

    ncols = 3
    nrows = -(-len(_PANEL_METRICS) // ncols)        # ceil → 2×3 for six panels
    # Wider per-panel width than the SDA plot to fit the directed-counterpart
    # markers offset beside each config.
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.25 * ncols, 1.7 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(_PANEL_METRICS):]:           # hide any unused cells
        ax.set_visible(False)

    frag_seen = band_labeled = False
    for k, (ax, (metric, panel_label)) in enumerate(zip(axes, _PANEL_METRICS.items())):
        ref = _BAND_OVERRIDE.get(metric, san.REF_RANGES.get(metric)) \
            if metric in _GOAL else None
        if ref is not None:
            lo, hi = ref
            lbl = "target range" if not band_labeled else None
            if hi is None:                          # one-sided lower bound → line
                ax.axhline(lo, color=_COL_BAND, ls="--", lw=1.0, alpha=0.8, label=lbl)
            else:                                   # range (KS: lo=None → 0, acceptable 0–hi)
                ax.axhspan(0.0 if lo is None else lo, hi,
                           color=_COL_BAND, alpha=0.25, zorder=1, label=lbl)
            band_labeled = True

        for x, (label, points) in enumerate(configs):
            vals = np.array([v for v, _ in points[metric]], dtype=float)
            frag = np.array([f for _, f in points[metric]], dtype=bool)
            if vals.size == 0:
                continue
            ax.scatter(np.full(vals[~frag].shape, x), vals[~frag], s=9, color="#7f7f7f",
                       alpha=0.8, linewidths=0, zorder=3,
                       label="realization" if (k == 0 and x == 0) else None)
            if frag.any():
                ax.scatter(np.full(vals[frag].shape, x), vals[frag], s=12, facecolors="none",
                           edgecolors="#7f7f7f", linewidths=0.9, zorder=3,
                           label=None if frag_seen else f"lcc $<$ {san.LCC_WARN}")
                frag_seen = True
            ax.errorbar(x, vals.mean(), yerr=vals.std(), fmt="D", markersize=4,
                        color=_COL_MEAN, capsize=3, elinewidth=1.0, zorder=4,
                        label="mean $\\pm$ SD" if (k == 0 and x == 0) else None)

        for x, dpts in enumerate(directed):          # directed counterpart (out-degree)
            dvals = np.array([v for v, _ in dpts[metric]], dtype=float)
            dvals = dvals[~np.isnan(dvals)]
            if dvals.size == 0:
                continue
            ax.scatter(np.full(dvals.shape, x + 0.24), dvals, s=9, color=_COL_DIR,
                       alpha=0.8, linewidths=0, zorder=3)
            ax.errorbar(x + 0.24, dvals.mean(), yerr=dvals.std(), fmt="D",
                        markersize=4, color=_COL_DIR, capsize=3, elinewidth=1.0,
                        zorder=4,
                        label="directed (out-deg.)" if (k == 0 and x == 0) else None)

        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([lbl for lbl, _ in configs], fontsize=6.5)
        ax.set_xlim(-0.5, len(configs) - 0.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        ax.text(0.5, -0.34, f"({'abcdef'[k]}) {panel_label}",
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
    print(f"[plot_network_targets_sdc] → {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/sensitivity/network_target_ranges_sdc.png")
    args = p.parse_args()
    plot_network_targets_sdc(args.out)


if __name__ == "__main__":
    main()
