"""Two-panel PHQ-9 conditioning heatmap: Qwen (left) vs Gemma (right).

Same panel as (b) of agent_phq9_combined.png, drawn twice with the same figure
geometry, fonts and colorbar docking as that figure. Each panel keeps its OWN
colour scale and colorbar, so colours are comparable within a panel, not across.
The diagonal is the within-band cosine (same band, different replicate), the
off-diagonal is the paired same-persona cosine between two bands.

Usage:
    ./.venv_vllm/bin/python data/sensitivity/plot_phq9_conditioning_panels.py
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = "data/sensitivity/seeded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default=f"{ROOT}/qwen/plots_sbert/phq9_conditioning_matrix.csv")
    ap.add_argument("--right", default=f"{ROOT}/gemma4/plots_sbert/phq9_conditioning_matrix.csv")
    ap.add_argument("--left-title", default="Qwen3.5-27B")
    ap.add_argument("--right-title", default="Gemma-4-31B-it")
    ap.add_argument("--cmap", default="Oranges")
    ap.add_argument("--out", default=f"{ROOT}/plots_sbert/phq9_conditioning_panels.png")
    args = ap.parse_args()

    mats = [pd.read_csv(p, index_col=0) for p in (args.left, args.right)]
    titles = [args.left_title, args.right_title]

    # Geometry follows plot_agent_phq9_combined but drawn smaller, so that at a
    # fixed \linewidth the annotations and tick labels come out proportionally
    # larger. The figure stays wide relative to its height only to open the gap
    # between panels: at a tighter wspace the left colorbar label lands on the
    # right panel's tick labels.
    fig = plt.figure(figsize=(5.8, 1.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.8)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    for ax, mat, title, tag in zip(axes, mats, titles, "ab"):
        vmin = float(np.nanmin(mat.values)) - 0.01
        vmax = float(np.nanmax(mat.values)) + 0.01
        sns.heatmap(
            mat.values, ax=ax,
            xticklabels=list(mat.index), yticklabels=list(mat.index),
            vmin=vmin, vmax=vmax,
            annot=True, fmt=".3f", annot_kws={"fontsize": 6},
            cmap=args.cmap, linewidths=0.4, linecolor="white",
            cbar=False, square=True,
        )
        ax.tick_params(axis="x", labelsize=7.5)
        # Anchored right-alignment keeps the longer band names from running into
        # each other now that the panels are drawn smaller.
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="y", rotation=0, labelsize=7.5)

        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="4%", pad=0.08)
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax),
                                   cmap=plt.get_cmap(args.cmap))
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("cosine similarity", fontsize=8)
        cbar_ax.tick_params(labelsize=7)

        ax.text(0.5, -0.42, f"({tag}) {title}", transform=ax.transAxes,
                ha="center", va="top", fontsize=10)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {args.out}")


if __name__ == "__main__":
    main()
