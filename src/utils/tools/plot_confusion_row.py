"""Confusion matrices as ONE row (LLM + BERT), with the S-BERT band cosine panel for SI Fig S8.

The one-row layout of `plot_assessment_diagnostics confusion` (same colormaps, annotations
and colorbars; Blues for the confusion matrices, Oranges for the cosine panel).

Subcommands:
  s8 OUT_PATH   PNAS SI Fig S8 (`Fig/confusion_matrix_vs_cosim.png`) on the seed-35 teacher
                test set: (a) seed-23 optimized prompt, (b) teacher BERT seed 35, (c) the
                cosine CSV that `plot_assessment_diagnostics sbert-cosine` writes.
  humanopt      the confusion matrices only (no cosine panel) for the human-optimized
                (iter_10) corpus, one PNG per generator, in
                data/test_post/method_comparison/multimodel/.

humanopt panels, all on the shared 300 held-out personas (test_posts_*.csv; same personas
and PHQ-9 in the minimal and human-opt corpora, none in the 3,000 training blocks):
  (a) the minimal LLM assessment prompt, on the human-opt posts (the LLM line of the
      linearity/bias figure; Gemma pools its five runs of that one prompt)
  (b) BERT+MLP fine-tuned on the minimal (iter_0) corpus, on its own minimal posts
  (c) BERT+MLP fine-tuned on the human-opt corpus, on the human-opt posts
Both BERT arms are seed 35. There are no LLM predictions on the 3,000 blocks yet.
Run: see src/README.md (Hand-run CLIs).
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.tools.plot_assessment_diagnostics import (
    BAND_LABELS, DISPLAY_LABELS, MATRIX_PATH, OUT_DIR, REPO, to_band)

# s8: seed-35 teacher test set.
S8_METHODS = [
    ("Optimized prompt (LLM)", os.path.join(
        REPO, "data/test_post/optimized_phq9/Qwen3.5-27B_seed23/eval_on_test_blocks_seed35/test_raw_scores.csv")),
    ("BERT+MLP", os.path.join(
        REPO, "data/assessors/bert/teacher/models/Qwen3.5-27B_seed35/test_raw_scores.csv")),
]

MM_DIR = os.path.join(OUT_DIR, "multimodel")
LLM = os.path.join(REPO, "data/test_post/optimized_phq9/Qwen3.5-27B_seed{s}")
BERT = os.path.join(REPO, "data/assessors/bert")
HUMANOPT = {
    "qwen": dict(llm=[LLM.format(s=23) + "/minimal_human300/test_raw_scores.csv"],
                 bert_min=f"{BERT}/qwen27_minimal/eval/on_qwen27_minimal/seed35.csv",
                 bert=f"{BERT}/qwen27_optimized/eval/on_qwen27_optimized/seed35.csv"),
    "gemma": dict(llm=[LLM.format(s=s) + "/minimal_gemma4300/test_raw_scores.csv"
                       for s in (23, 24, 25, 32, 33)],
                  bert_min=f"{BERT}/gemma4_minimal/eval/on_gemma4_minimal/seed35.csv",
                  bert=f"{BERT}/gemma4_optimized/eval/on_gemma4_optimized/seed35.csv"),
}


def load(path):
    """One prediction CSV, or a list of runs pooled (= the mean of their row proportions)."""
    paths = [path] if isinstance(path, str) else path
    df = pd.concat([pd.read_csv(p) for p in paths])
    return to_band(df["true_phq9"].values), to_band(df["pred_phq9"].values)


def draw_row(methods, cmat, out_path):
    """Draw one confusion matrix per (title, csv) in `methods`, then `cmat` as the cosine
    panel (no cosine panel when `cmat` is None)."""
    n = len(BAND_LABELS)

    # One row: the confusion matrices share the Blues colorbar placed right of the
    # last one; the cosine panel gets its own Oranges colorbar on the far right.
    # ~1.42 in per square heatmap. With two methods this is exactly the S8 layout.
    ratios = [1, 0.06] * (len(methods) - 1) + [1, 0.02, 0.20]
    if cmat is not None:
        ratios += [0.72, 1, 0.02, 0.20]
    fig = plt.figure(figsize=(6.5 * sum(ratios) / 4.22, 1.75))
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.0)
    ax_a = fig.add_subplot(gs[0, 0])
    axes = [ax_a] + [fig.add_subplot(gs[0, 2 * i], sharey=ax_a)
                     for i in range(1, len(methods))]
    ax_b = axes[-1]

    for ax, (title, path), tag in zip(axes, methods, "abcdef"):
        y_true, y_pred = load(path)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        row_sums = cm.sum(axis=1, keepdims=True)
        prop = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float),
                         where=row_sums > 0)
        first = tag == "a"
        sns.heatmap(
            prop, ax=ax, vmin=0.0, vmax=1.0,
            xticklabels=DISPLAY_LABELS, yticklabels=DISPLAY_LABELS,
            annot=True, fmt=".2f", annot_kws={"fontsize": 5.5},
            cmap="Blues", linewidths=0.4, linecolor="white",
            cbar=False, square=True,
        )
        ax.tick_params(axis="x", rotation=35, labelsize=5, length=2)
        ax.tick_params(axis="y", rotation=0, labelsize=5, length=2, labelleft=first)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.set_xlabel("Predicted class", fontsize=6)
        if first:
            ax.set_ylabel("True class", fontsize=6)
        ax.text(0.5, -0.52, f"({tag}) {title}",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.5)

    cbar_blues = ax_b.inset_axes([1.04, 0.0, 0.05, 1.0])
    sm_b = plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 1.0),
                                 cmap=plt.get_cmap("Blues"))
    cb_b = fig.colorbar(sm_b, cax=cbar_blues)
    cb_b.set_label("proportion of true class", fontsize=5.5)
    cbar_blues.tick_params(labelsize=5)

    if cmat is not None:
        ax_c = fig.add_subplot(gs[0, 2 * len(methods) + 2])
        vmin, vmax = float(np.nanmin(cmat)), float(np.nanmax(cmat))
        sns.heatmap(
            cmat, ax=ax_c, vmin=vmin - 0.01, vmax=vmax + 0.01,
            xticklabels=DISPLAY_LABELS, yticklabels=DISPLAY_LABELS,
            annot=True, fmt=".2f", annot_kws={"fontsize": 5.5},
            cmap="Oranges", linewidths=0.4, linecolor="white",
            cbar=False, square=True,
        )
        ax_c.tick_params(axis="x", rotation=35, labelsize=5, length=2)
        ax_c.tick_params(axis="y", rotation=0, labelsize=5, length=2)
        for lbl in ax_c.get_xticklabels():
            lbl.set_ha("right")
        ax_c.set_xlabel("PHQ-9 class", fontsize=6)
        ax_c.set_ylabel("PHQ-9 class", fontsize=6)
        ax_c.text(0.5, -0.52, f"({'abcdef'[len(methods)]}) S-BERT Cosim",
                  transform=ax_c.transAxes, ha="center", va="top", fontsize=6.5)

        cbar_oranges = ax_c.inset_axes([1.04, 0.0, 0.05, 1.0])
        sm_o = plt.cm.ScalarMappable(norm=plt.Normalize(vmin - 0.01, vmax + 0.01),
                                     cmap=plt.get_cmap("Oranges"))
        cb_o = fig.colorbar(sm_o, cax=cbar_oranges)
        cb_o.set_label("cosine similarity", fontsize=5.5)
        cbar_oranges.tick_params(labelsize=5)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def run_s8(out_path: str) -> None:
    """SI Fig S8 on the seed-35 teacher test set."""
    cos = pd.read_csv(MATRIX_PATH, index_col=0).loc[BAND_LABELS, BAND_LABELS]
    draw_row(S8_METHODS, cos.values, out_path)


def run_humanopt() -> None:
    """The 3-panel human-opt row, one PNG per generator."""
    for tag, cfg in HUMANOPT.items():
        methods = [("Minimal prompt (LLM)\nhuman-opt posts", cfg["llm"]),
                   ("BERT+MLP\nminimal FT", cfg["bert_min"]),
                   ("BERT+MLP\nhuman-opt FT", cfg["bert"])]
        draw_row(methods, None, f"{MM_DIR}/confusion_row_humanopt_{tag}.png")


def main(argv=None) -> None:
    """Parse the subcommand and run it."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("s8").add_argument("out_path", help="PNG to write, e.g. the PNAS Fig/ file.")
    sub.add_parser("humanopt")
    a = p.parse_args(argv)
    if a.cmd == "s8":
        run_s8(a.out_path)
    else:
        run_humanopt()


if __name__ == "__main__":
    main()
