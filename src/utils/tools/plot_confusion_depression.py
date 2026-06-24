#!/usr/bin/env python
"""Depression-class confusion matrices on the seed-35 BERT test set.

Compares the two PHQ-9 inference methods, both evaluated on the *same*
1125-block test set (``bert_regression/test_blocks_seed35.csv``):

  (a) best optimized prompt for PHQ-9 inferencing  -> lowest test MAE across the
      ``optimized_phq9`` seeds (seed 23, MAE 3.71)
  (b) BERT+MLP regressor, seed 35                  -> ``bert_regression``

Continuous PHQ-9 (0-27) is binned into the 5 standard severity classes and a
row-normalized (per-true-class) confusion matrix is drawn for each method,
matching the prompt-SA heatmap style (Blues cmap, white gridlines, square
cells). Per-class precision / recall / F1 (+ macro / weighted) are printed and
written to CSV.

Run:  .venv_vllm/bin/python src/utils/tools/plot_confusion_depression.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# 5-band PHQ-9 severity split (matches sa_analyze.PHQ9_BANDS).
PHQ9_BANDS = [
    (0,  4,  "Minimal"),
    (5,  9,  "Mild"),
    (10, 14, "Moderate"),
    (15, 19, "Mod. Severe"),
    (20, 27, "Severe"),
]
BAND_LABELS = [b[2] for b in PHQ9_BANDS]
# Short tick labels for the figure (data/CSV keep the full BAND_LABELS).
DISPLAY_LABELS = ["Minimal", "Mild", "Mod.", "Mod. Sev.", "Sev."]
BAND_EDGES = [b[1] for b in PHQ9_BANDS[:-1]]  # upper edges -> np.digitize bins
PHQ9_MAX = PHQ9_BANDS[-1][1]

# (panel label, raw-scores path) for the headline confusion-matrix figure:
# best PHQ-9 prompt (seed 23, lowest MAE) vs BERT+MLP seed 35, both on the
# seed-35 BERT test set.
METHODS = [
    ("Optimized prompt (LLM)",
     "data/test_post/optimized_phq9/Qwen3.5-27B_seed23/eval_on_test_blocks_seed35/test_raw_scores.csv"),
    ("BERT+MLP",
     "data/test_post/bert_regression/Qwen3.5-27B_seed35/test_raw_scores.csv"),
]

# Per-seed metrics. IMPORTANT — different test sets per group:
#   * BERT seeds each have their OWN held-out test split (test_blocks_seed{N});
#     each run's test_raw_scores.csv already holds preds on that own split.
#   * Prompt seeds are ALL evaluated on the single BERT seed-35 test set
#     (eval_on_test_blocks_seed35), so they are mutually comparable.
BERT_SEEDS = [34, 35, 36, 37, 38]
PROMPT_SEEDS = [23, 24, 25, 32, 33]


def bert_path(seed: int) -> str:
    return f"data/test_post/bert_regression/Qwen3.5-27B_seed{seed}/test_raw_scores.csv"


def prompt_path(seed: int) -> str:
    return (f"data/test_post/optimized_phq9/Qwen3.5-27B_seed{seed}/"
            "eval_on_test_blocks_seed35/test_raw_scores.csv")


OUT_DIR = os.path.join(REPO, "data", "test_post", "method_comparison")
FIG_PATH = os.path.join(OUT_DIR, "confusion_depression_classes.png")
METRICS_PATH = os.path.join(OUT_DIR, "confusion_depression_metrics.csv")
# Panel (c): S-BERT cosine conditioning matrix produced by
# plot_sbert_cosine_conditioning.py (run that first; embeddings are cached).
COSINE_MATRIX_PATH = os.path.join(OUT_DIR, "sbert_cosine_conditioning_seed35.csv")


def to_band(scores: np.ndarray) -> np.ndarray:
    """Map (possibly out-of-range / float) PHQ-9 scores to band indices 0-4."""
    clipped = np.clip(np.rint(scores), 0, PHQ9_MAX)
    return np.digitize(clipped, BAND_EDGES, right=True)


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(REPO, path))
    return to_band(df["true_phq9"].values), to_band(df["pred_phq9"].values)


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  method: str, test_set: str = "") -> pd.DataFrame:
    labels = list(range(len(BAND_LABELS)))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    rows = [
        {"method": method, "test_set": test_set, "class": BAND_LABELS[i],
         "support": int(support[i]), "precision": p[i], "recall": r[i], "f1": f1[i]}
        for i in labels
    ]
    for avg in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0)
        rows.append({"method": method, "test_set": test_set, "class": f"{avg} avg",
                     "support": int(support.sum()),
                     "precision": pa, "recall": ra, "f1": fa})
    acc = float((y_true == y_pred).mean())
    rows.append({"method": method, "test_set": test_set, "class": "accuracy",
                 "support": int(support.sum()),
                 "precision": acc, "recall": acc, "f1": acc})
    return pd.DataFrame(rows)


def all_seed_metrics() -> pd.DataFrame:
    """Per-class + macro/weighted P/R/F1 for every BERT and prompt seed."""
    frames = []
    for s in PROMPT_SEEDS:
        yt, yp = load(prompt_path(s))
        frames.append(class_metrics(yt, yp, f"Prompt seed {s}", "BERT seed-35 set"))
    for s in BERT_SEEDS:
        yt, yp = load(bert_path(s))
        frames.append(class_metrics(yt, yp, f"BERT seed {s}", f"seed-{s} set (own)"))
    return pd.concat(frames, ignore_index=True)


def compact_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per run: accuracy + macro/weighted P/R/F1 + n."""
    rows = []
    for method in metrics["method"].unique():
        sub = metrics[metrics["method"] == method]
        mac = sub[sub["class"] == "macro avg"].iloc[0]
        wgt = sub[sub["class"] == "weighted avg"].iloc[0]
        acc = sub[sub["class"] == "accuracy"].iloc[0]
        rows.append({
            "run": method, "test_set": sub["test_set"].iloc[0], "n": int(acc["support"]),
            "accuracy": acc["f1"],
            "macro_P": mac["precision"], "macro_R": mac["recall"], "macro_F1": mac["f1"],
            "weighted_F1": wgt["f1"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(BAND_LABELS)

    # Two rows of square heatmaps with two colorbars:
    #   top    (a,b) Blues confusion matrices share one [0,1] colorbar;
    #   bottom (c)   Oranges S-BERT cosine conditioning gets its own colorbar.
    # Subplot (square) sizes are unchanged from the old single-row layout; the
    # figure is just taller so the orange panel sits on its own row below.
    # Width is scaled so every column keeps the SAME absolute inch width as the
    # old single-row figure (~1.42" per ratio-unit: 5.2/3.66 then -> 3.7/2.6),
    # which keeps the square heatmaps exactly the original size.
    fig = plt.figure(figsize=(3.7, 4.1))
    # Per row: two/one tightly-packed square heatmaps, then a short colorbar on
    # the right. Spacer columns/row (with wspace/hspace=0) set the gaps. A blank
    # middle row separates the two heatmap rows, sized to hold the top row's
    # x tick labels + "Predicted class" + (a)/(b) captions without overlap.
    gs = fig.add_gridspec(
        3, 7,
        width_ratios=[1, 0.06, 1, 0.10, 0.05, 0.34, 0.05],
        height_ratios=[1, 0.55, 1],
        wspace=0.0, hspace=0.0,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 2], sharey=ax_a)
    ax_c = fig.add_subplot(gs[2, 0])
    # Colorbars are thin insets placed flush to the right of each row's
    # rightmost heatmap (Blues beside (b), Oranges beside (c)), each spanning
    # the full plot height; they are created with the panels below. The wide
    # right-hand gridspec columns just reserve room (trimmed by bbox="tight").

    for ax, (title, path), tag in zip((ax_a, ax_b), METHODS, "ab"):
        y_true, y_pred = load(path)

        # Row-normalized (per-true-class) confusion matrix -> proportions.
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        row_sums = cm.sum(axis=1, keepdims=True)
        prop = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float),
                         where=row_sums > 0)

        first = tag == "a"
        # Annotate each cell with the proportion only (counts removed). Both
        # panels keep the same y tick marks (shared axis); only the left panel
        # shows the class labels.
        sns.heatmap(
            prop, ax=ax, vmin=0.0, vmax=1.0,
            xticklabels=DISPLAY_LABELS, yticklabels=DISPLAY_LABELS,
            annot=True, fmt=".2f", annot_kws={"fontsize": 5.5},
            cmap="Blues", linewidths=0.4, linecolor="white",
            cbar=False, square=True,
        )
        ax.tick_params(axis="x", rotation=35, labelsize=5, length=2)
        ax.tick_params(axis="y", rotation=0, labelsize=5, length=2,
                       labelleft=first)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.set_xlabel("Predicted class", fontsize=6)
        if first:
            ax.set_ylabel("True class", fontsize=6)
        ax.text(0.5, -0.52, f"({tag}) {title}",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.5)

    # Blues colorbar shared by (a) and (b): a thin inset flush to the right of
    # (b), as tall as the heatmap (transAxes y 0->1 spans the square box).
    cbar_blues = ax_b.inset_axes([1.04, 0.0, 0.05, 1.0])
    sm_b = plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 1.0),
                                 cmap=plt.get_cmap("Blues"))
    cb_b = fig.colorbar(sm_b, cax=cbar_blues)
    cb_b.set_label("proportion of true class", fontsize=5.5)
    cbar_blues.tick_params(labelsize=5)

    # ----- Panel (c): S-BERT cosine conditioning (Oranges) -----
    cos = pd.read_csv(COSINE_MATRIX_PATH, index_col=0).loc[BAND_LABELS, BAND_LABELS]
    cmat = cos.values
    vmin, vmax = float(np.nanmin(cmat)), float(np.nanmax(cmat))
    # Symmetric matrix: rows == columns == bands. On its own bottom row it now
    # shows y tick labels too (both axes identify the bands).
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
    ax_c.text(0.5, -0.52, "(c) S-BERT Cosim",
              transform=ax_c.transAxes, ha="center", va="top", fontsize=6.5)

    # Oranges colorbar: a thin inset flush to the right of (c), as tall as the
    # heatmap.
    cbar_oranges = ax_c.inset_axes([1.04, 0.0, 0.05, 1.0])
    sm_o = plt.cm.ScalarMappable(norm=plt.Normalize(vmin - 0.01, vmax + 0.01),
                                 cmap=plt.get_cmap("Oranges"))
    cb_o = fig.colorbar(sm_o, cax=cbar_oranges)
    cb_o.set_label("cosine similarity", fontsize=5.5)
    cbar_oranges.tick_params(labelsize=5)

    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {FIG_PATH}")

    # Per-seed precision/recall/F1 for every BERT and prompt run.
    metrics = all_seed_metrics()
    metrics.to_csv(METRICS_PATH, index=False)
    print(f"[csv ] {METRICS_PATH}\n")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print("=== per-run summary (macro / weighted / accuracy) ===")
        print(compact_summary(metrics).to_string(index=False))
        print()
        for title in ("Prompt seed 23", "BERT seed 35"):
            print(f"=== {title}  (per-class, used in the figure) ===")
            sub = metrics[metrics.method == title].drop(columns=["method", "test_set"])
            print(sub.to_string(index=False))
            print()


if __name__ == "__main__":
    main()
