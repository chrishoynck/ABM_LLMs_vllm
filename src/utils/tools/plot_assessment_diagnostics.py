"""Appendix diagnostics on the seed-35 BERT test set: S-BERT band cosines + confusion matrices.

Subcommands over the same 1125-block test set (`assessors/bert/teacher/test_blocks_seed35.csv`):
`sbert-cosine` writes the 5x5 PHQ-9-band cosine matrix of plain S-BERT block embeddings
(the first run encodes ~15 min on CPU, then it is cached); `confusion` draws row-normalised
5-band confusion matrices for the best optimized prompt (seed 23) and the BERT+MLP
regressor (seed 35), with the cosine matrix as panel (c), and writes per-class P/R/F1;
`all` runs both in that order. Outputs: data/test_post/method_comparison/. Run: see src/README.md.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO, "data", "test_post", "method_comparison")

# 5-band PHQ-9 severity split (matches sa_analyze.PHQ9_BANDS).
PHQ9_BANDS = [
    (0,  4,  "Minimal"),
    (5,  9,  "Mild"),
    (10, 14, "Moderate"),
    (15, 19, "Mod. Severe"),
    (20, 27, "Severe"),
]
BAND_LABELS = [b[2] for b in PHQ9_BANDS]
DISPLAY_LABELS = ["Minimal", "Mild", "Mod.", "Mod. Sev.", "Sev."]  # short tick labels
BAND_EDGES = [b[1] for b in PHQ9_BANDS[:-1]]  # upper edges -> np.digitize bins
PHQ9_MAX = PHQ9_BANDS[-1][1]


def to_band(scores: np.ndarray) -> np.ndarray:
    """Map PHQ-9 scores (float or out of range allowed) to band indices 0-4."""
    clipped = np.clip(np.rint(scores), 0, PHQ9_MAX)
    return np.digitize(clipped, BAND_EDGES, right=True)


# --------------------------------------------------------------------------- #
# sbert-cosine
# --------------------------------------------------------------------------- #
CSV_PATH = os.path.join(REPO, "data/assessors/bert/teacher/test_blocks_seed35.csv")
MATRIX_PATH = os.path.join(OUT_DIR, "sbert_cosine_conditioning_seed35.csv")
CACHE_PATH = os.path.join(OUT_DIR, "sbert_blocks_seed35.npz")

INVALID = {"", "NO_POST", "NO_TWEET"}
N_SPLITS = 50          # random 2-way splits averaged for the within-band diagonal
RNG_SEED = 35


def parse_blocks(csv_path: str):
    """Group the posts CSV into per-agent blocks (list of posts) + PHQ-9 label.

    Same rule as `prompt_optimizer.parse_tweets_with_phq9_csv` (consecutive rows
    with the same agent_id and phq9 form a block), inlined so this CPU tool does
    not import the vLLM-heavy module.

    Args:
        csv_path (str): tweets_with_phq9-style CSV.

    Returns:
        tuple[list[list[str]], np.ndarray]: blocks and their PHQ-9 labels.
    """
    df = pd.read_csv(csv_path)
    df["tweet"] = df["tweet"].fillna("").astype(str)
    blocks, labels = [], []
    key = (df["agent_id"].astype(str) + "|" + df["phq9"].astype(str))
    run_id = (key != key.shift()).cumsum()
    for _, grp in df.groupby(run_id, sort=False):
        posts = [t for t in grp["tweet"] if t.strip() and t.upper() not in INVALID]
        if not posts:
            continue
        blocks.append(posts)
        labels.append(int(grp["phq9"].iloc[0]))
    return blocks, np.array(labels)


def block_embeddings(blocks, device: str, batch_size: int) -> np.ndarray:
    """Mean-pooled S-BERT (all-MiniLM-L6-v2) embedding per block, one row each."""
    from utils.metrics import generate_sbert_model  # slow import; only needed when encoding
    model = generate_sbert_model(mentalbert=False, device=device)

    flat, offsets = [], [0]
    for posts in blocks:
        flat.extend(posts)
        offsets.append(len(flat))
    print(f"[sbert] encoding {len(flat)} posts over {len(blocks)} blocks "
          f"(device={device}) ...", flush=True)
    emb = model.encode(flat, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=False)
    return np.stack([emb[offsets[i]:offsets[i + 1]].mean(axis=0)
                     for i in range(len(blocks))])


def conditioning_matrix(emb: np.ndarray, band_idx: np.ndarray) -> np.ndarray:
    """5x5 band-vs-band mean cosine; the diagonal uses random 2-way splits.

    Off-diagonal cells are the mean cosine over all cross-band pairs. Diagonal
    cells split the band into two random halves and take the mean cosine between
    the halves, averaged over N_SPLITS splits, so within-band similarity is not
    the trivial 1.0.

    Args:
        emb (np.ndarray): one embedding row per block.
        band_idx (np.ndarray): band index 0-4 per block.

    Returns:
        np.ndarray: 5x5 matrix, nan where a band has too few blocks.
    """
    n = len(BAND_LABELS)
    X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    by_band = [X[band_idx == b] for b in range(n)]
    sums = [g.sum(axis=0) for g in by_band]
    counts = [len(g) for g in by_band]

    mat = np.full((n, n), np.nan)
    rng = np.random.default_rng(RNG_SEED)
    for i in range(n):
        for j in range(n):
            if i != j:
                if counts[i] and counts[j]:
                    mat[i, j] = float(sums[i] @ sums[j]) / (counts[i] * counts[j])
                continue
            g = by_band[i]
            if counts[i] < 2:
                continue
            vals = []
            for _ in range(N_SPLITS):
                perm = rng.permutation(counts[i])
                h = counts[i] // 2
                a, b = g[perm[:h]], g[perm[h:]]
                vals.append(float(a.sum(axis=0) @ b.sum(axis=0)) / (len(a) * len(b)))
            mat[i, i] = float(np.mean(vals))
    return mat


def run_sbert_cosine(device: str = "cpu", batch_size: int = 128, refresh: bool = False) -> None:
    """Encode the test blocks (or load the cache) and write the cosine matrix CSV.

    The standalone heatmap (sbert_cosine_conditioning_seed35.png) was retired
    2026-09-22. The CSV stays: `run_confusion` draws it as panel (c) of
    confusion_depression_classes.png, and the PNAS figure_scripts/ read it.

    Args:
        device (str): torch device for the encoder.
        batch_size (int): encoder batch size.
        refresh (bool): re-encode even if the cache exists.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(CACHE_PATH) and not refresh:
        cache = np.load(CACHE_PATH)
        emb, band_idx = cache["emb"], cache["band_idx"]
        print(f"[cache] loaded {emb.shape[0]} block embeddings from {CACHE_PATH}")
    else:
        blocks, phq9 = parse_blocks(CSV_PATH)
        emb = block_embeddings(blocks, device, batch_size)
        band_idx = to_band(phq9)
        np.savez(CACHE_PATH, emb=emb, band_idx=band_idx, phq9=phq9)
        print(f"[cache] wrote {emb.shape[0]} block embeddings -> {CACHE_PATH}")

    counts = np.bincount(band_idx, minlength=len(BAND_LABELS))
    print("[bands] " + ", ".join(f"{l}={c}" for l, c in zip(BAND_LABELS, counts)))

    mat = conditioning_matrix(emb, band_idx)
    pd.DataFrame(mat, index=BAND_LABELS, columns=BAND_LABELS).to_csv(MATRIX_PATH)
    print(f"[csv ] {MATRIX_PATH}")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(pd.DataFrame(mat, index=BAND_LABELS, columns=BAND_LABELS).to_string())
    adj = [(f"{BAND_LABELS[i]}->{BAND_LABELS[i+1]}", mat[i, i + 1])
           for i in range(len(BAND_LABELS) - 1)]
    print("[adjacent-band] " + ", ".join(f"{k} {v:.3f}" for k, v in adj))


# --------------------------------------------------------------------------- #
# confusion
# --------------------------------------------------------------------------- #
# (panel label, raw-scores path): best PHQ-9 prompt (seed 23, lowest MAE) vs
# BERT+MLP seed 35, both scored on the seed-35 BERT test set.
METHODS = [
    ("Optimized prompt (LLM)",
     "data/test_post/optimized_phq9/Qwen3.5-27B_seed23/eval_on_test_blocks_seed35/test_raw_scores.csv"),
    ("BERT+MLP",
     "data/assessors/bert/teacher/models/Qwen3.5-27B_seed35/test_raw_scores.csv"),
]
# Per-seed metrics use different test sets per group: each BERT seed has its own
# held-out split; all prompt seeds are scored on the seed-35 BERT test set.
BERT_SEEDS = [34, 35, 36, 37, 38]
PROMPT_SEEDS = [23, 24, 25, 32, 33]

CONF_FIG_PATH = os.path.join(OUT_DIR, "confusion_depression_classes.png")
METRICS_PATH = os.path.join(OUT_DIR, "confusion_depression_metrics.csv")


def bert_path(seed: int) -> str:
    """Raw-scores CSV of one BERT+MLP seed on its own test split."""
    return f"data/assessors/bert/teacher/models/Qwen3.5-27B_seed{seed}/test_raw_scores.csv"


def prompt_path(seed: int) -> str:
    """Raw-scores CSV of one prompt seed scored on the seed-35 BERT test set."""
    return (f"data/test_post/optimized_phq9/Qwen3.5-27B_seed{seed}/"
            "eval_on_test_blocks_seed35/test_raw_scores.csv")


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a raw-scores CSV and return (true bands, predicted bands)."""
    df = pd.read_csv(os.path.join(REPO, path))
    return to_band(df["true_phq9"].values), to_band(df["pred_phq9"].values)


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  method: str, test_set: str = "") -> pd.DataFrame:
    """Per-class precision / recall / F1 plus macro, weighted and accuracy rows."""
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


def run_confusion() -> None:
    """Draw the 3-panel confusion figure and write the per-seed metrics CSV."""
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(BAND_LABELS)

    # Top row: (a,b) Blues confusion matrices with one shared colourbar; bottom
    # row: (c) Oranges S-BERT cosine matrix with its own colourbar. The spacer
    # columns/row only set the gaps; the wide right columns reserve room that
    # bbox="tight" trims again.
    fig = plt.figure(figsize=(3.7, 4.1))
    gs = fig.add_gridspec(
        3, 7,
        width_ratios=[1, 0.06, 1, 0.10, 0.05, 0.34, 0.05],
        height_ratios=[1, 0.55, 1],
        wspace=0.0, hspace=0.0,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 2], sharey=ax_a)
    ax_c = fig.add_subplot(gs[2, 0])

    for ax, (title, path), tag in zip((ax_a, ax_b), METHODS, "ab"):
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
        ax.tick_params(axis="y", rotation=0, labelsize=5, length=2,
                       labelleft=first)
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

    # Panel (c): the S-BERT cosine matrix written by run_sbert_cosine.
    cos = pd.read_csv(MATRIX_PATH, index_col=0).loc[BAND_LABELS, BAND_LABELS]
    cmat = cos.values
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
    ax_c.text(0.5, -0.52, "(c) S-BERT Cosim",
              transform=ax_c.transAxes, ha="center", va="top", fontsize=6.5)

    cbar_oranges = ax_c.inset_axes([1.04, 0.0, 0.05, 1.0])
    sm_o = plt.cm.ScalarMappable(norm=plt.Normalize(vmin - 0.01, vmax + 0.01),
                                 cmap=plt.get_cmap("Oranges"))
    cb_o = fig.colorbar(sm_o, cax=cbar_oranges)
    cb_o.set_label("cosine similarity", fontsize=5.5)
    cbar_oranges.tick_params(labelsize=5)

    fig.savefig(CONF_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {CONF_FIG_PATH}")

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


def main(argv=None) -> None:
    """Parse the subcommand and run it."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("sbert-cosine", "all"):
        s = sub.add_parser(name)
        s.add_argument("--device", default="cpu")
        s.add_argument("--batch-size", type=int, default=128)
        s.add_argument("--refresh", action="store_true", help="Re-encode even if a cache exists.")
    sub.add_parser("confusion")
    a = p.parse_args(argv)

    if a.cmd in ("sbert-cosine", "all"):
        run_sbert_cosine(a.device, a.batch_size, a.refresh)
    if a.cmd in ("confusion", "all"):
        run_confusion()


if __name__ == "__main__":
    main()
