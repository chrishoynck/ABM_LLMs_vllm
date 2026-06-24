#!/usr/bin/env python
"""S-BERT PHQ-9 conditioning heatmap on the seed-35 BERT test set.

Reproduces the right ("PHQ-9 conditioning") panel of the prompt-SA figure, but
on real test data instead of generated reps:

  * Unit = a block (one agent's 10 posts) from
    ``bert_regression/test_blocks_seed35.csv`` — the seed-35 BERT test set.
  * Each block is embedded with plain S-BERT (all-MiniLM-L6-v2, i.e.
    ``generate_sbert_model(mentalbert=False)``): mean-pooled over the block's
    valid posts.
  * Blocks are grouped into the 5 PHQ-9 severity bands. Cell (i, j) is the mean
    cosine similarity between every block in band i and every block in band j.
      - off-diagonal: all cross-band pairs (disjoint sets, no self-pairs).
      - diagonal (within-band): the embeddings of the band are randomly split
        into two halves and the mean cosine BETWEEN the halves is taken
        (averaged over several splits). This gives an honest within-band
        similarity instead of the trivial self-cosine of 1.0.

Colour = cosine similarity (Blues), annotation = the value. Saves the matrix to
CSV next to the figure.

Run (encode is ~15 min on CPU the first time; cached afterwards):
    .venv_vllm/bin/python src/utils/tools/plot_sbert_cosine_conditioning.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
BAND_EDGES = [b[1] for b in PHQ9_BANDS[:-1]]

CSV_PATH = os.path.join(REPO, "data/test_post/bert_regression/test_blocks_seed35.csv")
OUT_DIR = os.path.join(REPO, "data", "test_post", "method_comparison")
FIG_PATH = os.path.join(OUT_DIR, "sbert_cosine_conditioning_seed35.png")
MATRIX_PATH = os.path.join(OUT_DIR, "sbert_cosine_conditioning_seed35.csv")
CACHE_PATH = os.path.join(OUT_DIR, "sbert_blocks_seed35.npz")

INVALID = {"", "NO_POST", "NO_TWEET"}
N_SPLITS = 50          # random 2-way splits averaged for the within-band diagonal
RNG_SEED = 35


def parse_blocks(csv_path: str):
    """Group the tweets CSV into per-agent blocks (list of posts) + PHQ-9 label.

    Mirrors prompt_optimizer.parse_tweets_with_phq9_csv (consecutive same
    agent_id/phq9 rows form a block) without importing the heavy module.
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
    """Mean-pooled S-BERT embedding per block (one row each)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

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


def to_band(phq9: np.ndarray) -> np.ndarray:
    return np.digitize(np.clip(phq9, 0, 27), BAND_EDGES, right=True)


def conditioning_matrix(emb: np.ndarray, band_idx: np.ndarray) -> np.ndarray:
    """5x5 band-vs-band mean cosine; diagonal via random 2-way splits."""
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
            # within-band: mean cosine between two random halves, averaged.
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


def plot(mat: np.ndarray) -> None:
    fig = plt.figure(figsize=(3.4, 3.0))
    ax = fig.add_subplot(111)
    vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
    sns.heatmap(
        mat, ax=ax, xticklabels=BAND_LABELS, yticklabels=BAND_LABELS,
        vmin=vmin - 0.01, vmax=vmax + 0.01,
        annot=True, fmt=".3f", annot_kws={"fontsize": 7},
        cmap="Blues", linewidths=0.4, linecolor="white", cbar=False, square=True,
    )
    ax.tick_params(axis="x", rotation=30, labelsize=7.5)
    ax.tick_params(axis="y", rotation=0, labelsize=7.5)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="4%", pad=0.08)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin - 0.01, vmax + 0.01),
                               cmap=plt.get_cmap("Blues"))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("cosine similarity", fontsize=8)
    cbar_ax.tick_params(labelsize=7)
    ax.text(0.5, -0.34, "PHQ-9 conditioning (S-BERT)",
            transform=ax.transAxes, ha="center", va="top", fontsize=10)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {FIG_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--refresh", action="store_true",
                    help="Re-encode even if a cache exists.")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(CACHE_PATH) and not args.refresh:
        cache = np.load(CACHE_PATH)
        emb, band_idx = cache["emb"], cache["band_idx"]
        print(f"[cache] loaded {emb.shape[0]} block embeddings from {CACHE_PATH}")
    else:
        blocks, phq9 = parse_blocks(CSV_PATH)
        emb = block_embeddings(blocks, args.device, args.batch_size)
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
    # Adjacent-band (super-diagonal) cosine — the "(a)" ladder, for reference.
    adj = [(f"{BAND_LABELS[i]}->{BAND_LABELS[i+1]}", mat[i, i + 1])
           for i in range(len(BAND_LABELS) - 1)]
    print("[adjacent-band] " + ", ".join(f"{k} {v:.3f}" for k, v in adj))
    plot(mat)


if __name__ == "__main__":
    main()
