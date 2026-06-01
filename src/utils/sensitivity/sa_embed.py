"""Encode every SA run's posts.csv with MentalBERT, save one ``embeddings.npz``
per run inside the same dir as the CSV.

Walks ``data/sensitivity/{axis}/setting_*/rep_*/posts.csv`` (or whichever root
you pass in), idempotent on existing ``embeddings.npz``. Encoder is shared
across all runs (loaded once); typical wall-clock on a CPU is a few minutes
for the full 24-run sweep, sub-minute on a GPU.

Each .npz contains:
    embeddings  : (n_posts, 768)  float32     MentalBERT mean-pooled.
    agent_ids   : (n_posts,)      int64       From posts.csv `agent_id` column.
    rounds      : (n_posts,)      int64       From posts.csv `step` column.
    phq9        : (n_posts,)      int64       Persona's true PHQ-9 score.
    texts       : (n_posts,)      object      Raw post text (kept for spot-checks).
A sibling meta.json records encoder, dim, source CSV path.

Usage::

    PYTHONPATH=src python -m utils.sensitivity.sa_embed
    PYTHONPATH=src python -m utils.sensitivity.sa_embed --sbert     # use MiniLM-L6 instead
    PYTHONPATH=src python -m utils.sensitivity.sa_embed --force     # re-encode existing
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from utils.metrics import create_embedding, generate_sbert_model


def find_runs(root: str) -> list[str]:
    """Return sorted list of posts.csv paths anywhere under ``root``.

    Catches both layouts:
      - axes:  <root>/<axis>/setting_*/rep_*/posts.csv
      - phq9:  <root>/phq9/<band>/posts.csv
    Recursive glob avoids missing future layout changes.
    """
    pattern = os.path.join(root, "**", "posts.csv")
    return sorted(glob.glob(pattern, recursive=True))


def encode_run(model, posts_csv: str) -> dict:
    """Encode one posts.csv → embedding array + aligned metadata."""
    df = pd.read_csv(posts_csv)
    texts = df["tweet"].fillna("").astype(str).tolist()
    embs = create_embedding(model, texts).cpu().numpy().astype(np.float32)
    return {
        "embeddings": embs,
        "agent_ids":  df["agent_id"].astype(np.int64).values,
        "rounds":     df["step"].astype(np.int64).values,
        "phq9":       df["phq9"].astype(np.int64).values,
        "texts":      np.array(texts, dtype=object),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/sensitivity",
                        help="Directory containing axis subdirs (default: data/sensitivity).")
    parser.add_argument("--sbert", action="store_true",
                        help="Use SBERT all-MiniLM-L6-v2 (384-dim) instead of MentalBERT (768-dim, default).")
    parser.add_argument("--force", action="store_true",
                        help="Re-encode runs whose embeddings.npz already exists.")
    args = parser.parse_args()

    mentalbert = not args.sbert
    encoder_name = "MentalBERT" if mentalbert else "SBERT-MiniLM-L6-v2"
    print(f"[embed] encoder = {encoder_name}")

    posts_csvs = find_runs(args.root)
    if not posts_csvs:
        raise SystemExit(f"No posts.csv found under {args.root}")
    print(f"[embed] found {len(posts_csvs)} runs")

    # Load model once.
    model = generate_sbert_model(mentalbert=mentalbert)

    for i, csv_path in enumerate(posts_csvs, 1):
        run_dir = os.path.dirname(csv_path)
        out_npz = os.path.join(run_dir, "embeddings.npz")
        meta_path = os.path.join(run_dir, "meta.json")
        if os.path.exists(out_npz) and not args.force:
            print(f"  ({i:2d}/{len(posts_csvs)}) [skip] {out_npz}")
            continue

        print(f"  ({i:2d}/{len(posts_csvs)}) {csv_path}")
        data = encode_run(model, csv_path)
        np.savez_compressed(out_npz, **data)

        meta = {
            "encoder":        encoder_name,
            "embedding_dim":  int(data["embeddings"].shape[1]),
            "n_posts":        int(data["embeddings"].shape[0]),
            "n_agents":       int(len(np.unique(data["agent_ids"]))),
            "n_rounds":       int(len(np.unique(data["rounds"]))),
            "source_csv":     os.path.relpath(csv_path),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        print(f"    → {out_npz}  ({meta['n_posts']} posts, dim {meta['embedding_dim']})")

    print(f"\n[done] encodings written under {args.root}/")
    print("       next: python -m utils.sensitivity.sa_analyze")


if __name__ == "__main__":
    main()
