"""Embed every tweet of the empirical Twitter + PHQ-9 set once, with both encoders.

The only step that runs a transformer, so everything downstream (ladder, scoring,
bias) reads the saved vectors instead of re-encoding. Same encoder setup as the
synthetic pipeline (`utils.metrics.generate_sbert_model`, as called by `sa_embed`
and `eval_bert_on_csv`): S-BERT all-MiniLM-L6-v2 for the severity ladder, MentalBERT
for the PHQ-9 regressors. Writes `sbert_tweets.npz` and `mentalbert_tweets.npz`
(embeddings, agent_ids, phq9, texts) and skips an encoder whose file exists.
Run: empirical/run_empirical.job (step 1).
"""

import argparse
import os

import numpy as np
import pandas as pd

from utils.metrics import create_embedding, generate_sbert_model

ENCODERS = {"sbert": False, "mentalbert": True}   # file prefix -> generate_sbert_model(mentalbert=)


def main() -> None:
    """Parse args and write one .npz per encoder that is not already on disk."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tweets", required=True, help="CSV with agent_id, phq9, tweet.")
    parser.add_argument("--out-dir", required=True, help="Directory for the .npz files.")
    parser.add_argument("--force", action="store_true", help="Re-encode even if the .npz exists.")
    parser.add_argument("--device", default=None, help="torch device (default: auto).")
    args = parser.parse_args()

    tw = pd.read_csv(args.tweets)
    texts = tw["tweet"].fillna("").astype(str).tolist()
    os.makedirs(args.out_dir, exist_ok=True)
    for name, mentalbert in ENCODERS.items():
        out = os.path.join(args.out_dir, f"{name}_tweets.npz")
        if os.path.exists(out) and not args.force:
            print(f"[embed] {name}: {out} exists, skipping")
            continue
        model = generate_sbert_model(mentalbert=mentalbert, device=args.device)
        print(f"[embed] {name}: {len(texts)} tweets, {tw['agent_id'].nunique()} users on {model.device}")
        emb = create_embedding(model, texts).cpu().numpy().astype(np.float32)
        np.savez_compressed(out, embeddings=emb,
                            agent_ids=tw["agent_id"].to_numpy(np.int64),
                            phq9=tw["phq9"].to_numpy(np.int64),
                            texts=np.array(texts, dtype=object))
        print(f"[embed] {name}: {emb.shape} -> {out}")


if __name__ == "__main__":
    main()
