"""Embed every tweet of the empirical Twitter + PHQ-9 set, one vector per tweet.

Same encoder setup as the synthetic pipeline (`utils.metrics.generate_sbert_model`,
as called by `sa_embed`), so empirical and generated embeddings are comparable:
S-BERT all-MiniLM-L6-v2 by default (the severity ladder), MentalBERT with
--mentalbert. Writes an .npz with embeddings, agent_ids, phq9 and texts.
Run: sbatch empirical/run_empirical.job (step 1).
"""

import argparse
import os

import numpy as np
import pandas as pd

from utils.metrics import create_embedding, generate_sbert_model


def main() -> None:
    """Parse args, embed the tweets CSV and save the .npz."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tweets", required=True, help="CSV with agent_id, phq9, tweet.")
    parser.add_argument("--out", required=True, help="Output .npz.")
    parser.add_argument("--mentalbert", action="store_true",
                        help="Encode with MentalBERT (768-d) instead of S-BERT MiniLM (384-d).")
    parser.add_argument("--device", default=None, help="torch device (default: auto).")
    args = parser.parse_args()

    tw = pd.read_csv(args.tweets)
    texts = tw["tweet"].fillna("").astype(str).tolist()
    model = generate_sbert_model(mentalbert=args.mentalbert, device=args.device)
    print(f"[embed] {len(texts)} tweets, {tw['agent_id'].nunique()} users, "
          f"encoder={'MentalBERT' if args.mentalbert else 'SBERT-MiniLM'} on {model.device}")
    emb = create_embedding(model, texts).cpu().numpy().astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, embeddings=emb,
                        agent_ids=tw["agent_id"].to_numpy(np.int64),
                        phq9=tw["phq9"].to_numpy(np.int64),
                        texts=np.array(texts, dtype=object))
    print(f"[embed] {emb.shape} -> {args.out}")


if __name__ == "__main__":
    main()
