"""Score every MentalBERT+MLP assessor (arm x seed) on the empirical users from saved embeddings.

Same scoring as `eval_bert_on_csv.evaluate` (the `--mode bert-eval` path): each user's
tweets become the (mean | max | std) centroid the regressor was trained on, and the
prediction is rounded and clipped to 0-27. The difference is the input: the per-tweet
MentalBERT vectors from `empirical/embed.py`, so the 25 regressors run in seconds on
CPU instead of re-encoding the tweets 25 times. Writes `<out-root>/<arm>/seed<NN>.csv`
with the same columns as `bert-eval`, which `bias.py` and the notebook read.
Run: empirical/run_empirical.job (step 3).
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ASSESSORS = "data/assessors/bert"


class neural_net_BERT(nn.Module):
    """Unpickling stand-in for `prompt_optimizer.neural_net_BERT`.

    The regressors were pickled whole from `python -m utils.prompt_optimizer`, so they
    reference `__main__.neural_net_BERT`. Unpickling restores the layers without
    calling __init__, so this only needs the name and the forward pass, and spares
    importing prompt_optimizer (vLLM, TextGrad) just to load a 3-layer MLP.
    """

    def forward(self, x):
        """Predict one PHQ-9 score per row of x."""
        return self.model(x)


def user_centroids(npz_path: str) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """(mean | max | std) centroid per user, exactly as `eval_bert_on_csv` builds it.

    Args:
        npz_path: mentalbert_tweets.npz from empirical/embed.py.

    Returns:
        (agent_ids, true_phq9, centroids) with one row per user, in first-seen order.
    """
    d = np.load(npz_path, allow_pickle=True)
    emb, aids, phq = torch.from_numpy(d["embeddings"]), d["agent_ids"], d["phq9"]
    order = pd.unique(aids)
    rows = []
    for a in order:
        e = emb[aids == a]
        var = e.var(dim=0)
        if torch.isnan(var).any():          # single tweet: no spread
            var = torch.zeros_like(var)
        rows.append(torch.cat([e.mean(dim=0), e.max(dim=0)[0], torch.sqrt(var + 1e-8)]))
    true = np.array([phq[aids == a][0] for a in order])
    return order, true, torch.stack(rows)


def main() -> None:
    """Parse args, run every regressor found under data/assessors/bert on the centroids."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emb", required=True, help="mentalbert_tweets.npz from empirical/embed.py.")
    parser.add_argument("--out-root", required=True, help="Writes <out-root>/<arm>/seed<NN>.csv.")
    parser.add_argument("--arms", nargs="*", default=None, help="Assessor arms (default: all with models/).")
    args = parser.parse_args()

    aids, true, X = user_centroids(args.emb)
    print(f"[score] {len(aids)} users, centroid dim {X.shape[1]}")
    paths = sorted(glob.glob(os.path.join(ASSESSORS, "*", "models", "*_seed*", "regressor.pt")))
    for path in paths:
        arm = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        if args.arms and arm not in args.arms:
            continue
        seed = re.search(r"_seed(\d+)", path).group(1)
        reg = torch.load(path, map_location="cpu", weights_only=False).eval()
        with torch.no_grad():
            raw = reg(X).squeeze(-1).numpy()
        pred = np.clip(np.round(raw).astype(int), 0, 27)
        err = pred - true
        out_dir = os.path.join(args.out_root, arm)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({"agent_id": aids, "true_phq9": true, "pred_phq9": pred, "raw_pred": raw,
                      "abs_error": np.abs(err), "signed_bias": err}
                     ).to_csv(os.path.join(out_dir, f"seed{seed}.csv"), index=False)
        print(f"[score] {arm} seed {seed}: MAE={np.abs(err).mean():.2f}  bias={err.mean():+.2f}")


if __name__ == "__main__":
    main()
