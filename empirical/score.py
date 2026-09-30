"""Score every MentalBERT+MLP assessor (arm x seed) on the empirical users from saved embeddings.

Same scoring as `eval_bert_on_csv.evaluate` (the `--mode bert-eval` path): a block of
posts becomes the (mean | max | std) centroid the regressor was trained on. The
synthetic blocks are 10 posts, so each user's tweets are cut into consecutive blocks
of 10 in the order they were exported (chronological), a remainder under 10 is dropped
and a user with fewer than 10 tweets is one block of all of them. Every block is
scored and a user's prediction is the mean over their blocks, rounded and clipped to
0-27. The input is the per-tweet MentalBERT vectors from `empirical/embed.py`, so the
25 regressors run in seconds on CPU instead of re-encoding the tweets 25 times. Writes
`<out-root>/<arm>/seed<NN>.csv` with the same columns as `bert-eval` plus `n_blocks`,
which `bias.py` and the notebook read.
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


BLOCK = 10                             # posts per block, as in the synthetic training data


def block_centroids(npz_path: str, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """(mean | max | std) centroid per block of `block` consecutive tweets, as `eval_bert_on_csv` builds it.

    Args:
        npz_path: mentalbert_tweets.npz from empirical/embed.py.
        block: tweets per block; a remainder shorter than this is dropped, and a user
            with fewer tweets in total is one block of all of them.

    Returns:
        (agent_ids, true_phq9, block_agent_ids, centroids): one entry per user in
        first-seen order for the first two, one row per block for the last two.
    """
    d = np.load(npz_path, allow_pickle=True)
    emb, aids, phq = torch.from_numpy(d["embeddings"]), d["agent_ids"], d["phq9"]
    first = np.flatnonzero(np.r_[True, aids[1:] != aids[:-1]])     # tweets are stored grouped by user
    order, counts = aids[first], np.diff(np.r_[first, len(aids)])
    assert len(np.unique(order)) == len(order), "tweets must be grouped by agent_id"
    rows, block_aids = [], []
    for a, start, n in zip(order, first, counts):
        e = emb[start:start + n]
        n_blocks = max(1, n // block)
        e = e[:n_blocks * block].reshape(n_blocks, -1, e.shape[1]) if n >= block else e[None]
        var = e.var(dim=1) if e.shape[1] > 1 else torch.zeros(e.shape[0], e.shape[2])
        rows.append(torch.cat([e.mean(dim=1), e.max(dim=1)[0], torch.sqrt(var + 1e-8)], dim=1))
        block_aids.append(np.full(len(e), a))
    return order, phq[first], np.concatenate(block_aids), torch.cat(rows)


def user_means(block_aids: np.ndarray, raw: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean block prediction and number of blocks per user, in the order of `order`."""
    g = pd.Series(raw).groupby(block_aids)
    return g.mean().reindex(order).to_numpy(), g.size().reindex(order).to_numpy()


def main() -> None:
    """Parse args, run every regressor found under data/assessors/bert on the centroids."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emb", required=True, help="mentalbert_tweets.npz from empirical/embed.py.")
    parser.add_argument("--out-root", required=True, help="Writes <out-root>/<arm>/seed<NN>.csv.")
    parser.add_argument("--arms", nargs="*", default=None, help="Assessor arms (default: all with models/).")
    args = parser.parse_args()

    aids, true, baids, X = block_centroids(args.emb)
    print(f"[score] {len(aids)} users, {len(X)} blocks of up to {BLOCK} tweets, centroid dim {X.shape[1]}")
    paths = sorted(glob.glob(os.path.join(ASSESSORS, "*", "models", "*_seed*", "regressor.pt")))
    for path in paths:
        arm = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        if args.arms and arm not in args.arms:
            continue
        seed = re.search(r"_seed(\d+)", path).group(1)
        reg = torch.load(path, map_location="cpu", weights_only=False).eval()
        with torch.no_grad():
            raw, n_blocks = user_means(baids, reg(X).squeeze(-1).numpy(), aids)
        pred = np.clip(np.round(raw).astype(int), 0, 27)
        err = pred - true
        out_dir = os.path.join(args.out_root, arm)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({"agent_id": aids, "true_phq9": true, "pred_phq9": pred, "raw_pred": raw,
                      "abs_error": np.abs(err), "signed_bias": err, "n_blocks": n_blocks}
                     ).to_csv(os.path.join(out_dir, f"seed{seed}.csv"), index=False)
        print(f"[score] {arm} seed {seed}: MAE={np.abs(err).mean():.2f}  bias={err.mean():+.2f}")


if __name__ == "__main__":
    main()
