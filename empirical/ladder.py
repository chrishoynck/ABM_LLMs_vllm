"""Demographically matched severity ladder: cosine between adjacent PHQ-9 bands on real users.

The synthetic ladder (`sa_analyze.phq9_adjacent_band_ladder`) holds the persona fixed
and moves only PHQ-9. Real users have one score each, so instead every user in band b
is paired with the closest user in band b+1: same gender, nearest age (ties broken at
random, matching with replacement). The cell is the mean cosine between the paired
users' tweet-block embeddings (mean of their tweet embeddings, as in
`plot_assessment_diagnostics.block_embeddings`), with a bootstrap 95% CI over pairs.
A random-partner baseline (any user in band b+1) shows what the matching controls for.
Run: sbatch empirical/run_empirical.job (step 2).
"""

import argparse
import os

import numpy as np
import pandas as pd

from utils.sensitivity.sa_analyze import BAND_LABELS, phq9_to_band

N_BOOT = 1000


def block_embeddings(npz_path: str) -> pd.DataFrame:
    """Unit-normalised mean tweet embedding per agent_id.

    Args:
        npz_path: per-tweet .npz written by empirical/embed.py.

    Returns:
        DataFrame indexed by agent_id, one embedding dimension per column.
    """
    d = np.load(npz_path, allow_pickle=True)
    blocks = pd.DataFrame(d["embeddings"]).groupby(d["agent_ids"]).mean()
    X = blocks.to_numpy()
    blocks.loc[:, :] = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return blocks


def matched_ladder(users: pd.DataFrame, blocks: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """One row per adjacent band step: matched cosine (+ CI), random-partner cosine, match quality.

    Args:
        users: indexed by agent_id with phq9, age (numeric) and gender columns.
        blocks: block embeddings indexed by agent_id (from `block_embeddings`).
        seed: seed for tie-breaking, the random baseline and the bootstrap.

    Returns:
        DataFrame with step, n_from, n_to, n_pairs, n_unmatched, cos, ci_lo, ci_hi,
        cos_random, mean_age_gap.
    """
    rng = np.random.default_rng(seed)
    users = users.assign(band=users["phq9"].map(phq9_to_band))
    rows = []
    for b0, b1 in zip(BAND_LABELS[:-1], BAND_LABELS[1:]):
        src, dst = users[users["band"] == b0], users[users["band"] == b1]
        cos, cos_rand, gaps, unmatched = [], [], [], 0
        for aid, u in src.iterrows():
            cand = dst[dst["gender"] == u["gender"]]
            if cand.empty:
                unmatched += 1
                continue
            gap = (cand["age"] - u["age"]).abs()
            j = rng.choice(gap.index[gap == gap.min()])
            cos.append(float(blocks.loc[aid] @ blocks.loc[j]))
            cos_rand.append(float(blocks.loc[aid] @ blocks.loc[rng.choice(dst.index)]))
            gaps.append(gap.min())
        cos = np.asarray(cos)
        boot = ([rng.choice(cos, cos.size).mean() for _ in range(N_BOOT)]
                if cos.size else [np.nan])
        rows.append({"step": f"{b0} -> {b1}", "n_from": len(src), "n_to": len(dst),
                     "n_pairs": int(cos.size), "n_unmatched": unmatched,
                     "cos": cos.mean() if cos.size else np.nan,
                     "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                     "cos_random": np.mean(cos_rand) if cos_rand else np.nan,
                     "mean_age_gap": np.mean(gaps) if gaps else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    """Parse args, build the matched ladder and write it as CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--users", required=True, help="CSV with agent_id, phq9, age, gender.")
    parser.add_argument("--emb", required=True, help="Per-tweet .npz from empirical/embed.py.")
    parser.add_argument("--out", required=True, help="Output CSV.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    users = pd.read_csv(args.users).set_index("agent_id")
    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    users["gender"] = users["gender"].astype(str)
    missing = users["age"].isna() | users["gender"].isin(["nan", "None", ""])
    print(f"[ladder] {len(users)} users, {int(missing.sum())} dropped for missing age/gender")
    users = users[~missing]

    ladder = matched_ladder(users, block_embeddings(args.emb), seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ladder.to_csv(args.out, index=False)
    print(ladder.to_string(index=False, float_format="%.3f"))
    print(f"[ladder] -> {args.out}")


if __name__ == "__main__":
    main()
