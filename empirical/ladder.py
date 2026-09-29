"""Demographically matched severity ladder on real users: adjacent-band and within-band cosine.

The synthetic ladder (`sa_analyze.phq9_adjacent_band_ladder`) holds the persona fixed
and moves only PHQ-9. Real users have one score each, so instead every user is paired
with the closest other user (same gender, nearest age, ties broken at random, matching
with replacement), either in the next band up (the ladder) or in their own band (the
within-band reference). A cell is the mean cosine between the paired users' tweet-block
embeddings (mean of their tweet embeddings, as in
`plot_assessment_diagnostics.block_embeddings`), with a bootstrap 95% CI over pairs. A
random partner from the same target band is the baseline for what the matching controls
for. If adjacent-band pairs are as similar as within-band pairs, the bands are not
separable in the language.
Run: empirical/run_empirical.job (step 2).
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


def match_cell(src: pd.DataFrame, dst: pd.DataFrame, blocks: pd.DataFrame,
               rng: np.random.Generator) -> dict:
    """Pair every user in `src` with their nearest same-gender, closest-age other user in `dst`.

    Args:
        src: users to match (indexed by agent_id, with age and gender).
        dst: candidate partners; a user is never paired with themselves.
        blocks: block embeddings indexed by agent_id.
        rng: generator for tie-breaking, the random partner and the bootstrap.

    Returns:
        dict with n_pairs, n_unmatched, cos, ci_lo, ci_hi, cos_random, mean_age_gap.
    """
    cos, cos_rand, gaps, unmatched = [], [], [], 0
    for aid, u in src.iterrows():
        others = dst.drop(index=aid, errors="ignore")
        cand = others[others["gender"] == u["gender"]]
        if cand.empty:
            unmatched += 1
            continue
        gap = (cand["age"] - u["age"]).abs()
        j = rng.choice(gap.index[gap == gap.min()])
        cos.append(float(blocks.loc[aid] @ blocks.loc[j]))
        cos_rand.append(float(blocks.loc[aid] @ blocks.loc[rng.choice(others.index)]))
        gaps.append(gap.min())
    cos = np.asarray(cos)
    boot = [rng.choice(cos, cos.size).mean() for _ in range(N_BOOT)] if cos.size else [np.nan]
    return {"n_pairs": int(cos.size), "n_unmatched": unmatched,
            "cos": cos.mean() if cos.size else np.nan,
            "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
            "cos_random": np.mean(cos_rand) if cos_rand else np.nan,
            "mean_age_gap": np.mean(gaps) if gaps else np.nan}


def ladders(users: pd.DataFrame, blocks: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjacent-band ladder (one row per step) and within-band reference (one row per band).

    Args:
        users: indexed by agent_id with phq9, age (numeric) and gender columns.
        blocks: block embeddings indexed by agent_id (from `block_embeddings`).
        seed: seed for tie-breaking, the random baseline and the bootstrap.

    Returns:
        (adjacent, within) DataFrames.
    """
    rng = np.random.default_rng(seed)
    users = users.assign(band=users["phq9"].map(phq9_to_band))
    by_band = {b: users[users["band"] == b] for b in BAND_LABELS}
    adjacent = pd.DataFrame([{"step": f"{b0} -> {b1}", "n_from": len(by_band[b0]), "n_to": len(by_band[b1]),
                              **match_cell(by_band[b0], by_band[b1], blocks, rng)}
                             for b0, b1 in zip(BAND_LABELS[:-1], BAND_LABELS[1:])])
    within = pd.DataFrame([{"band": b, "n_users": len(by_band[b]),
                            **match_cell(by_band[b], by_band[b], blocks, rng)} for b in BAND_LABELS])
    return adjacent, within


def main() -> None:
    """Parse args, build both ladders and write them as CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--users", required=True, help="CSV with agent_id, phq9, age, gender.")
    parser.add_argument("--emb", required=True, help="Per-tweet .npz from empirical/embed.py.")
    parser.add_argument("--out", required=True, help="Adjacent-band CSV; the within-band one goes "
                                                     "next to it as <stem>_within.csv.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    users = pd.read_csv(args.users).set_index("agent_id")
    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    users["gender"] = users["gender"].astype(str)
    missing = users["age"].isna() | users["gender"].isin(["nan", "None", ""])
    print(f"[ladder] {len(users)} users, {int(missing.sum())} dropped for missing age/gender")
    users = users[~missing]

    adjacent, within = ladders(users, block_embeddings(args.emb), seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    within_out = os.path.splitext(args.out)[0] + "_within.csv"
    adjacent.to_csv(args.out, index=False)
    within.to_csv(within_out, index=False)
    print(adjacent.to_string(index=False, float_format="%.3f"))
    print(within.to_string(index=False, float_format="%.3f"))
    print(f"[ladder] -> {args.out}, {within_out}")


if __name__ == "__main__":
    main()
