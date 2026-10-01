"""Demographically matched severity ladder on real users: adjacent-band and within-band cosine.

The synthetic ladder (`sa_analyze.phq9_adjacent_band_ladder`) holds the persona fixed
and moves only PHQ-9. Real users have one score each, so instead every user is paired
with the closest other user (same gender, nearest age, ties broken at random, matching
with replacement), either in the next band up (the ladder) or in their own band (the
within-band reference). A random partner from the same target band is the baseline
for what the matching controls for.

The synthetic cosines are between single posts, so the main empirical cell is too: the
mean cosine over all tweet pairs of the two users. For unit-length tweet embeddings
that is the dot product of the users' unnormalised mean embeddings, so it costs no more
than a block comparison. `*_profile` columns keep the cosine between the users'
normalised mean embeddings (as in `plot_assessment_diagnostics.block_embeddings`),
which averages away post-level variation and so sits much higher. Cells carry a
bootstrap 95% CI over pairs.

As in `checks/phq9_band_significance.py` for the synthetic runs, every adjacent step
also tests "starting from this band, is the next band up further away than this band
itself?": each user in band b has an adjacent cosine (matched partner in b+1) and a
within cosine (matched partner in b), and delta = adjacent - within gets a paired
t-test over those users (the user is the unit, as the persona is there).
Run: empirical/run_empirical.job (step 2).
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from utils.sensitivity.sa_analyze import BAND_LABELS, phq9_to_band

N_BOOT = 1000
METRICS = {"": "post", "_profile": "profile"}   # column suffix -> embedding frame


def user_embeddings(npz_path: str) -> dict:
    """Per-user mean tweet embeddings in the two forms the ladder compares.

    Args:
        npz_path: per-tweet .npz written by empirical/embed.py.

    Returns:
        dict with "post" (mean of unit tweet vectors, unnormalised: the dot product of
        two users is their mean pairwise tweet cosine) and "profile" (the same mean,
        normalised), both DataFrames indexed by agent_id.
    """
    d = np.load(npz_path, allow_pickle=True)
    E = d["embeddings"].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    post = pd.DataFrame(E).groupby(d["agent_ids"]).mean()
    X = post.to_numpy()
    profile = pd.DataFrame(X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12), index=post.index)
    return {"post": post, "profile": profile}


def match_cell(src: pd.DataFrame, dst: pd.DataFrame, emb: dict,
               rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    """Pair every user in `src` with their nearest same-gender, closest-age other user in `dst`.

    Args:
        src: users to match (indexed by agent_id, with age and gender).
        dst: candidate partners; a user is never paired with themselves.
        emb: output of `user_embeddings`.
        rng: generator for tie-breaking, the random partner and the bootstrap.

    Returns:
        (summary, per_user): summary has n_pairs, n_unmatched, mean_age_gap and, per
        metric suffix, cos, ci_lo, ci_hi, cos_random; per_user is indexed by the
        matched src agent_ids with one cosine column per metric suffix.
    """
    pairs, gaps, unmatched = [], [], 0
    for aid, u in src.iterrows():
        others = dst.drop(index=aid, errors="ignore")
        cand = others[others["gender"] == u["gender"]]
        if cand.empty:
            unmatched += 1
            continue
        gap = (cand["age"] - u["age"]).abs()
        pairs.append((aid, rng.choice(gap.index[gap == gap.min()]), rng.choice(others.index)))
        gaps.append(gap.min())
    summary = {"n_pairs": len(pairs), "n_unmatched": unmatched,
               "mean_age_gap": np.mean(gaps) if gaps else np.nan}
    per_user = pd.DataFrame(index=pd.Index([a for a, _, _ in pairs], name="agent_id"))
    for suffix, name in METRICS.items():
        E = emb[name]
        cos = np.array([E.loc[a] @ E.loc[j] for a, j, _ in pairs])
        rand = np.array([E.loc[a] @ E.loc[r] for a, _, r in pairs])
        boot = [rng.choice(cos, cos.size).mean() for _ in range(N_BOOT)] if cos.size else [np.nan]
        summary.update({f"cos{suffix}": cos.mean() if cos.size else np.nan,
                        f"ci_lo{suffix}": np.percentile(boot, 2.5), f"ci_hi{suffix}": np.percentile(boot, 97.5),
                        f"cos_random{suffix}": rand.mean() if rand.size else np.nan})
        per_user[f"cos{suffix}"] = cos
    return summary, per_user


def paired_test(adj: pd.Series, within: pd.Series) -> dict:
    """Within (this band) vs adjacent (next band) cosine over the users that have both: paired t-test."""
    both = pd.concat({"adj": adj, "within": within}, axis=1).dropna()
    if len(both) < 3:
        return {"n_paired": len(both), "within": np.nan, "adjacent": np.nan, "delta": np.nan,
                "t": np.nan, "p": np.nan, "dz": np.nan}
    d = both["adj"] - both["within"]
    t, p = stats.ttest_rel(both["adj"], both["within"])
    return {"n_paired": len(both), "within": both["within"].mean(), "adjacent": both["adj"].mean(),
            "delta": d.mean(), "t": t, "p": p, "dz": d.mean() / d.std(ddof=1)}


def ladders(users: pd.DataFrame, emb: dict, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjacent-band ladder (one row per step, with the paired test) and within-band reference.

    Args:
        users: indexed by agent_id with phq9, age (numeric) and gender columns.
        emb: output of `user_embeddings`.
        seed: seed for tie-breaking, the random baseline and the bootstrap.

    Returns:
        (adjacent, within) DataFrames.
    """
    rng = np.random.default_rng(seed)
    users = users.assign(band=users["phq9"].map(phq9_to_band))
    by_band = {b: users[users["band"] == b] for b in BAND_LABELS}
    within_rows, within_users = [], {}
    for b in BAND_LABELS:
        summary, within_users[b] = match_cell(by_band[b], by_band[b], emb, rng)
        within_rows.append({"band": b, "n_users": len(by_band[b]), **summary})
    adjacent_rows = []
    for b0, b1 in zip(BAND_LABELS[:-1], BAND_LABELS[1:]):
        summary, adj_users = match_cell(by_band[b0], by_band[b1], emb, rng)
        test = paired_test(adj_users["cos"], within_users[b0]["cos"])
        adjacent_rows.append({"step": f"{b0} -> {b1}", "n_from": len(by_band[b0]), "n_to": len(by_band[b1]),
                              **summary, **{f"test_{k}": v for k, v in test.items()}})
    return pd.DataFrame(adjacent_rows), pd.DataFrame(within_rows)


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

    adjacent, within = ladders(users, user_embeddings(args.emb), seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    within_out = os.path.splitext(args.out)[0] + "_within.csv"
    adjacent.to_csv(args.out, index=False)
    within.to_csv(within_out, index=False)
    print(adjacent.to_string(index=False, float_format="%.3f"))
    print(within.to_string(index=False, float_format="%.3f"))
    print(f"[ladder] -> {args.out}, {within_out}")


if __name__ == "__main__":
    main()
