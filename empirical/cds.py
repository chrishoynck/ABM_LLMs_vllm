"""Cognitive-distortion (CDS) n-grams by PHQ-9 on real users: the CDS check replicated.

Same detector and tables as `checks/check_cds_tracks_phq9.py` on synthetic posts
(share of posts with any CDS per PHQ-9 score, category x band), plus a per-user test:
each user's share of CDS tweets against their PHQ-9 (Spearman), since a user's tweets
are not independent. Regex only, seconds on any CPU.
Run: PYTHONPATH=src:checks python empirical/cds.py --tweets <TAG>_tweets_phq.csv --out-dir <results>/cds
"""

import argparse
import os

import pandas as pd
from scipy.stats import spearmanr

from check_cds_tracks_phq9 import BAND_ORDER, score_posts, summarize
from utils.tools.cds import compile_category_patterns, load_ngrams_by_category


def per_user(df: pd.DataFrame) -> pd.DataFrame:
    """One row per user: PHQ-9, severity band, n tweets and % of tweets with any CDS."""
    users = df.groupby("agent_id").agg(phq9=("phq9", "first"), severity=("severity", "first"),
                                       n=("is_cds", "size"), pct_cds=("is_cds", "mean"))
    users["pct_cds"] *= 100
    return users


def main() -> None:
    """Score the tweets, print the per-user test and tables, write them as CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tweets", required=True, help="CSV with agent_id, phq9, tweet.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ngrams", default="data/distorted_language_ngrams.tsv")
    args = parser.parse_args()

    patterns = compile_category_patterns(load_ngrams_by_category(args.ngrams))
    df, cat_cols = score_posts(args.tweets, patterns)
    per_score, cat_band, cat_overall = summarize(df, patterns, cat_cols)
    cat_band = cat_band.loc[cat_overall.sort_values(ascending=False).index]
    users = per_user(df)
    rho, p = spearmanr(users["phq9"], users["pct_cds"])
    by_band = users.groupby("severity")["pct_cds"].agg(["size", "mean", "sem"]).reindex(BAND_ORDER)

    print(f"[cds] {len(df)} tweets, {len(users)} users, {100 * df['is_cds'].mean():.1f}% CDS overall")
    print(f"[cds] per-user Spearman(PHQ-9, % CDS tweets) = {rho:+.3f} (p = {p:.3g}, n = {len(users)})")
    print("\n% CDS tweets per user, by band (mean, SEM):")
    print(by_band.round(2).to_string())
    print("\n% of tweets per CDS category, by band:")
    print(cat_band.round(1).to_string())

    os.makedirs(args.out_dir, exist_ok=True)
    per_score.to_csv(os.path.join(args.out_dir, "cds_by_phq9.csv"), index=False)
    users.to_csv(os.path.join(args.out_dir, "cds_by_user.csv"))
    by_band.to_csv(os.path.join(args.out_dir, "cds_by_band_users.csv"))
    cat_band.to_csv(os.path.join(args.out_dir, "cds_by_category_band.csv"))
    print(f"[cds] -> {args.out_dir}")


if __name__ == "__main__":
    main()
