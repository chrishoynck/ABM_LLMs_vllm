"""Cognitive-distortion (CDS) n-grams by PHQ-9 on real users: the CDS check replicated.

Same detector and tables as `checks/check_cds_tracks_phq9.py` on synthetic posts
(share of posts with any CDS per PHQ-9 score, category x band), plus a per-user test:
each user's share of CDS tweets against their PHQ-9 (Spearman), since a user's tweets
are not independent. The empirical line is the mean over users of their % of CDS
tweets per score, so heavy tweeters do not dominate when users have different numbers
of tweets (with equal numbers it equals the % of posts). The figure is the synthetic
one with the users added: (a) the Qwen and Gemma lines plus the empirical line, (b) the
category x band heatmap for the users (% of tweets) with Qwen in brackets. A second
figure has the same comparison by severity band: mean % of CDS posts per user (per
10-post block for the synthetic corpora) in each band, +- SEM. Regex only, seconds on
any CPU. `figures.ipynb` section 6 calls `analyse`, `make_figure` and `make_band_figure`.
Run: PYTHONPATH=src:checks python empirical/cds.py --tweets <TAG>_tweets_phq.csv --out-dir <results>/cds
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from check_cds_tracks_phq9 import (ARMS, BAND_ORDER, FIG_BAND_LABELS, HEATMAP_CMAP,
                                   score_posts, summarize)
from utils.tools.cds import compile_category_patterns, load_ngrams_by_category

GENERATORS = ARMS["human"]                       # the paper's CDS figure: human-optimized Qwen and Gemma
EMP_COLOUR, EMP_MARKER = "#222222", "s"          # empirical = near-black squares, as in figures.ipynb
NGRAMS = "data/distorted_language_ngrams.tsv"


def per_user(df: pd.DataFrame) -> pd.DataFrame:
    """One row per user: PHQ-9, severity band, n tweets and % of tweets with any CDS."""
    users = df.groupby("agent_id").agg(phq9=("phq9", "first"), severity=("severity", "first"),
                                       n=("is_cds", "size"), pct_cds=("is_cds", "mean"))
    users["pct_cds"] *= 100
    return users


def band_means(units: pd.DataFrame) -> pd.DataFrame:
    """Mean, SEM and count of `pct_cds` over units (users or blocks) per severity band."""
    return units.groupby("severity")["pct_cds"].agg(["size", "mean", "sem"]).reindex(BAND_ORDER)


def analyse(tweets: str, ngrams: str = NGRAMS) -> dict:
    """CDS tables for the users and for the synthetic corpora.

    Args:
        tweets: CSV with agent_id, phq9, tweet.
        ngrams: CDS n-gram TSV.

    Returns:
        dict with per_score, cat_band (rows by overall prevalence), users, by_band,
        rho, p (per-user Spearman) and synth (generator tag -> (per_score, cat_band)).
    """
    patterns = compile_category_patterns(load_ngrams_by_category(ngrams))
    df, cat_cols = score_posts(tweets, patterns)
    per_score, cat_band, cat_overall = summarize(df, patterns, cat_cols)
    users = per_user(df)
    per_score = users.groupby("phq9").agg(n_users=("pct_cds", "size"), n_posts=("n", "sum"),
                                          pct_cds=("pct_cds", "mean")).reset_index()
    rho, p = spearmanr(users["phq9"], users["pct_cds"])
    synth, synth_band = {}, {}
    for tag, _, _, path in GENERATORS:
        sdf, scols = score_posts(path, patterns)
        synth[tag] = summarize(sdf, patterns, scols)[:2]
        synth_band[tag] = band_means(per_user(sdf))
    return {"n_tweets": len(df), "pct_all": 100 * df["is_cds"].mean(), "per_score": per_score,
            "cat_band": cat_band.loc[cat_overall.sort_values(ascending=False).index],
            "users": users, "rho": rho, "p": p, "synth": synth, "synth_band": synth_band,
            "by_band": band_means(users)}


def make_figure(res: dict) -> plt.Figure:
    """The synthetic CDS figure with the users added (per-user mean per score), from `analyse` output."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.4, 3.2), gridspec_kw={"width_ratios": [1, 1.6]})

    for tag, label, colour, _ in GENERATORS:
        ps = res["synth"][tag][0]
        ax0.plot(ps["phq9"], ps["pct_cds"], "o-", color=colour, lw=2, ms=4, label=label)
    emp = res["per_score"]
    ax0.plot(emp["phq9"], emp["pct_cds"], EMP_MARKER + "-", color=EMP_COLOUR, lw=1.2, ms=3.5,
             label="Empirical")
    ax0.set_xlabel("PHQ-9 sum-score")
    ax0.set_ylabel("% of posts containing CDS")
    ax0.grid(axis="y", linestyle=":", alpha=0.5)
    ax0.legend(loc="best", frameon=False, fontsize=8)
    ax0.text(0.5, -0.34, "(a) CDS vs PHQ-9", transform=ax0.transAxes, ha="center", va="top", fontsize=9.5)

    band = res["cat_band"]
    ref = res["synth"][GENERATORS[0][0]][1].reindex(index=band.index)[BAND_ORDER].values
    data = band[BAND_ORDER].values
    im = ax1.imshow(data, aspect="auto", cmap=HEATMAP_CMAP, vmin=0, vmax=np.nanmax(data))
    ax1.set_xticks(range(len(BAND_ORDER)))
    ax1.set_xticklabels(FIG_BAND_LABELS, rotation=30, ha="right", fontsize=9)
    ax1.set_yticks(range(len(band)))
    ax1.set_yticklabels(band.index, fontsize=9)
    thresh = np.nanmax(data) * 0.6
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax1.text(j, i, f"{data[i, j]:.1f} ({ref[i, j]:.1f})", ha="center", va="center",
                         fontsize=6.2, color="white" if data[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.04).set_label("% of posts")
    ax1.text(0.5, -0.34, f"(b) CDS category by PHQ-9 band, Empirical ({GENERATORS[0][1]})",
             transform=ax1.transAxes, ha="center", va="top", fontsize=9.5)
    fig.tight_layout()
    return fig


def make_band_figure(res: dict) -> plt.Figure:
    """% of CDS posts by PHQ-9 severity band: Qwen and Gemma (per block) and the users, mean +- SEM."""
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    x = np.arange(len(BAND_ORDER))
    for tag, label, colour, _ in GENERATORS:
        b = res["synth_band"][tag]
        ax.errorbar(x, b["mean"], yerr=b["sem"], fmt="o-", color=colour, lw=2, ms=4, capsize=2.5, label=label)
    b = res["by_band"]
    ax.errorbar(x, b["mean"], yerr=b["sem"], fmt=EMP_MARKER + "-", color=EMP_COLOUR, lw=1.2, ms=4,
                capsize=2.5, label="Empirical")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={int(n)})" for lab, n in zip(FIG_BAND_LABELS, b["size"].fillna(0))],
                       fontsize=8)
    ax.set_xlabel("PHQ-9 severity band (n users)")
    ax.set_ylabel("% of posts containing CDS")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.margins(x=0.08, y=0.12)
    fig.tight_layout()
    return fig


def report(res: dict) -> None:
    """Print the per-user test and the band / category tables."""
    print(f"[cds] {res['n_tweets']} tweets, {len(res['users'])} users "
          f"(median {res['users']['n'].median():.0f} tweets/user), {res['pct_all']:.1f}% CDS overall")
    print(f"[cds] per-user Spearman(PHQ-9, % CDS tweets) = {res['rho']:+.3f} "
          f"(p = {res['p']:.3g}, n = {len(res['users'])})")
    print("\n% CDS tweets per user, by band (mean, SEM):")
    print(res["by_band"].round(2).to_string())
    print("\n% of tweets per CDS category, by band:")
    print(res["cat_band"].round(1).to_string())


def main() -> None:
    """Score the tweets and the synthetic corpora, print the tables, write CSVs + figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tweets", required=True, help="CSV with agent_id, phq9, tweet.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ngrams", default=NGRAMS)
    args = parser.parse_args()

    res = analyse(args.tweets, args.ngrams)
    report(res)
    os.makedirs(args.out_dir, exist_ok=True)
    res["per_score"].to_csv(os.path.join(args.out_dir, "cds_by_phq9.csv"), index=False)
    res["users"].to_csv(os.path.join(args.out_dir, "cds_by_user.csv"))
    res["by_band"].to_csv(os.path.join(args.out_dir, "cds_by_band_users.csv"))
    res["cat_band"].to_csv(os.path.join(args.out_dir, "cds_by_category_band.csv"))
    fig_path = os.path.join(args.out_dir, "cds_vs_synthetic.png")
    make_figure(res).savefig(fig_path, dpi=300, bbox_inches="tight")
    band_path = os.path.join(args.out_dir, "cds_by_band.png")
    make_band_figure(res).savefig(band_path, dpi=300, bbox_inches="tight")
    print(f"[cds] -> {args.out_dir} (figures: {fig_path}, {band_path})")


if __name__ == "__main__":
    main()
