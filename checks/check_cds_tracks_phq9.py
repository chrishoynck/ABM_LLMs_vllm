"""Check that cognitive-distortion (CDS) n-grams get more common as PHQ-9 rises.

Loads the fine-tuning post CSVs, flags each post with the category-aware CDS
detector (`utils.tools.cds`, the same one `network_evolution` uses) and reports
the share of CDS posts per PHQ-9 score, overall and per category. If CDS are a
real depression signal that share should go up with PHQ-9. Writes a two-panel
figure (overall trend + category x severity-band heatmap) that the paper uses.
Run: see checks/README.md.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.tools.cds import compile_category_patterns, load_ngrams_by_category

DEFAULT_FILES = ("train_posts.csv", "test_posts.csv", "test_posts_extra.csv",
                 "calibration_posts.csv")

# Standard PHQ-9 severity bands (sum-score cut-offs).
SEVERITY_BANDS = [
    (0, 4, "minimal"),
    (5, 9, "mild"),
    (10, 14, "moderate"),
    (15, 19, "moderately\nsevere"),
    (20, 27, "severe"),
]
BAND_ORDER = [b[2] for b in SEVERITY_BANDS]


def load_posts(data_dir: str, files=DEFAULT_FILES) -> pd.DataFrame:
    """Concatenate the fine-tuning post CSVs that exist in `data_dir` into one frame."""
    frames = []
    for name in files:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(path)
        df["source"] = name
        frames.append(df)
        print(f"  [load] {name}: {len(df)} posts")
    if not frames:
        raise FileNotFoundError(f"No post CSVs found in {data_dir} (looked for {files})")
    return pd.concat(frames, ignore_index=True)


def _severity(phq9: int) -> str:
    """Map a PHQ-9 sum score to its severity band label."""
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= phq9 <= hi:
            return label
    return "unknown"


# Colours / labels matched to the SA cosine figure in sensitivity/sa_analyze.py
# (blue = within-setting bar, orange = cross-setting bar, Oranges heatmap).
COLOUR_BLUE = "#1f77b4"
COLOUR_ORANGE = "#ff7f0e"
HEATMAP_CMAP = "Oranges"
# Reference panel-(b) band labels (single line, Title case) so the x-axis
# matches "Minimal / Mild / Moderate / Mod. Severe / Severe".
FIG_BAND_LABELS = ["Minimal", "Mild", "Moderate", "Mod. Severe", "Severe"]


def make_figure(per_score: pd.DataFrame, cat_band: pd.DataFrame,
                r_agg: float, fig_path: str):
    """Write the two-panel figure: overall trend (left) + category heatmap (right).

    Colours and proportions match the SA cosine figure in `sensitivity/sa_analyze.py`.

    Args:
        per_score (pd.DataFrame): columns phq9, pct_cds (overall, any category).
        cat_band (pd.DataFrame): index=category, columns=severity bands, values=% CDS.
        r_agg (float): per-score Pearson r (% CDS vs PHQ-9), shown on the left panel.
        fig_path (str): output path (.png).
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.5, 3.5),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # ── Left: overall % CDS vs PHQ-9 score ──────────────────────────────────
    ax0.plot(per_score["phq9"], per_score["pct_cds"], "o-",
             color=COLOUR_BLUE, lw=2, ms=5, label="% CDS posts")
    # linear fit to make the trend explicit
    coef = np.polyfit(per_score["phq9"], per_score["pct_cds"], 1)
    xs = np.array([per_score["phq9"].min(), per_score["phq9"].max()])
    ax0.plot(xs, np.polyval(coef, xs), "--", color=COLOUR_ORANGE, lw=1.8,
             label=f"linear fit (r={r_agg:+.2f})")
    ax0.set_xlabel("PHQ-9 sum-score")
    ax0.set_ylabel("% of posts containing CDS")
    ax0.set_ylim(25, 90)
    ax0.grid(axis="y", linestyle=":", alpha=0.5)
    ax0.legend(loc="lower right", frameon=False)
    ax0.text(0.5, -0.26, "(a) CDS vs PHQ-9", transform=ax0.transAxes,
             ha="center", va="top", fontsize=11)

    # ── Right: category x severity-band heatmap (Oranges) ───────────────────
    data = cat_band.values
    im = ax1.imshow(data, aspect="auto", cmap=HEATMAP_CMAP, vmin=0,
                    vmax=np.nanmax(data))
    ax1.set_xticks(range(cat_band.shape[1]))
    ax1.set_xticklabels(FIG_BAND_LABELS, rotation=30, ha="right", fontsize=9)
    ax1.set_yticks(range(cat_band.shape[0]))
    ax1.set_yticklabels(cat_band.index, fontsize=9)
    # annotate cells (white on dark, black on light)
    thresh = np.nanmax(data) * 0.6
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                     color="white" if val > thresh else "black")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.04)
    cbar.set_label("% of posts")
    ax1.text(0.5, -0.26, "(b) CDS category by PHQ-9 band", transform=ax1.transAxes,
             ha="center", va="top", fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote figure to {fig_path}")


def main():
    """Score the posts, print the per-score and per-category tables, write the figure."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/finetune",
                        help="Directory holding the *_posts.csv files.")
    parser.add_argument("--ngrams", default="data/distorted_language_ngrams.tsv",
                        help="TSV of distorted-language n-grams (categories|markers|variants).")
    parser.add_argument("--fig", default="plots/cds_validation.png",
                        help="Output path for the figure.")
    parser.add_argument("--out", default=None,
                        help="Optional path to write the per-PHQ-9 CDS table as CSV "
                             "(a *_by_category.csv sibling is written alongside it).")
    args = parser.parse_args()

    print(f"Loading n-grams from {args.ngrams} ...")
    by_cat = load_ngrams_by_category(args.ngrams)
    patterns = compile_category_patterns(by_cat)
    total_ngrams = sum(len(v) for v in by_cat.values())
    print(f"Loaded {total_ngrams} n-grams across {len(patterns)} categories.\n")

    print(f"Loading posts from {args.data_dir} ...")
    df = load_posts(args.data_dir)
    df = df.dropna(subset=["tweet", "phq9"]).copy()
    df["phq9"] = df["phq9"].astype(int)
    df["tweet"] = df["tweet"].astype(str)
    df["severity"] = df["phq9"].apply(_severity)
    print(f"Total posts scored: {len(df)}\n")

    # ── Per-post detection: one boolean column per category + overall ───────
    cat_cols = []
    for cat, pat in patterns.items():
        col = f"cds::{cat}"
        df[col] = df["tweet"].str.contains(pat)
        cat_cols.append(col)
    df["is_cds"] = df[cat_cols].any(axis=1)

    # ── Average percentage CDS per PHQ-9 score (overall) ────────────────────
    per_score = (
        df.groupby("phq9")["is_cds"]
        .agg(n_posts="size", n_cds="sum")
        .reset_index()
    )
    per_score["pct_cds"] = 100.0 * per_score["n_cds"] / per_score["n_posts"]

    print("Average % CDS posts per PHQ-9 score (any category)")
    print("-" * 48)
    print(f"{'PHQ-9':>5} {'n_posts':>8} {'n_cds':>7} {'% CDS':>8}")
    for _, r in per_score.iterrows():
        print(f"{int(r.phq9):>5} {int(r.n_posts):>8} {int(r.n_cds):>7} {r.pct_cds:>7.2f}%")
    print(f"{'all':>5} {len(df):>8} {int(df.is_cds.sum()):>7} "
          f"{100.0 * df.is_cds.mean():>7.2f}%\n")

    # ── Category x severity-band table (% posts, used by the heatmap) ───────
    rows = {}
    for cat, col in zip(patterns.keys(), cat_cols):
        rows[cat] = df.groupby("severity")[col].mean().mul(100.0)
    cat_band = pd.DataFrame(rows).T.reindex(columns=BAND_ORDER)
    # overall category prevalence, used to sort rows (most common at top)
    cat_overall = (df[cat_cols].mean() * 100.0)
    cat_overall.index = list(patterns.keys())
    cat_band = cat_band.loc[cat_overall.sort_values(ascending=False).index]
    # per-category Pearson r of % CDS vs PHQ-9 severity midpoint, for the trend
    band_mid = {b[2]: (b[0] + b[1]) / 2 for b in SEVERITY_BANDS}

    # short, single-line band labels so the stdout columns line up (the figure
    # keeps the full two-line labels)
    console_band = {"minimal": "minimal", "mild": "mild", "moderate": "moderate",
                    "moderately\nsevere": "mod.sev", "severe": "severe"}
    print("Average % CDS posts per category, by PHQ-9 severity band")
    print("-" * 78)
    hdr = "".join(f"{console_band[b]:>12}" for b in BAND_ORDER)
    print(f"{'category':<28}{hdr}{'overall':>10}{'trend r':>9}")
    for cat in cat_band.index:
        vals = "".join(f"{cat_band.loc[cat, b]:>11.1f}%" for b in BAND_ORDER)
        mids = np.array([band_mid[b] for b in BAND_ORDER])
        r_cat = np.corrcoef(mids, cat_band.loc[cat, BAND_ORDER].values)[0, 1]
        print(f"{cat:<28}{vals}{cat_overall[cat]:>9.1f}%{r_cat:>+9.2f}")
    print()

    # ── Does CDS get more probable with higher PHQ-9? ───────────────────────
    cds = df["is_cds"].astype(float).values
    phq = df["phq9"].astype(float).values
    r_post = np.corrcoef(phq, cds)[0, 1]
    rank_r = pd.Series(phq).corr(pd.Series(cds), method="spearman")
    r_agg = per_score["phq9"].corr(per_score["pct_cds"])

    print("Correlation between PHQ-9 and CDS probability (any category)")
    print("-" * 60)
    print(f"  per-post Pearson (point-biserial) : {r_post:+.3f}")
    print(f"  per-post Spearman                 : {rank_r:+.3f}")
    print(f"  per-score Pearson (% vs PHQ-9)    : {r_agg:+.3f}")
    verdict = "YES" if r_post > 0 else "NO"
    print(f"\n=> CDS more probable for higher PHQ-9? {verdict} "
          f"(positive correlation means yes)")

    # ── Figure ──────────────────────────────────────────────────────────────
    make_figure(per_score, cat_band, r_agg, args.fig)

    # ── Optional CSVs ─────────────────────────────────────────────────────────
    if args.out:
        per_score.to_csv(args.out, index=False)
        cat_out = args.out.replace(".csv", "_by_category.csv")
        if cat_out == args.out:
            cat_out = args.out + ".by_category.csv"
        cat_band.to_csv(cat_out)
        print(f"\nWrote per-PHQ-9 table to {args.out}")
        print(f"Wrote per-category table to {cat_out}")


if __name__ == "__main__":
    main()
