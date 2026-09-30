"""Check that cognitive-distortion (CDS) n-grams get more common as PHQ-9 rises.

Loads the human-optimized-prompt training corpus of each generator (Qwen and
Gemma: same personas, scores and neighbour draws, so the two are paired), flags
each post with the category-aware CDS detector (`utils.tools.cds`, the same one
`network_evolution` uses) and reports the share of CDS posts per PHQ-9 score,
overall and per category. If CDS are a real depression signal that share should
go up with PHQ-9. Writes a two-panel figure (overall trend per generator +
category x severity-band heatmap, Qwen value with Gemma in brackets) that the
paper uses. `--arm minimal` runs the same check on the minimal-prompt (iter_0)
corpora of the same personas. Run: see checks/README.md.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.tools.cds import compile_category_patterns, load_ngrams_by_category

# (tag, label, colour, posts file) per prompt arm. "human" is the iter_10 prompt,
# "minimal" the iter_0 prompt, both over the same 3,000 (persona, PHQ-9) blocks
# and neighbour draws, so all four corpora are paired; colours follow the
# multi-model figures in `utils.visualization.MULTIMODEL_GENERATORS`. The first
# entry is the primary one (heatmap colour, row order); the second goes in
# brackets.
ARMS = {
    "human": [
        ("qwen", "Qwen", "#eb6834", "data/finetune/qwen/train_posts_qwen.csv"),
        ("gemma4", "Gemma", "#2a78d6", "data/finetune/gemma4/train_posts_gemma4.csv"),
    ],
    "minimal": [
        ("qwen_minimal", "Qwen", "#eb6834",
         "data/finetune/qwen_minimal/train_posts_qwen_minimal.csv"),
        ("gemma4_minimal", "Gemma", "#2a78d6",
         "data/finetune/gemma4_minimal/train_posts_gemma4_minimal.csv"),
    ],
}
ARM_LABELS = {"human": "Human-optimized", "minimal": "Minimal"}

# Standard PHQ-9 severity bands (sum-score cut-offs).
SEVERITY_BANDS = [
    (0, 4, "minimal"),
    (5, 9, "mild"),
    (10, 14, "moderate"),
    (15, 19, "moderately\nsevere"),
    (20, 27, "severe"),
]
BAND_ORDER = [b[2] for b in SEVERITY_BANDS]


def _severity(phq9: int) -> str:
    """Map a PHQ-9 sum score to its severity band label."""
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= phq9 <= hi:
            return label
    return "unknown"


def score_posts(path: str, patterns: dict) -> tuple[pd.DataFrame, list[str]]:
    """Load one posts CSV and flag every post per CDS category.

    Args:
        path (str): posts CSV with `tweet` and `phq9` columns.
        patterns (dict): category -> compiled regex (`compile_category_patterns`).

    Returns:
        tuple: (frame with `severity`, one `cds::<cat>` bool column per category
        and `is_cds`, list of those category column names).
    """
    df = pd.read_csv(path).dropna(subset=["tweet", "phq9"]).copy()
    df["phq9"] = df["phq9"].astype(int)
    df["tweet"] = df["tweet"].astype(str)
    df["severity"] = df["phq9"].apply(_severity)
    cat_cols = []
    for cat, pat in patterns.items():
        col = f"cds::{cat}"
        df[col] = df["tweet"].str.contains(pat)
        cat_cols.append(col)
    df["is_cds"] = df[cat_cols].any(axis=1)
    return df, cat_cols


def summarize(df: pd.DataFrame, patterns: dict, cat_cols: list[str]):
    """Per-score CDS share, category x band table and overall category prevalence."""
    per_score = (
        df.groupby("phq9")["is_cds"]
        .agg(n_posts="size", n_cds="sum")
        .reset_index()
    )
    per_score["pct_cds"] = 100.0 * per_score["n_cds"] / per_score["n_posts"]
    rows = {cat: df.groupby("severity")[col].mean().mul(100.0)
            for cat, col in zip(patterns.keys(), cat_cols)}
    cat_band = pd.DataFrame(rows).T.reindex(columns=BAND_ORDER)
    cat_overall = df[cat_cols].mean() * 100.0
    cat_overall.index = list(patterns.keys())
    return per_score, cat_band, cat_overall


# Heatmap colour matched to the SA cosine figure in sensitivity/sa_analyze.py.
HEATMAP_CMAP = "Oranges"
# Reference panel-(b) band labels (single line, Title case) so the x-axis
# matches "Minimal / Mild / Moderate / Mod. Severe / Severe".
FIG_BAND_LABELS = ["Minimal", "Mild", "Moderate", "Mod. Severe", "Severe"]


def make_figure(results: dict, generators: list, fig_path: str):
    """Write the two-panel figure: overall trend per generator (left) + category heatmap (right).

    The heatmap is coloured by the primary generator; each cell reads
    "primary (secondary)".

    Args:
        results (dict): tag -> dict(per_score, cat_band, r_agg), in `generators` order.
        generators (list): the arm's (tag, label, colour, path) entries.
        fig_path (str): output path (.png).
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.4, 3.2),
                                   gridspec_kw={"width_ratios": [1, 1.6]})

    # ── Left: overall % CDS vs PHQ-9 score, one line per generator ──────────
    for tag, label, colour, _ in generators:
        ps = results[tag]["per_score"]
        ax0.plot(ps["phq9"], ps["pct_cds"], "o-", color=colour, lw=2, ms=4,
                 label=label)
    ax0.set_xlabel("PHQ-9 sum-score")
    ax0.set_ylabel("% of posts containing CDS")
    # paper range, widened only if an arm falls outside it
    lo = min(results[g[0]]["per_score"]["pct_cds"].min() for g in generators)
    hi = max(results[g[0]]["per_score"]["pct_cds"].max() for g in generators)
    ax0.set_ylim(min(33, np.floor(lo) - 1), max(78, np.ceil(hi) + 1))
    ax0.grid(axis="y", linestyle=":", alpha=0.5)
    ax0.legend(loc="lower right", frameon=False, fontsize=9)
    ax0.text(0.5, -0.34, "(a) CDS vs PHQ-9", transform=ax0.transAxes,
             ha="center", va="top", fontsize=9.5)

    # ── Right: category x severity-band heatmap (Oranges), primary (secondary) ─
    primary, secondary = generators[0][0], generators[1][0]
    cat_band = results[primary]["cat_band"]
    other = results[secondary]["cat_band"].reindex(index=cat_band.index)
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
            ax1.text(j, i, f"{val:.1f} ({other.values[i, j]:.1f})",
                     ha="center", va="center", fontsize=6.2,
                     color="white" if val > thresh else "black")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.04)
    cbar.set_label("% of posts")
    ax1.text(0.5, -0.34, "(b) CDS category by PHQ-9 band, Qwen (Gemma)",
             transform=ax1.transAxes, ha="center", va="top", fontsize=9.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote figure to {fig_path}")


def run_arm(generators: list, patterns: dict, fig_path: str, out: str | None):
    """Score one arm's corpora, print the per-score and per-category tables, write the figure."""
    results = {}
    for tag, label, _, path in generators:
        print(f"Loading {label} posts from {path} ...")
        df, cat_cols = score_posts(path, patterns)
        per_score, cat_band, cat_overall = summarize(df, patterns, cat_cols)
        cds = df["is_cds"].astype(float).values
        phq = df["phq9"].astype(float).values
        results[tag] = {
            "label": label, "n": len(df),
            "per_score": per_score, "cat_band": cat_band, "cat_overall": cat_overall,
            "r_post": np.corrcoef(phq, cds)[0, 1],
            "rank_r": pd.Series(phq).corr(pd.Series(cds), method="spearman"),
            "r_agg": per_score["phq9"].corr(per_score["pct_cds"]),
        }
        print(f"  {len(df)} posts scored, {100.0 * df.is_cds.mean():.2f}% CDS overall")
    print()

    # rows sorted by the primary generator's overall prevalence (most common at top)
    primary = generators[0][0]
    order = results[primary]["cat_overall"].sort_values(ascending=False).index
    for tag in results:
        results[tag]["cat_band"] = results[tag]["cat_band"].reindex(index=order)

    # ── Per-score table, one % column per generator ──────────────────────────
    tags = [g[0] for g in generators]
    print("Average % CDS posts per PHQ-9 score (any category)")
    print("-" * 48)
    print(f"{'PHQ-9':>5}" + "".join(f"{results[t]['label']:>18}" for t in tags))
    merged = results[tags[0]]["per_score"][["phq9"]].copy()
    for t in tags:
        merged[t] = results[t]["per_score"]["pct_cds"].values
    for _, r in merged.iterrows():
        print(f"{int(r.phq9):>5}" + "".join(f"{r[t]:>17.2f}%" for t in tags))
    print()

    # ── Category x severity-band table per generator ─────────────────────────
    band_mid = {b[2]: (b[0] + b[1]) / 2 for b in SEVERITY_BANDS}
    console_band = {"minimal": "minimal", "mild": "mild", "moderate": "moderate",
                    "moderately\nsevere": "mod.sev", "severe": "severe"}
    hdr = "".join(f"{console_band[b]:>12}" for b in BAND_ORDER)
    mids = np.array([band_mid[b] for b in BAND_ORDER])
    for t in tags:
        res = results[t]
        print(f"Average % CDS posts per category, by PHQ-9 severity band: {res['label']}")
        print("-" * 78)
        print(f"{'category':<28}{hdr}{'overall':>10}{'trend r':>9}")
        for cat in res["cat_band"].index:
            vals = "".join(f"{res['cat_band'].loc[cat, b]:>11.1f}%" for b in BAND_ORDER)
            r_cat = np.corrcoef(mids, res["cat_band"].loc[cat, BAND_ORDER].values)[0, 1]
            print(f"{cat:<28}{vals}{res['cat_overall'][cat]:>9.1f}%{r_cat:>+9.2f}")
        print()

    # ── Does CDS get more probable with higher PHQ-9? ────────────────────────
    print("Correlation between PHQ-9 and CDS probability (any category)")
    print("-" * 60)
    for t in tags:
        res = results[t]
        print(f"  {res['label']}")
        print(f"    per-post Pearson (point-biserial) : {res['r_post']:+.3f}")
        print(f"    per-post Spearman                 : {res['rank_r']:+.3f}")
        print(f"    per-score Pearson (% vs PHQ-9)    : {res['r_agg']:+.3f}")
        verdict = "YES" if res["r_post"] > 0 else "NO"
        print(f"    => CDS more probable for higher PHQ-9? {verdict}")

    # ── Figure ───────────────────────────────────────────────────────────────
    make_figure(results, generators, fig_path)

    # ── Optional CSVs (long format: one block per generator) ─────────────────
    if out:
        per = pd.concat([results[t]["per_score"].assign(generator=t) for t in tags],
                        ignore_index=True)
        per.to_csv(out, index=False)
        cat_out = out.replace(".csv", "_by_category.csv")
        if cat_out == out:
            cat_out = out + ".by_category.csv"
        cat = pd.concat([results[t]["cat_band"].assign(generator=t) for t in tags])
        cat.to_csv(cat_out)
        print(f"\nWrote per-PHQ-9 table to {out}")
        print(f"Wrote per-category table to {cat_out}")



def main():
    """Run the check for each requested prompt arm."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ngrams", default="data/distorted_language_ngrams.tsv",
                        help="TSV of distorted-language n-grams (categories|markers|variants).")
    parser.add_argument("--arm", nargs="+", choices=list(ARMS), default=["human"],
                        help="Prompt arm(s): human (iter_10, the paper's corpora) and/or "
                             "minimal (iter_0).")
    parser.add_argument("--fig", default="plots/cds_validation.png",
                        help="Output path for the figure (`_minimal` is appended for "
                             "the minimal arm).")
    parser.add_argument("--out", default=None,
                        help="Optional path to write the per-PHQ-9 CDS table as CSV, long "
                             "format with a `generator` column (a *_by_category.csv "
                             "sibling is written alongside it; `_minimal` is appended for "
                             "the minimal arm).")
    args = parser.parse_args()

    print(f"Loading n-grams from {args.ngrams} ...")
    by_cat = load_ngrams_by_category(args.ngrams)
    patterns = compile_category_patterns(by_cat)
    total_ngrams = sum(len(v) for v in by_cat.values())
    print(f"Loaded {total_ngrams} n-grams across {len(patterns)} categories.\n")

    for arm in args.arm:
        suffix = "" if arm == "human" else f"_{arm}"
        print(f"=== {ARM_LABELS[arm]} prompt ===")
        run_arm(ARMS[arm], patterns, args.fig.replace(".png", f"{suffix}.png"),
                args.out.replace(".csv", f"{suffix}.csv") if args.out else None)
        print()


if __name__ == "__main__":
    main()
