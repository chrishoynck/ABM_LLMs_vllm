"""Check how the generated personas engage with other users' posts as PHQ-9 rises.

Two online patterns of depressed users from the literature:
  (a) less engagement with peers' content (De Choudhury et al. 2013), and
  (b) more engagement with cognitively distorted content (Hasan et al. 2025).

Same data as `check_cds_tracks_phq9.py`: the paired training corpora of each
generator (same 3,000 personas, scores and neighbour draws; 10 posts per persona).
Every post was written after seeing five posts from others, listed in the prompt
as `- @user_<id>: <post>`. Those five come from a fixed pool (`gather_neighbor_pool`)
with a per-(agent, round) seed, so we rebuild them exactly, as in
`TestLLMs._build_aligned_tweet_messages`. A shown post counts as engaged when the
new post mentions its author (`@user_<id>`); the share of mentions that hit a
shown author is printed as a check on the rebuild. The `interaction` column in the
corpora is a constant write flag and is not used.

  (a) % of posts that mention at least one shown author, per PHQ-9 score and band.
  (b) per band, % of shown CDS posts that get mentioned vs % of shown non-CDS posts
      (same CDS detector as the CDS check). Rounds where two shown posts share an
      author id (~1%) are left out of (b), since the mention cannot be assigned.

CIs are over personas. Significance on persona means (Welch t, Cohen's d):
non-depressed (PHQ-9 < 10) vs depressed (>= 10), and adjacent bands; for (b) the
tested value is each persona's CDS rate minus its non-CDS rate. `--arm minimal`
runs it on the minimal-prompt corpora of the same personas. Run: see
checks/README.md.
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

from utils.create_data.loaders import DEFAULT_NEIGHBOR_ROOTS, gather_neighbor_pool
from utils.tools.cds import compile_category_patterns, load_ngrams_by_category

# Same corpora, colours and bands as check_sentence_length_phq9.py.
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

# Neighbour settings of every corpus above (see their .meta.json).
NEIGHBOR_SEED = 42
NUM_NEIGHBORS = 5

SEVERITY_BANDS = [
    (0, 4, "minimal"),
    (5, 9, "mild"),
    (10, 14, "moderate"),
    (15, 19, "mod.sev"),
    (20, 27, "severe"),
]
BAND_ORDER = [b[2] for b in SEVERITY_BANDS]
FIG_BAND_LABELS = ["Minimal", "Mild", "Moderate", "Mod. Severe", "Severe"]
# PHQ-9 >= 10 counts as depressed (standard screening cut-off).
DEPRESSION_CUTOFF = 10

# Per-persona measures, (column, console label). All in %.
METRICS = [
    ("pct_engaging", "posts mentioning"),
    ("cds_rate", "CDS mentioned"),
    ("noncds_rate", "non-CDS mentioned"),
    ("cds_pref", "CDS - non-CDS"),
]
# The two measures that get significance tests.
TESTED = [("pct_engaging", "posts mentioning"), ("cds_pref", "CDS - non-CDS")]

MENTION = re.compile(r"@user_(\d+)")


def _severity(phq9: int) -> str:
    """Map a PHQ-9 sum score to its severity band label."""
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= phq9 <= hi:
            return label
    return "unknown"


def load_pool(ngrams: str) -> tuple[list, np.ndarray]:
    """The neighbour pool the corpora were generated with, plus a CDS flag per pool post."""
    pool = gather_neighbor_pool(list(DEFAULT_NEIGHBOR_ROOTS))
    patterns = compile_category_patterns(load_ngrams_by_category(ngrams))
    text = pd.Series([t for _, t in pool])
    is_cds = np.zeros(len(pool), dtype=bool)
    for pat in patterns.values():
        is_cds |= text.str.contains(pat).values
    print(f"  {100.0 * is_cds.mean():.1f}% of pool posts contain a CDS")
    return pool, is_cds


def score_posts(path: str, pool: list, pool_cds: np.ndarray):
    """Rebuild the five shown posts of every generated post and flag which got mentioned.

    Returns:
        tuple: (posts frame with `severity` and `engaging`, one row per (post, shown
        post) with `engaged`, `cds`, `dup_author`, share of mentions that hit a
        shown author, share of posts with a repeated author id among the shown).
    """
    df = pd.read_csv(path).dropna(subset=["tweet", "phq9"]).copy()
    df["phq9"] = df["phq9"].astype(int)
    df["severity"] = df["phq9"].apply(_severity)

    rows, engaging = [], []
    n_mentions = n_matched = n_dup = 0
    for aid, step, tweet in zip(df.agent_id, df.step, df.tweet.astype(str)):
        # Same sub-RNG as TestLLMs: the round counter is step + 1 when the prompt is built.
        rng = np.random.default_rng(np.random.SeedSequence([NEIGHBOR_SEED, int(aid), int(step) + 1]))
        idx = rng.choice(len(pool), size=NUM_NEIGHBORS, replace=False)
        authors = [pool[i][0] for i in idx]
        mentioned = set(MENTION.findall(tweet))
        n_mentions += len(mentioned)
        n_matched += len(mentioned & set(authors))
        engaging.append(bool(mentioned & set(authors)))
        dup = len(set(authors)) < len(authors)
        n_dup += dup
        for i, a in zip(idx, authors):
            rows.append((aid, a in mentioned, pool_cds[i], dup))
    df["engaging"] = engaging
    shown = pd.DataFrame(rows, columns=["agent_id", "engaged", "cds", "dup_author"])
    return df, shown, n_matched / max(n_mentions, 1), n_dup / len(df)


def persona_table(df: pd.DataFrame, shown: pd.DataFrame) -> pd.DataFrame:
    """One row per persona: % posts mentioning a shown author, % CDS / non-CDS posts mentioned."""
    agents = (df.groupby(["agent_id", "phq9", "severity"])["engaging"].mean()
              .mul(100.0).rename("pct_engaging").reset_index())
    s = shown[~shown["dup_author"]]
    rate = s.groupby(["agent_id", "cds"])["engaged"].mean().mul(100.0).unstack()
    agents["cds_rate"] = agents["agent_id"].map(rate[True])
    agents["noncds_rate"] = agents["agent_id"].map(rate[False])
    agents["cds_pref"] = agents["cds_rate"] - agents["noncds_rate"]
    return agents


def per_group(agents: pd.DataFrame, key: str) -> pd.DataFrame:
    """Mean and 95% CI half-width of each metric over persona means, per `key`."""
    out = []
    for val, g in agents.groupby(key):
        row = {key: val, "n_personas": len(g)}
        for col, _ in METRICS:
            x = g[col].dropna()
            row[col] = x.mean()
            row[f"{col}_ci"] = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
        out.append(row)
    return pd.DataFrame(out)


def holm(p) -> np.ndarray:
    """Holm-adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def compare(x: pd.Series, y: pd.Series, name: str) -> dict:
    """Group means, difference y - x, Cohen's d and Welch p."""
    x, y = x.dropna(), y.dropna()
    diff = y.mean() - x.mean()
    return {"comparison": name, "n_a": len(x), "n_b": len(y),
            "mean_a": x.mean(), "mean_b": y.mean(), "diff": diff,
            "d": diff / np.sqrt((x.var() + y.var()) / 2),
            "p": stats.ttest_ind(y, x, equal_var=False).pvalue}


def significance(agents: pd.DataFrame, col: str) -> pd.DataFrame:
    """Non-depressed vs depressed, then adjacent bands (Holm over those 4)."""
    c = DEPRESSION_CUTOFF
    dep = compare(agents.loc[agents.phq9 < c, col], agents.loc[agents.phq9 >= c, col],
                  f"PHQ-9 <{c} vs >={c}")
    bands = pd.DataFrame([compare(agents.loc[agents.severity == lo, col],
                                  agents.loc[agents.severity == hi, col], f"{lo}->{hi}")
                          for lo, hi in zip(BAND_ORDER, BAND_ORDER[1:])])
    bands["p_holm"] = holm(bands["p"])
    return pd.concat([pd.DataFrame([dep]).assign(test="depressed", p_holm=dep["p"]),
                      bands.assign(test="band")], ignore_index=True)


def make_figure(results: dict, generators: list, fig_path: str):
    """Write the two-panel figure: % posts mentioning vs PHQ-9 (left) + CDS vs non-CDS by band (right).

    Args:
        results (dict): tag -> dict(per_score, per_band), in `generators` order.
        generators (list): one ARMS entry.
        fig_path (str): output path (.png).
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.4, 3.2),
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    # ── Left: % of posts mentioning a shown author, 95% CI over personas ────
    for tag, label, colour, _ in generators:
        ps = results[tag]["per_score"]
        y, ci = ps["pct_engaging"], ps["pct_engaging_ci"]
        ax0.fill_between(ps["phq9"], y - ci, y + ci, color=colour, alpha=0.2, lw=0)
        ax0.plot(ps["phq9"], y, "o-", color=colour, lw=2, ms=4, label=label)
    ax0.set_xlabel("PHQ-9 sum-score")
    ax0.set_ylabel("% of posts mentioning\nanother user")
    ax0.set_ylim(0, 100)
    ax0.grid(axis="y", linestyle=":", alpha=0.5)
    ax0.legend(loc="lower right", frameon=False, fontsize=9)
    ax0.text(0.5, -0.34, "(a) Engagement with others vs PHQ-9", transform=ax0.transAxes,
             ha="center", va="top", fontsize=9.5)

    # ── Right: % of shown CDS / non-CDS posts that get mentioned, per band ──
    x = np.arange(len(BAND_ORDER))
    for (tag, label, colour, _), off in zip(generators, [-0.06, 0.06]):
        pb = results[tag]["per_band"]
        for col, style, face in [("cds_rate", "-", colour), ("noncds_rate", "--", "white")]:
            ax1.errorbar(x + off, pb[col], yerr=pb[f"{col}_ci"], fmt="o" + style,
                         color=colour, mfc=face, lw=1.8, ms=5, capsize=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(FIG_BAND_LABELS, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("% of shown posts\nmentioned")
    ax1.set_ylim(bottom=0)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(handles=[Line2D([], [], color="0.3", ls="-", marker="o", label="CDS post"),
                        Line2D([], [], color="0.3", ls="--", marker="o", mfc="white",
                               label="non-CDS post")],
               loc="lower right", frameon=False, fontsize=9)
    ax1.text(0.5, -0.34, "(b) Engagement with distorted posts by PHQ-9 band",
             transform=ax1.transAxes, ha="center", va="top", fontsize=9.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote figure to {fig_path}")


def run_arm(generators: list, pool: list, pool_cds: np.ndarray,
            fig_path: str, out: str | None):
    """Score one arm's corpora, print the tables, write the figure (+ CSVs)."""
    results = {}
    for tag, label, _, path in generators:
        print(f"Loading {label} posts from {path} ...")
        df, shown, match, dup = score_posts(path, pool, pool_cds)
        agents = persona_table(df, shown)
        results[tag] = {
            "label": label, "agents": agents,
            "per_score": per_group(agents, "phq9"),
            "per_band": per_group(agents, "severity").set_index("severity").reindex(BAND_ORDER),
        }
        print(f"  {len(df)} posts, {len(agents)} personas, "
              f"{100.0 * df.engaging.mean():.1f}% mention a shown author; "
              f"{100.0 * match:.1f}% of mentions hit a shown author (rebuild check); "
              f"{100.0 * dup:.1f}% of posts left out of (b)")
    print()
    tags = [g[0] for g in generators]

    # ── Per-band table, all metrics per generator ────────────────────────────
    hdr = "".join(f"{b:>16}" for b in BAND_ORDER)
    for t in tags:
        pb = results[t]["per_band"]
        print(f"Per PHQ-9 severity band, % (mean ± 95% CI over personas): {results[t]['label']}")
        print("-" * 100)
        print(f"{'metric':<20}{hdr}")
        for col, name in METRICS:
            vals = "".join(f"{pb.loc[b, col]:>9.1f} ± {pb.loc[b, col + '_ci']:<4.1f}"
                           for b in BAND_ORDER)
            print(f"{name:<20}{vals}")
        print(f"{'personas':<20}" + "".join(f"{int(pb.loc[b, 'n_personas']):>16}"
                                            for b in BAND_ORDER))
        print()

    # ── Significance: non-depressed vs depressed, adjacent bands ─────────────
    print("Significance (Welch t on persona means; band rows Holm over 4)")
    sig = []
    for t in tags:
        print("-" * 88)
        print(f"{results[t]['label']:<20}{'comparison':<22}{'a':>8}{'b':>8}"
              f"{'b - a':>8}{'d':>8}{'p_holm':>10}")
        for col, name in TESTED:
            s = significance(results[t]["agents"], col)
            sig.append(s.assign(generator=t, metric=col))
            for i, r in s.iterrows():
                print(f"{name if i == 0 else '':<20}{r.comparison:<22}{r.mean_a:>8.1f}"
                      f"{r.mean_b:>8.1f}{r['diff']:>+8.1f}{r.d:>+8.2f}{r.p_holm:>10.1e}")
    print()

    make_figure(results, generators, fig_path)

    # ── Optional CSVs (long format: one block per generator) ─────────────────
    if out:
        per = pd.concat([results[t]["per_score"].assign(generator=t) for t in tags],
                        ignore_index=True)
        per.to_csv(out, index=False)
        band_out = out.replace(".csv", "_by_band.csv")
        band = pd.concat([results[t]["per_band"].assign(generator=t) for t in tags])
        band.to_csv(band_out)
        sig_out = out.replace(".csv", "_significance.csv")
        pd.concat(sig, ignore_index=True).to_csv(sig_out, index=False)
        print(f"\nWrote per-PHQ-9 table to {out}")
        print(f"Wrote per-band table to {band_out}")
        print(f"Wrote significance tests to {sig_out}")


def main():
    """Run the check for each requested arm."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", nargs="+", choices=list(ARMS), default=["human"],
                        help="Generation prompt(s): human-optimized (iter_10) and/or "
                             "minimal (iter_0).")
    parser.add_argument("--ngrams", default="data/distorted_language_ngrams.tsv",
                        help="TSV of distorted-language n-grams (categories|markers|variants).")
    parser.add_argument("--fig", default="plots/engagement_validation.png",
                        help="Output path for the figure (`_minimal` is appended for "
                             "the minimal arm).")
    parser.add_argument("--out", default=None,
                        help="Optional path (.csv) for the per-PHQ-9 table, long format "
                             "with a `generator` column (*_by_band.csv and "
                             "*_significance.csv siblings are written alongside it; "
                             "`_minimal` is appended for the minimal arm).")
    args = parser.parse_args()

    print("Loading neighbour pool ...")
    pool, pool_cds = load_pool(args.ngrams)
    print()
    for arm in args.arm:
        suffix = "" if arm == "human" else f"_{arm}"
        print(f"=== {ARM_LABELS[arm]} prompt ===")
        run_arm(ARMS[arm], pool, pool_cds, args.fig.replace(".png", f"{suffix}.png"),
                args.out.replace(".csv", f"{suffix}.csv") if args.out else None)
        print()


if __name__ == "__main__":
    main()
