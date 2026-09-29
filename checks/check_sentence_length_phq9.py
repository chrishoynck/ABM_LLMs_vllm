"""Check how sentence length changes with PHQ-9 in the generated posts.

Same data as `check_cds_tracks_phq9.py`: the human-optimized-prompt training
corpus of each generator (Qwen and Gemma: same 3,000 personas, scores and
neighbour draws, so the two are paired; 10 posts per persona). Each post is
split into sentences on `.`, `!`, `?` (a run like `...` or `?!` counts once, a
trailing fragment without a terminator is a sentence too) and words are counted
as runs of letters, digits and apostrophes (emoji do not count, `@user_12` is
one word). Per post we keep words per sentence, sentences per post and words
per post, and report them per PHQ-9 score and per severity band. Confidence
intervals are over personas (each persona has one PHQ-9 score and 10 posts, so
posts are not independent). Writes a two-panel figure (words per sentence vs
PHQ-9 per generator + per-band spread of the persona means). `--arm minimal`
runs the same check on the minimal-prompt (iter_0) corpora of the same personas.

Significance, per generator and metric, all on persona means (Welch t-test,
Cohen's d): non-depressed (PHQ-9 < 10) vs depressed (PHQ-9 >= 10, the standard
screening cut-off), and each pair of adjacent bands (Holm over the 4). `--tex`
writes the non-depressed vs depressed LaTeX table, with the engagement rows of
`check_engagement_phq9.py` (read from its significance CSVs) below the sentence rows.

Run: see checks/README.md.
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Same corpora, colours and bands as check_cds_tracks_phq9.py ("human", the
# iter_10 prompt). "minimal" is the iter_0 prompt over the same 3,000 personas,
# scores and neighbour draws, so all four corpora are paired.
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
MODEL_NAMES = {"Qwen": "Qwen3.5-27B", "Gemma": "Gemma-4-31B-it"}  # as in the paper

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

# The three per-post measures, (column, console label).
METRICS = [
    ("words_per_sentence", "words/sentence"),
    ("n_sentences", "sentences/post"),
    ("n_words", "words/post"),
]

SENT_END = re.compile(r"[.!?]+(?=\s|$)")
WORD = re.compile(r"[A-Za-z0-9_']+(?:[.,][0-9]+)*")


def _severity(phq9: int) -> str:
    """Map a PHQ-9 sum score to its severity band label."""
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= phq9 <= hi:
            return label
    return "unknown"


def sentence_stats(text: str) -> tuple[int, int]:
    """Return (words, sentences) of one post; sentences without words are dropped."""
    parts = SENT_END.split(text)
    counts = [len(WORD.findall(p)) for p in parts]
    counts = [c for c in counts if c > 0]
    return sum(counts), len(counts)


def score_posts(path: str) -> pd.DataFrame:
    """Load one posts CSV and add severity, n_words, n_sentences, words_per_sentence."""
    df = pd.read_csv(path).dropna(subset=["tweet", "phq9"]).copy()
    df["phq9"] = df["phq9"].astype(int)
    df["severity"] = df["phq9"].apply(_severity)
    stats = df["tweet"].astype(str).apply(sentence_stats)
    df["n_words"] = [s[0] for s in stats]
    df["n_sentences"] = [s[1] for s in stats]
    df = df[df["n_sentences"] > 0].copy()
    df["words_per_sentence"] = df["n_words"] / df["n_sentences"]
    return df


def per_group(agents: pd.DataFrame, key: str) -> pd.DataFrame:
    """Mean and 95% CI half-width of each metric over persona means, per `key`."""
    out = []
    for val, g in agents.groupby(key):
        row = {key: val, "n_personas": len(g)}
        for col, _ in METRICS:
            row[col] = g[col].mean()
            row[f"{col}_ci"] = 1.96 * g[col].std(ddof=1) / np.sqrt(len(g))
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
    """Write the two-panel figure: words/sentence vs PHQ-9 (left) + per-band boxes (right).

    Args:
        results (dict): tag -> dict(per_score, agents), in `generators` order.
        generators (list): one ARMS entry.
        fig_path (str): output path (.png).
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.4, 3.2),
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    # ── Left: mean words per sentence per PHQ-9 score, 95% CI over personas ──
    for tag, label, colour, _ in generators:
        ps = results[tag]["per_score"]
        y, ci = ps["words_per_sentence"], ps["words_per_sentence_ci"]
        ax0.fill_between(ps["phq9"], y - ci, y + ci, color=colour, alpha=0.2, lw=0)
        ax0.plot(ps["phq9"], y, "o-", color=colour, lw=2, ms=4, label=label)
    ax0.set_xlabel("PHQ-9 sum-score")
    ax0.set_ylabel("Words per sentence")
    ax0.grid(axis="y", linestyle=":", alpha=0.5)
    ax0.legend(loc="best", frameon=False, fontsize=9)
    ax0.text(0.5, -0.34, "(a) Sentence length vs PHQ-9", transform=ax0.transAxes,
             ha="center", va="top", fontsize=9.5)

    # ── Right: persona-mean words/sentence per band, generators side by side ─
    width = 0.36
    offsets = np.linspace(-width / 2, width / 2, len(generators))
    for (tag, label, colour, _), off in zip(generators, offsets):
        agents = results[tag]["agents"]
        data = [agents.loc[agents.severity == b, "words_per_sentence"].values
                for b in BAND_ORDER]
        ax1.boxplot(data, positions=np.arange(len(BAND_ORDER)) + off, widths=width * 0.85,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colour, alpha=0.35, edgecolor=colour),
                    medianprops=dict(color=colour, lw=2),
                    whiskerprops=dict(color=colour), capprops=dict(color=colour))
    ax1.set_xticks(range(len(BAND_ORDER)))
    ax1.set_xticklabels(FIG_BAND_LABELS, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Words per sentence\n(persona mean)")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.text(0.5, -0.34, "(b) Sentence length by PHQ-9 band", transform=ax1.transAxes,
             ha="center", va="top", fontsize=9.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote figure to {fig_path}")


def run_arm(generators: list, fig_path: str, out: str | None) -> pd.DataFrame:
    """Score one arm's corpora, print the tables, write the figure (+ CSVs); return the tests."""
    results = {}
    for tag, label, _, path in generators:
        print(f"Loading {label} posts from {path} ...")
        df = score_posts(path)
        cols = [c for c, _ in METRICS]
        agents = df.groupby(["agent_id", "phq9", "severity"])[cols].mean().reset_index()
        results[tag] = {
            "label": label, "agents": agents,
            "per_score": per_group(agents, "phq9"),
            "per_band": per_group(agents, "severity").set_index("severity").reindex(BAND_ORDER),
            "r_post": df["phq9"].corr(df["words_per_sentence"]),
            "rank_r": df["phq9"].corr(df["words_per_sentence"], method="spearman"),
            "r_agent": agents["phq9"].corr(agents["words_per_sentence"]),
        }
        print(f"  {len(df)} posts, {len(agents)} personas, "
              f"{df.words_per_sentence.mean():.2f} words/sentence, "
              f"{df.n_sentences.mean():.2f} sentences/post, {df.n_words.mean():.1f} words/post")
    print()
    tags = [g[0] for g in generators]

    # ── Per-score table, words/sentence per generator ────────────────────────
    print("Mean words per sentence per PHQ-9 score (95% CI over personas)")
    print("-" * 48)
    print(f"{'PHQ-9':>5}" + "".join(f"{results[t]['label']:>21}" for t in tags))
    merged = results[tags[0]]["per_score"][["phq9"]].copy()
    for _, r in merged.iterrows():
        line = f"{int(r.phq9):>5}"
        for t in tags:
            ps = results[t]["per_score"].set_index("phq9").loc[r.phq9]
            line += f"{ps.words_per_sentence:>14.2f} ± {ps.words_per_sentence_ci:.2f}"
        print(line)
    print()

    # ── Per-band table, all three metrics per generator ──────────────────────
    hdr = "".join(f"{b:>16}" for b in BAND_ORDER)
    for t in tags:
        pb = results[t]["per_band"]
        print(f"Per PHQ-9 severity band (mean ± 95% CI over personas): {results[t]['label']}")
        print("-" * 96)
        print(f"{'metric':<16}{hdr}")
        for col, name in METRICS:
            vals = "".join(f"{pb.loc[b, col]:>9.2f} ± {pb.loc[b, col + '_ci']:<4.2f}"
                           for b in BAND_ORDER)
            print(f"{name:<16}{vals}")
        print(f"{'personas':<16}" + "".join(f"{int(pb.loc[b, 'n_personas']):>16}"
                                            for b in BAND_ORDER))
        print()

    # ── Does sentence length change with PHQ-9? ──────────────────────────────
    print("Correlation between PHQ-9 and words per sentence")
    print("-" * 60)
    for t in tags:
        res = results[t]
        print(f"  {res['label']}")
        print(f"    per-post Pearson                  : {res['r_post']:+.3f}")
        print(f"    per-post Spearman                 : {res['rank_r']:+.3f}")
        print(f"    per-persona Pearson (10-post mean): {res['r_agent']:+.3f}")
    print()

    # ── Significance: non-depressed vs depressed, adjacent bands ─────────────
    print("Significance (Welch t on persona means; band rows Holm over 4)")
    sig = []
    for t in tags:
        print("-" * 84)
        print(f"{results[t]['label']:<16}{'comparison':<22}{'a':>8}{'b':>8}"
              f"{'b - a':>8}{'d':>8}{'p_holm':>10}")
        for col, name in METRICS:
            s = significance(results[t]["agents"], col)
            sig.append(s.assign(generator=t, metric=col))
            for i, r in s.iterrows():
                print(f"{name if i == 0 else '':<16}{r.comparison:<22}{r.mean_a:>8.2f}"
                      f"{r.mean_b:>8.2f}{r['diff']:>+8.2f}{r.d:>+8.2f}{r.p_holm:>10.1e}")
    print()

    make_figure(results, generators, fig_path)

    # ── Optional CSVs (long format: one block per generator) ─────────────────
    sig = pd.concat(sig, ignore_index=True)
    if out:
        per = pd.concat([results[t]["per_score"].assign(generator=t) for t in tags],
                        ignore_index=True)
        per.to_csv(out, index=False)
        band_out = out.replace(".csv", "_by_band.csv")
        if band_out == out:
            band_out = out + ".by_band.csv"
        band = pd.concat([results[t]["per_band"].assign(generator=t) for t in tags])
        band.to_csv(band_out)
        sig_out = band_out.replace("_by_band", "_significance")
        if sig_out == band_out:
            sig_out = out + ".significance.csv"
        sig.to_csv(sig_out, index=False)
        print(f"\nWrote per-PHQ-9 table to {out}")
        print(f"Wrote per-band table to {band_out}")
        print(f"Wrote significance tests to {sig_out}")
    return sig


def fmt_p(p: float) -> str:
    """p-value for LaTeX: two decimals, or a x10^e form below 0.01."""
    if p >= 0.01:
        return f"{p:.2f}"
    e = int(np.floor(np.log10(p)))
    return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"


# Significance CSVs of checks/check_engagement_phq9.py (--out data/finetune/engagement_by_phq9.csv);
# the LaTeX table adds its two measures below words per sentence.
ENGAGEMENT_SIG = {"human": "data/finetune/engagement_by_phq9_significance.csv",
                  "minimal": "data/finetune/engagement_by_phq9_minimal_significance.csv"}

# Table blocks: (source, metric column, block title, decimals).
TEX_MEASURES = [
    ("sentence", "words_per_sentence", "Words per sentence", 2),
    ("engagement", "pct_engaging", r"Posts mentioning another user (\%)", 1),
    ("engagement", "cds_pref", "CDS minus non-CDS posts mentioned (percentage points)", 1),
]


def write_tex(sig: dict, path: str):
    """LaTeX table of the non-depressed vs depressed tests: sentence length + engagement.

    Args:
        sig (dict): arm -> significance frame from `run_arm`.
        path (str): output .tex path.
    """
    c = DEPRESSION_CUTOFF
    labels = {tag: label for gens in ARMS.values() for tag, label, _, _ in gens}
    source = {}
    for arm, s in sig.items():
        if not os.path.isfile(ENGAGEMENT_SIG[arm]):
            raise SystemExit(f"{ENGAGEMENT_SIG[arm]} not found; run checks/check_engagement_phq9.py "
                             f"--arm {' '.join(sig)} --out data/finetune/engagement_by_phq9.csv first.")
        source[arm] = {"sentence": s, "engagement": pd.read_csv(ENGAGEMENT_SIG[arm])}
    first = next(iter(sig.values()))
    first = first[(first.test == "depressed")].iloc[0]
    L = [r"\begin{table}[ht]", r"\centering", r"\small",
         r"\caption{Do depressed personas write and engage differently? Non-depressed "
         rf"(PHQ-9 $<{c}$, $n={int(first.n_a)}$) vs depressed (PHQ-9 $\geq{c}$, "
         rf"$n={int(first.n_b)}$) personas in the fine-tuning corpora (3,000 personas "
         r"with 10 posts each; the same personas, scores and neighbor posts for every "
         r"prompt and model). \emph{Words per sentence}: posts are split into sentences "
         r"at \texttt{.}, \texttt{!} and \texttt{?}, and words are alphanumeric tokens "
         r"(emoji excluded). \emph{Posts mentioning another user}: share of a persona's "
         r"posts that mention (\texttt{@user\_<id>}) the author of at least one of the "
         r"five neighbor posts shown in the prompt, a proxy for engagement with peers. "
         r"\emph{CDS minus non-CDS}: share of the shown neighbor posts containing a "
         r"cognitive distortion that the persona mentions, minus the same share for "
         r"shown posts without one, a proxy for preferential engagement with distorted "
         r"content. Values are means over personas of each persona's 10-post value. "
         r"$\Delta$ is depressed minus non-depressed, $d$ is Cohen's $d$ and $p$ is a "
         r"Welch $t$-test on persona means.}",
         r"\label{tab:sentence_length_depressed}",
         r"\begin{tabular}{llrrrrr}", r"\hline",
         rf"Prompt & Model & PHQ-9 $<{c}$ & PHQ-9 $\geq{c}$ & $\Delta$ & $d$ & $p$ \\",
         r"\hline"]
    for src, metric, title, dec in TEX_MEASURES:
        L.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{title}}}}} \\")
        for arm in sig:
            s = source[arm][src]
            dep = s[(s.test == "depressed") & (s.metric == metric)]
            for i, (_, r) in enumerate(dep.iterrows()):
                a, b = (f"${x:.{dec}f}$" if x < 0 else f"{x:.{dec}f}" for x in (r.mean_a, r.mean_b))
                L.append(f"{ARM_LABELS[arm] if i == 0 else ''} & "
                         f"{MODEL_NAMES[labels[r.generator]]} & {a} & {b} & "
                         f"${r['diff']:+.{dec}f}$ & ${r.d:+.2f}$ & {fmt_p(r.p)} \\\\")
        L.append(r"\hline")
    L += [r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\nWrote LaTeX table to {path}")


def main():
    """Run the check for each requested arm, then optionally write the LaTeX table."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", nargs="+", choices=list(ARMS), default=["human"],
                        help="Generation prompt(s): human-optimized (iter_10) and/or "
                             "minimal (iter_0).")
    parser.add_argument("--fig", default="plots/sentence_length_validation.png",
                        help="Output path for the figure (`_minimal` is appended for "
                             "the minimal arm).")
    parser.add_argument("--out", default=None,
                        help="Optional path to write the per-PHQ-9 table as CSV, long "
                             "format with a `generator` column (*_by_band.csv and "
                             "*_significance.csv siblings are written alongside it; "
                             "`_minimal` is appended for the minimal arm).")
    parser.add_argument("--tex", default=None,
                        help="Optional path for a LaTeX table of the non-depressed vs "
                             "depressed tests, one row per arm and generator: words per "
                             "sentence plus the two engagement measures of "
                             "check_engagement_phq9.py (run that with --out first).")
    args = parser.parse_args()

    sig = {}
    for arm in args.arm:
        suffix = "" if arm == "human" else f"_{arm}"
        print(f"=== {ARM_LABELS[arm]} prompt ===")
        sig[arm] = run_arm(ARMS[arm], args.fig.replace(".png", f"{suffix}.png"),
                           args.out.replace(".csv", f"{suffix}.csv") if args.out else None)
        print()
    if args.tex:
        write_tex(sig, args.tex)


if __name__ == "__main__":
    main()
