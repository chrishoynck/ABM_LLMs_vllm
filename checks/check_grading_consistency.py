"""Check whether `prompt_optimizer --mode grade-posts` agrees with the original teacher grades.

The paper's post-quality scores came from `rerun_test_tweets` (100 fresh agents, 3
posts each, human-optimized prompt: 7.41). The multi-model panel grades the 300-block
held-out sets (10 posts per block) with the same rating prompt but through
`grade-posts`, and Qwen lands far lower (5.8). This check separates the grader from
the data: `prepare` writes 3-post truncations of both 300-block sets and the paper's
100 test personas; the job then regrades the ORIGINAL iter_10 test posts (three
vLLM seeds, for the grader's own noise), the truncations, and fresh 3-post blocks
that Qwen and Gemma generate for those 100 personas; `report` compares regrade vs
original score per agent, 3-post vs 10-post grades per block, and the two generators
on the 100 x 3 protocol. `prompts` (CPU) tests whether the human-optimized prompt's
stored scores beat the TextGrad runs and the minimal prompt on those same 100 agents.
Run: see checks/README.md.
"""
import argparse
import os

import numpy as np
import pandas as pd

GRADES = "data/test_post/teacher_grades"
ORIG_POSTS = "data/prompt_optimization_h/qwen27_baseline/iter_10/test_posts.csv"
REGRADE_REPS = ["regrade_iter10_test100", "regrade_iter10_test100_rep2", "regrade_iter10_test100_rep3"]
# Stored per-agent quality scores of each post-generation prompt, all on the same 100 agents.
PROMPT_SCORES = {
    "human-optimized (iter_10)": "data/prompt_optimization_h/qwen27_baseline/iter_10/test_raw_scores.csv",
    "minimal (iter_0)": "data/prompt_optimization_h/qwen27_baseline/iter_0/test_raw_scores.csv",
    **{f"TextGrad seed {s}": f"data/test_post/optimized_tweets/Qwen3.5-27B_seed{s}/test_raw_scores.csv"
       for s in [24, 25, 28, 29, 53]},
}
SETS = {"qwen": "data/finetune/test_posts.csv", "gemma4": "data/finetune/gemma4/test_posts_gemma4.csv"}
BANDS = [(0, 4, "Minimal"), (5, 9, "Mild"), (10, 14, "Moderate"), (15, 19, "Mod. severe"), (20, 27, "Severe")]


def per_band(df, col="score"):
    """Mean of `col` per PHQ-9 severity band, as one formatted string."""
    return ", ".join(f"{n} {df.loc[df['phq9'].between(lo, hi), col].mean():.2f}" for lo, hi, n in BANDS)


def prepare(n_posts):
    """Write the 3-post truncations of the 300-block sets and the paper's 100 test personas (persona, phq9)."""
    os.makedirs(GRADES, exist_ok=True)
    for tag, path in SETS.items():
        df = pd.read_csv(path)
        out = f"{GRADES}/{tag}300_first{n_posts}.csv"
        df.groupby("agent_id", sort=False).head(n_posts).to_csv(out, index=False)
        print(f"[prepare] {out}: {df['agent_id'].nunique()} blocks x {n_posts} posts")
    orig = pd.read_csv(ORIG_POSTS).groupby("agent_id", sort=False)[["persona", "phq9"]].first()
    orig.to_csv(f"{GRADES}/personas_test100.csv", index=False)
    print(f"[prepare] {GRADES}/personas_test100.csv: {len(orig)} personas")


def report(n_posts):
    """Compare the regrade of the original test posts with their stored scores, and 3-post vs 10-post grades."""
    orig = pd.read_csv(ORIG_POSTS).groupby("agent_id")[["phq9", "agent_score"]].first()
    re = pd.read_csv(f"{GRADES}/regrade_iter10_test100.csv").set_index("agent_id")
    both = orig.join(re["score"]).dropna()
    diff = both["score"] - both["agent_score"]
    print(f"[original iter_10 test set, {len(both)} agents x 3 posts]")
    print(f"  stored score   mean {both['agent_score'].mean():.2f}  | {per_band(both, 'agent_score')}")
    print(f"  regrade        mean {both['score'].mean():.2f}  | {per_band(both, 'score')}")
    print(f"  regrade - stored: mean {diff.mean():+.2f}, mean |diff| {diff.abs().mean():.2f}, "
          f"Pearson r {both['score'].corr(both['agent_score']):.2f}, |diff| <= 1 in {(diff.abs() <= 1).mean():.0%}")
    reps = [pd.read_csv(f"{GRADES}/{r}.csv").set_index("agent_id")["score"] for r in REGRADE_REPS
            if os.path.isfile(f"{GRADES}/{r}.csv")]
    if len(reps) > 1:
        reps = pd.concat(reps, axis=1, keys=range(len(reps))).dropna()
        pair = np.mean([(reps[i] - reps[j]).abs().mean() for i in reps for j in reps if i < j])
        print(f"  grader noise over {len(reps.columns)} regrades of the same posts: means "
              f"{', '.join(f'{m:.2f}' for m in reps.mean())} (SD of mean {reps.mean().std(ddof=1):.2f}); "
              f"mean |diff| between two regrades of one set {pair:.2f}")

    fresh = {}
    for tag in SETS:
        f = f"{GRADES}/grades_100x3_{tag}.csv"
        if os.path.isfile(f):
            fresh[tag] = pd.read_csv(f).set_index("agent_id")
    if fresh:
        print(f"[paper protocol on the 100 test personas, fresh 3-post blocks from each generator]")
        for tag, d in fresh.items():
            d = d.dropna(subset=["score"])
            print(f"  {tag:7s} mean {d['score'].mean():.2f} +/- {d['score'].std():.2f} (n={len(d)}) | {per_band(d)}")
        if len(fresh) == 2:
            q, g = (fresh[t]["score"] for t in SETS)
            both = pd.concat([q, g], axis=1, keys=["q", "g"]).dropna()
            dq = both["q"] - both["g"]
            print(f"  paired Qwen - Gemma: mean {dq.mean():+.2f}, Qwen higher in {(dq > 0).mean():.0%}, tie {(dq == 0).mean():.0%}")

    for tag in SETS:
        full = pd.read_csv(f"{GRADES}/grades_{tag}300.csv").set_index("agent_id")
        first = pd.read_csv(f"{GRADES}/grades_{tag}300_first{n_posts}.csv").set_index("agent_id")
        both = full[["phq9", "score"]].join(first["score"], rsuffix="_first").dropna()
        print(f"[{tag} 300-block set, {len(both)} blocks]")
        print(f"  all 10 posts   mean {both['score'].mean():.2f}  | {per_band(both, 'score')}")
        print(f"  first {n_posts} posts  mean {both['score_first'].mean():.2f}  | {per_band(both, 'score_first')}")
        print(f"  first - all: mean {(both['score_first'] - both['score']).mean():+.2f}, "
              f"Pearson r {both['score'].corr(both['score_first']):.2f}")


def _paired(a, b):
    """Paired t-test and Wilcoxon of two per-agent score series: 'mean diff [95% CI], t p, Wilcoxon p'."""
    from scipy import stats
    d = (a - b).dropna()
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.sem())
    w = stats.wilcoxon(d[d != 0]).pvalue if (d != 0).any() else 1.0
    return (f"{d.mean():+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}], paired t p={stats.ttest_1samp(d, 0).pvalue:.2g}, "
            f"Wilcoxon p={w:.2g}, n={len(d)}")


def prompts():
    """Test the human-optimized prompt's stored quality scores against TextGrad and the minimal prompt (same 100 agents)."""
    from scipy import stats
    s = {k: pd.read_csv(v).set_index("agent_id")["score"] for k, v in PROMPT_SCORES.items()}
    human, minimal = s["human-optimized (iter_10)"], s["minimal (iter_0)"]
    tg = {k: v for k, v in s.items() if k.startswith("TextGrad")}
    tg_mean = pd.concat(tg.values(), axis=1).mean(axis=1)  # per-agent mean over the 5 runs
    best = max(tg, key=lambda k: tg[k].mean())
    print("[stored quality scores, 100 shared agents x 3 posts]")
    for k, v in s.items():
        print(f"  {k:26s} {v.mean():.2f} +/- {v.std():.2f}")
    print(f"  human - TextGrad best ({best}): {_paired(human, tg[best])}")
    print(f"  human - TextGrad per-agent mean of 5 runs: {_paired(human, tg_mean)}")
    print(f"  human - minimal:                 {_paired(human, minimal)}")
    print(f"  TextGrad best - minimal:         {_paired(tg[best], minimal)}")
    runs = np.array([v.mean() for v in tg.values()])
    t = stats.ttest_1samp(runs, human.mean())
    print(f"  run level: 5 TextGrad run means {runs.mean():.2f} +/- {runs.std(ddof=1):.2f} vs human {human.mean():.2f}: "
          f"one-sample t p={t.pvalue:.2g} (human is one run; grader noise on a 100-agent mean ~0.1-0.2, see report)")


def main():
    """Run the requested stage (prepare before the job's grading calls, report after; prompts is CPU-only)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["prepare", "report", "prompts"])
    ap.add_argument("--n-posts", type=int, default=3, help="posts kept per block in the truncated sets")
    a = ap.parse_args()
    {"prepare": lambda: prepare(a.n_posts), "report": lambda: report(a.n_posts), "prompts": prompts}[a.stage]()


if __name__ == "__main__":
    main()
