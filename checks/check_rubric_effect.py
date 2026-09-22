"""Does the 2026-05-30 rubric change explain the human-opt vs TextGrad collapse?

The paper's post-quality scores were produced by the teacher rubric as it stood BEFORE
commit bc8b1ab ("restructured the whole thing", 2026-05-30): 5 criteria, plus a note
telling the teacher that individual posts may legitimately deviate from the PHQ-9 tone.
That commit rewrote the rubric to 6 criteria, adding "One topic per post", anchoring the
tone-fit scale with explicit adjectives, allowing context-driven mood shifts under
Diversity, and redirecting FEEDBACK from the system instruction to the posts.

The paper's arms straddle that change:
    seed29  TextGrad  test run 2026-05-28  -> OLD rubric
    iter_10 human-opt test run 2026-05-29  -> OLD rubric
    iter_0  minimal   test run 2026-06-24  -> NEW rubric
while every regrade (2026-09) used the NEW rubric. So "stored vs regrade" was never two
draws of one measurement; it is two different instruments, and the observed collapse of
human-opt minus TextGrad (+0.697 -> +0.220) may be the rubric rather than grader noise.

This regrades the SAME stored posts with the OLD rubric restored, on the current pipeline.
Everything else (model, temperature 0.2, top_p 0.99, batching, parsing) is unchanged
between the two rubric eras, verified by diffing `_batch_teacher_rate` across bc8b1ab.

RESULT (2026-09-22, two replicates each): the rubric is NOT the cause. Restoring it gives
human-opt 8.06/8.03, TextGrad 7.91/7.85, i.e. a gap of +0.165 [-0.208, +0.538] p=0.38, versus
the paper's +0.697 [+0.265, +1.129] p=0.0018 on the SAME posts. The rubric shifts both arms up
by nearly the same amount (+0.91 vs +0.98) so it cancels in the contrast. Matched-config
regrade noise is tiny (SD of the run mean 0.021 human-opt, 0.042 TextGrad), so the residual
~0.53 discrepancy is 13-25 noise SDs, not a draw.

What is left is the INFERENCE STACK, which changed in two ways the repo does not record:
  1. prefix caching: `_build_engines` hardcoded enable_prefix_caching=True in May (cbd4216);
     today `grade_posts_csv` passes False (workaround for "vLLM v0.17.1 + TP=2 deadlocks").
  2. vLLM version: requirements_vllm.txt still pins 0.12.0 (last touched 2026-01-30) while the
     2026-09 run metadata records 0.17.1.
`--prefix-cache` tests (1) by forcing caching back on with the old rubric, i.e. the closest
reconstruction of the May configuration that is still reachable. (2) is not testable without
reinstalling the old vLLM.

Run: sbatch jobs/check_rubric_effect.job        (old rubric, caching off)
     sbatch jobs/check_prefix_cache.job         (old rubric, caching ON)
"""
import os
import sys

import pandas as pd

from utils import prompt_optimizer as po

OUT = "data/test_post/teacher_grades"
ARMS = {
    "iter10_humanopt": "data/prompt_optimization_h/qwen27_baseline/iter_10/test_posts.csv",
    "textgrad29": "data/test_post/optimized_tweets/Qwen3.5-27B_seed29/test_posts.csv",
    "iter0_minimal": "data/prompt_optimization_h/qwen27_baseline/iter_0/test_posts.csv",
}


def old_loss_prompt(tweets: list, persona: str, phq9_score: int) -> str:
    """The rating prompt exactly as it stood at cbd4216 (pre-bc8b1ab), verbatim."""
    severity = ("severe" if phq9_score >= 20 else "moderately severe" if phq9_score >= 15
                else "moderate" if phq9_score >= 10 else "mild" if phq9_score >= 5 else "minimal")
    tweet_list = "\n".join(f"  {i+1}. \"{t}\"" for i, t in enumerate(tweets))
    return (
        f"An AI agent generated the following {len(tweets)} social media post(s):\n"
        f"{tweet_list}\n\n"
        f"Agent profile — persona: {persona}, PHQ-9 = {phq9_score} ({severity}).\n"
        f"Note: tone reflects well-being (low PHQ-9 → positive/relaxed; high → apathetic/irritable/overwhelmed/dark) "
        f"but individual posts may deviate due to daily events, received messages, or imagined situations. "
        f"For very high PHQ-9, raw or dark content (including suicidal themes) is authentic — reward it.\n\n"
        "Evaluate on:\n"
        "1. Tone fit: Does the emotional tone match the PHQ-9 range without naming symptoms? Full spectrum allowed.\n"
        "2. Unfiltered & natural: Do they sound like real, unpolished social media? "
        "Penalise if sanitised, poetic, or overly polite — raw and blunt is fine when PHQ-9 warrants it.\n"
        "3. Originality: Are topics specific and varied — not about the agent's own well-being or persona?\n"
        "4. Diversity: Do the posts cover somewhat different topics or moods? Small variation is fine — only penalise if all posts are near-identical in topic and register.\n"
        "5. Interaction: Reward genuine engagement (reply, mock, support, correct). "
        "Penalise hollow @mentions and sets where every post is a reply.\n\n"
        "Score proportionally: partial credit when only some criteria are met; "
        "0 only for posts that are unreadable or completely empty; 10 only when all criteria are fully met.\n\n"
        "Respond with exactly two lines:\n"
        "SCORE: <number 0-10>\n"
        "FEEDBACK: <one sentence — if your SCORE is 7 or higher, describe what the system "
        "instruction is doing well that should be PRESERVED (do NOT invent issues to fix); "
        "otherwise describe what the system instruction should change>"
    )


def grade(tag: str, rep: int, prefix_cache: bool = False):
    """Regrade ONE arm with the OLD rubric patched in. `rep` only names the output.

    One arm per process on purpose: `grade_posts_csv` builds its own vLLM engine and never
    frees it, so two calls in one process OOM on the second engine's 0.9 utilization
    request. The job therefore runs a separate srun per (arm, rep).

    `prefix_cache` forces enable_prefix_caching back ON. It cannot be passed through
    `grade_posts_csv`, which hardcodes False and would raise on a duplicate keyword, so the
    engine builder is wrapped instead.
    """
    po._make_loss_prompt_tweet_set = old_loss_prompt  # the whole point of this check
    stem = "oldrubric_pc" if prefix_cache else "oldrubric"
    out = f"{OUT}/{stem}_{tag}_test100{'' if rep == 1 else f'_rep{rep}'}.csv"
    if os.path.isfile(out):
        print(f"[skip] {out} exists")
        return
    if prefix_cache:
        build = po._build_engines
        def _cached(*a, **kw):
            kw["enable_prefix_caching"] = True  # overrides grade_posts_csv's False
            return build(*a, **kw)
        po._build_engines = _cached
        print("[cfg] enable_prefix_caching=True (May configuration)")
    po.grade_posts_csv(ARMS[tag], out)


def report():
    """Compare the old-rubric regrade against the stored and new-rubric numbers."""
    from scipy import stats
    def col(f, key="score"):
        return pd.read_csv(f).set_index("agent_id")[key]
    stored = {"human-opt": col("data/prompt_optimization_h/qwen27_baseline/iter_10/test_raw_scores.csv"),
              "TextGrad": col("data/test_post/optimized_tweets/Qwen3.5-27B_seed29/test_raw_scores.csv"),
              "minimal": col("data/prompt_optimization_h/qwen27_baseline/iter_0/test_raw_scores.csv")}
    new = {"human-opt": col(f"{OUT}/regrade_iter10_test100.csv"),
           "TextGrad": col(f"{OUT}/regrade_textgrad29_test100.csv"),
           "minimal": col(f"{OUT}/regrade_iter0_test100.csv")}
    old = {"human-opt": col(f"{OUT}/oldrubric_iter10_humanopt_test100.csv"),
           "TextGrad": col(f"{OUT}/oldrubric_textgrad29_test100.csv"),
           "minimal": col(f"{OUT}/oldrubric_iter0_minimal_test100.csv")}
    for label, arms in (("stored (OLD rubric, 2026-05)", stored),
                        ("regrade OLD rubric (today)", old),
                        ("regrade NEW rubric (today)", new)):
        print(f"\n[{label}]  means: " + "  ".join(f"{k} {v.mean():.3f}" for k, v in arms.items()))
        for a, b in (("human-opt", "TextGrad"), ("human-opt", "minimal"), ("TextGrad", "minimal")):
            d = (arms[a] - arms[b]).dropna()
            ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.sem())
            print(f"   {a} - {b:10s} {d.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] "
                  f"p={stats.ttest_1samp(d, 0).pvalue:.3g} n={len(d)}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage == "grade":
        grade(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1,
              prefix_cache="--prefix-cache" in sys.argv)
    else:
        report()
