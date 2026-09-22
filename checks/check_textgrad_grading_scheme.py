"""Grade the paper's stored posts with TextGrad's OPTIMISATION-time scheme, not the test scheme.

The prompts were optimised against one grader and reported against another:

  OPTIMISATION (call_optimizer_tweets, what TextGrad minimised)
      combined = tg.sum(predictions)                     # the student's RAW outputs
      loss_fn  = tg.TextLoss(_make_loss_prompt_tweet_set(parsed_tweets, persona, phq9))
      loss     = loss_fn(combined)
    so the teacher receives TextGrad's TextLoss template, containing the rubric (with the
    posts numbered inside it) AND the variable's value (the posts again), routed through
    `_TeacherEngine.generate` with max_tokens 10240.

  TEST / REPORTING (_evaluate_tweet_instruction -> _batch_teacher_rate)
      the rubric prompt ALONE, max_tokens 4096.

Both use temperature 0.2, top_p 0.99, n=1, thinking enabled, and the same rubric text, so
the schemes differ only in the framing and the token budget. Every number in the paper, for
BOTH the TextGrad and the human arm, came from the TEST scheme; the TextLoss scheme is what
the TextGrad prompts were actually selected under. Grading with it puts the TextGrad arm on
the instrument it was optimised against, which is what a fair prompt-vs-prompt comparison
arguably requires.

LIMITATION, stated up front: the optimizer put the student's RAW outputs in `combined` and
the PARSED tweets in the rubric. Only the parsed tweets were ever stored (`test_posts.csv`),
so `combined` here is rebuilt from parsed text. The framing is reproduced; the raw/parsed
asymmetry is not.

Rubric era matters too: the prompts were optimised 2026-05-21/22 and tested 2026-05-28/29,
both BEFORE commit bc8b1ab (2026-05-30) rewrote the rubric 5 -> 6 criteria, so `--rubric old`
(the default) is the era-correct choice. See checks/check_rubric_effect.py.

Run: sbatch jobs/check_textgrad_grading_scheme.job
"""
import os
import sys

import pandas as pd
import textgrad as tg

from utils import prompt_optimizer as po
from check_rubric_effect import ARMS, old_loss_prompt

OUT = "data/test_post/teacher_grades"


def grade(tag: str, rep: int, rubric: str = "old"):
    """Score one arm's stored posts through tg.TextLoss, as the optimizer did.

    One arm per process: `_build_engines` is never freed, so a second engine in the same
    process OOMs on its 0.9 utilization request.
    """
    stem = f"tgloss_{rubric}"
    out = f"{OUT}/{stem}_{tag}_test100{'' if rep == 1 else f'_rep{rep}'}.csv"
    if os.path.isfile(out):
        print(f"[skip] {out} exists")
        return
    make_prompt = old_loss_prompt if rubric == "old" else po._make_loss_prompt_tweet_set
    print(f"[cfg] TextLoss framing, {rubric} rubric, max_tokens 10240")

    blocks, answers, personas, agent_ids = po.parse_tweets_with_phq9_csv(ARMS[tag])
    blocks = [[t for t in b if t not in ("NO_POST", "NO_TWEET")] for b in blocks]
    tp = len((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(","))
    # _build_engines also registers the teacher as TextGrad's backward engine; prefix caching
    # left ON to match the May configuration.
    _, teacher = po._build_engines(po.QWEN_27, tp, 0.90, max_model_len=16384,
                                   enable_prefix_caching=True)

    # tg.TextLoss(rubric)(tg.sum(posts)) reduces, via LLMCall.forward, to exactly
    #   teacher(content="\n".join(posts), system_prompt=rubric)
    # and _TeacherEngine.generate then appends the concise note to the system role and sends
    # [system, user] with thinking on at max_tokens 10240. Replicating that directly lets all
    # 100 blocks go through vLLM in ONE batched call; calling TextLoss per block runs ~45 s
    # each (no batching), which is ~75 min per arm and overruns the wall clock.
    concise = ("Keep your thinking short. In one or two sentences, identify the key point and "
               "then answer. Do not re-examine the same idea or draft multiple alternatives — "
               "one pass is enough.")
    inputs = []
    for block, persona, phq9 in zip(blocks, personas, answers):
        sys_prompt = make_prompt(block, persona, int(phq9)) + "\n" + concise
        conv = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": "\n".join(block)}]
        ids = teacher.base.tokenizer.apply_chat_template(
            conv, tokenize=True, add_generation_prompt=True, enable_thinking=True)
        if len(ids) > 16384:
            ids = ids[-16384:]
        inputs.append({"prompt_token_ids": ids})
    print(f"[tgloss] batching {len(inputs)} blocks in one vLLM call (max_tokens 10240)")
    sp = po.SamplingParams(temperature=0.2, max_tokens=10240, top_p=0.99, n=1)
    raw = teacher.base.client.generate(inputs, sp)

    rows = []
    for resp, block, persona, phq9, aid in zip(raw, blocks, personas, answers, agent_ids):
        # NB: not `out` - that name holds the output CSV path.
        text = po.Agent.strip_model_thinking(resp.outputs[0].text)
        score, feedback = po._parse_score_feedback(text or "")
        rows.append({"model": po.QWEN_27.split("/")[-1], "agent_id": aid, "persona": persona,
                     "phq9": int(phq9), "n_posts": len(block), "score": score,
                     "feedback": feedback})

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(out, index=False)
    ok = df["score"].dropna()
    print(f"[tgloss] mean {ok.mean():.2f} +/- {ok.std():.2f} over {len(ok)} blocks "
          f"({df['score'].isna().sum()} unparsed) -> {out}")


def report(rubric: str = "old"):
    """TextLoss-scheme scores vs the test-scheme scores on the same posts."""
    from scipy import stats
    def col(f):
        return pd.read_csv(f).set_index("agent_id")["score"] if os.path.isfile(f) else None
    name = {"iter10_humanopt": "human-opt", "textgrad29": "TextGrad", "iter0_minimal": "minimal"}
    sets = {
        "PAPER stored (test scheme, May)": {
            "human-opt": col("data/prompt_optimization_h/qwen27_baseline/iter_10/test_raw_scores.csv"),
            "TextGrad": col("data/test_post/optimized_tweets/Qwen3.5-27B_seed29/test_raw_scores.csv"),
            "minimal": col("data/prompt_optimization_h/qwen27_baseline/iter_0/test_raw_scores.csv")},
        "test scheme, old rubric, today": {
            name[t]: col(f"{OUT}/oldrubric_{t}_test100.csv") for t in ARMS},
        f"TextLoss scheme, {rubric} rubric, today": {
            name[t]: col(f"{OUT}/tgloss_{rubric}_{t}_test100.csv") for t in ARMS},
    }
    for label, arms in sets.items():
        if any(v is None for v in arms.values()):
            print(f"\n[{label}]  (incomplete)")
            continue
        print(f"\n[{label}]  means: " + "  ".join(f"{k} {v.mean():.3f}" for k, v in arms.items()))
        for a, b in (("human-opt", "TextGrad"), ("human-opt", "minimal"), ("TextGrad", "minimal")):
            d = (arms[a] - arms[b]).dropna()
            ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.sem())
            print(f"   {a} - {b:10s} {d.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] "
                  f"p={stats.ttest_1samp(d, 0).pvalue:.3g} n={len(d)}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    rub = "new" if "--rubric" in sys.argv and sys.argv[sys.argv.index("--rubric") + 1] == "new" else "old"
    if stage == "grade":
        grade(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 1, rub)
    else:
        report(rub)
