"""Audit the gradings the paper reports: were all arms optimized AND scored under one rubric?

The teacher rubric (`_make_loss_prompt_tweet_set`) is versioned code and it CHANGED in
commit bc8b1ab, 2026-05-30 ("restructured the whole thing"): 5 criteria -> 6, adding
"One topic per post", anchoring tone-fit, loosening Diversity, and dropping the note that
posts may deviate from tone. The optimizer uses that same function as its LOSS, so the
rubric governs both how a prompt was optimized and how it was later scored.

This script checks, per arm, that the two line up, plus the usual hygiene (shared personas,
matching decoding config, parse failures, score distribution). It is CPU-only and reads
only files already on disk.

Run: PYTHONPATH=src ./.venv_vllm/bin/python checks/check_paper_grading_consistency.py
"""
import os
import re
import subprocess

import numpy as np
import pandas as pd

RUBRIC_CHANGE = pd.Timestamp("2026-05-30")  # commit bc8b1ab
BASE = "data/prompt_optimization_h/qwen27_baseline"
TG = "data/test_post/optimized_tweets/Qwen3.5-27B_seed{s}"
# arm -> (prompt file, its optimization mtime source, test scores, test posts)
ARMS = {
    **{f"human iter_{i}": (f"{BASE}/iter_{i}/prompt.txt", f"{BASE}/iter_{i}/test_raw_scores.csv",
                           f"{BASE}/iter_{i}/test_posts.csv", f"{BASE}/iter_{i}/eval_meta.txt")
       for i in (0, 7, 8, 9, 10)},
    **{f"TextGrad seed{s}": (TG.format(s=s) + "/optimized_instruction_tweet.txt",
                             TG.format(s=s) + "/test_raw_scores.csv",
                             TG.format(s=s) + "/test_posts.csv",
                             TG.format(s=s) + "/eval_meta.txt") for s in (24, 25, 28, 29, 53)},
}


def _meta(path, key):
    """One `key: value` line out of an eval_meta.txt, or None."""
    if not os.path.isfile(path):
        return None
    for line in open(path):
        if line.startswith(key):
            return line.split(":", 1)[1].strip()
    return None


def _era(ts):
    """Which rubric was live at `ts`."""
    if ts is None or pd.isna(ts):
        return "?"
    return "OLD(5)" if pd.Timestamp(ts) < RUBRIC_CHANGE else "NEW(6)"


def rubric_eras():
    """Per arm: when the prompt was written, when it was scored, and the rubric live at each."""
    print("=" * 108)
    print("1. RUBRIC ERA  (the optimizer's loss IS the rubric, so 'optimized under' matters as much as 'scored under')")
    print("=" * 108)
    print(f"{'arm':20s} {'prompt written':20s} {'era':8s} {'test run':20s} {'era':8s} {'mean':>7s}  consistent?")
    rows = []
    for name, (prompt, scores, _posts, meta) in ARMS.items():
        w = pd.Timestamp(os.path.getmtime(prompt), unit="s") if os.path.isfile(prompt) else None
        t = _meta(meta, "timestamp")
        t = pd.Timestamp(t) if t else None
        m = pd.read_csv(scores)["score"].mean() if os.path.isfile(scores) else np.nan
        ok = "YES" if _era(w) == _era(t) != "?" else "** NO **"
        print(f"{name:20s} {str(w)[:19]:20s} {_era(w):8s} {str(t)[:19]:20s} {_era(t):8s} {m:>7.3f}  {ok}")
        rows.append({"arm": name, "written": w, "tested": t, "era_written": _era(w),
                     "era_tested": _era(t), "mean": m})
    df = pd.DataFrame(rows)
    bad = df[df.era_written != df.era_tested]
    print(f"\n  -> {len(bad)} of {len(df)} arms were scored under a DIFFERENT rubric than they were optimized under"
          + (f": {', '.join(bad.arm)}" if len(bad) else ""))
    same = df[df.era_written == df.era_tested]
    print(f"  -> arms mutually comparable (all OLD, optimized and scored): "
          f"{', '.join(same[same.era_tested == 'OLD(5)'].arm)}")
    return df


def shared_inputs():
    """Every arm must score the same personas with the same decoding, or the means are not comparable."""
    print("\n" + "=" * 108)
    print("2. SHARED INPUTS  (personas, PHQ-9, decoding, neighbour pool)")
    print("=" * 108)
    ref_ids = ref_phq = None
    for name, (_p, scores, _posts, meta) in ARMS.items():
        if not os.path.isfile(scores):
            continue
        d = pd.read_csv(scores)
        ids, phq = tuple(d.agent_id), tuple(d.phq9)
        if ref_ids is None:
            ref_ids, ref_phq = ids, phq
        fields = {k: _meta(meta, k) for k in ("seed", "sample_seed", "num_agents",
                                              "tweets_per_sample", "neighbor_pool_size")}
        print(f"  {name:20s} ids={'same' if ids == ref_ids else '** DIFFER **':14s} "
              f"phq9={'same' if phq == ref_phq else '** DIFFER **':14s} "
              + " ".join(f"{k}={v}" for k, v in fields.items() if v))


def grading_hygiene():
    """Parse failures, zeros, and the score distribution of each arm."""
    print("\n" + "=" * 108)
    print("3. GRADING HYGIENE  (a failed parse must be NaN, never 0)")
    print("=" * 108)
    print(f"{'arm':20s} {'n':>4s} {'NaN':>4s} {'zeros':>6s} {'min':>4s} {'max':>4s} {'mean':>7s} {'sd':>6s}")
    for name, (_p, scores, _posts, _m) in ARMS.items():
        if not os.path.isfile(scores):
            continue
        s = pd.read_csv(scores)["score"]
        print(f"{name:20s} {len(s):>4d} {s.isna().sum():>4d} {(s == 0).sum():>6d} "
              f"{s.min():>4.1f} {s.max():>4.1f} {s.mean():>7.3f} {s.std():>6.3f}")


def rubric_mechanism():
    """Does each prompt trip the rubric criterion that penalises all-reply sets?

    Both rubric eras carry "Penalise hollow @mentions and sets where every post is a reply",
    but it is 1 of 5 criteria in the OLD rubric (20% of the score) and 1 of 6 in the NEW
    (16.7%). An arm that saturates replies is therefore penalised LESS under the new rubric.
    """
    print("\n" + "=" * 108)
    print("4. REPLY SATURATION vs the rubric's all-reply penalty (criterion 5 of 5 OLD -> 6 of 6 NEW)")
    print("=" * 108)
    print(f"{'arm':20s} {'reply frac':>11s} {'all-reply blocks':>17s}   prompt demands a reply per post?")
    for name, (prompt, _s, posts, _m) in ARMS.items():
        if not os.path.isfile(posts):
            continue
        d = pd.read_csv(posts)
        per = d.assign(r=d["tweet"].astype(str).str.contains("@")).groupby("agent_id")["r"].mean()
        demands = bool(re.search(r"(question or reply|reply to another user).{0,40}every post|"
                                 r"every post.{0,40}(question or reply|reply)",
                                 open(prompt).read(), re.I | re.S)) if os.path.isfile(prompt) else False
        print(f"{name:20s} {per.mean():>11.3f} {(per == 1).mean():>16.1%}   {'YES' if demands else 'no'}")


def git_context():
    """The commit live at each arm's test run, so the code state is auditable."""
    print("\n" + "=" * 108)
    print("5. CODE STATE  (commit live when each arm was scored)")
    print("=" * 108)
    try:
        log = subprocess.run(["git", "log", "--format=%h|%ad|%s", "--date=format:%Y-%m-%d %H:%M:%S",
                              "--", "src/utils/prompt_optimizer.py"],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except Exception as e:  # not a git checkout, or git missing
        print(f"  (git unavailable: {e})")
        return
    commits = [(pd.Timestamp(d), h, s) for h, d, s in (l.split("|", 2) for l in log)]
    for name, (_p, _s, _posts, meta) in ARMS.items():
        t = _meta(meta, "timestamp")
        if not t:
            continue
        t = pd.Timestamp(t)
        live = next(((h, s) for d, h, s in commits if d <= t), None)
        print(f"  {name:20s} {str(t)[:19]}  ->  {live[0] if live else '?':9s} {live[1][:58] if live else ''}")


if __name__ == "__main__":
    rubric_eras()
    shared_inputs()
    grading_hygiene()
    rubric_mechanism()
    git_context()
