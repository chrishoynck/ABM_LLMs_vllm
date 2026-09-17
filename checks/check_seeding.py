"""Seeding check: is seeded generation reproducible, and does it make agents repeat each other?

Reads the runs written by checks/check_seeding.job (data/sensitivity/seed_test/<tag>/
<mode>_seed<S>[b]/posts.csv) next to the unseeded SA reps (<sa root>/phq9/mild/rep_*/posts.csv)
and prints, per model and mode:

  reproducible   identical posts, same seed run twice (seed42 vs seed42b); unseeded: rep1 vs rep2
  independent    identical posts, different seeds (seed42 vs seed43); unseeded: rep1 vs rep3
  first-N-rounds identical fraction in rounds 0-2 of the "independent" pair (the early-round
                 correlation seen in the unseeded Gemma reps)
  dup-in-round   fraction of posts that are an exact copy of another agent's post in the same round
  cross-agent J  mean word-set Jaccard between different agents' posts of the same round
  self-copy      fraction of posts identical to one of the agent's own earlier posts
  self J         mean Jaccard between an agent's consecutive posts
  distinct-2     unique bigrams / total bigrams over the run

Lower repetition numbers are better; reproducible should be ~1 for seeded modes and ~0 unseeded;
independent should be ~0 everywhere. Averages are over the runs of a mode. Run on CPU:
    PYTHONPATH=src .venv_vllm/bin/python checks/check_seeding.py
"""
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd

MODELS = {  # tag: (label, unseeded SA reps dir)
    "qwen": ("Qwen3.5-27B", "data/sensitivity/phq9/mild"),
    "gemma4": ("Gemma-4-31B-it", "data/sensitivity/gemma4/phq9/mild"),
}
MODES = {  # mode: (reproducibility pair, independence pair, all runs)
    "unseeded": (("rep_1", "rep_2"), ("rep_1", "rep_3"), ["rep_1", "rep_2", "rep_3"]),
    "shared":   (("shared_seed42", "shared_seed42b"), ("shared_seed42", "shared_seed43"),
                 ["shared_seed42", "shared_seed42b", "shared_seed43"]),
    "agent":    (("agent_seed42", "agent_seed42b"), ("agent_seed42", "agent_seed43"),
                 ["agent_seed42", "agent_seed42b", "agent_seed43"]),
}
_WORD = re.compile(r"[a-z0-9']+")


def _load(path):
    d = pd.read_csv(path)
    d["tweet"] = d["tweet"].fillna("").astype(str)
    return d.sort_values(["step", "agent_id"]).reset_index(drop=True)


def _words(t):
    return set(_WORD.findall(t.lower()))


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def identical(a, b, rounds=None):
    m = a.merge(b, on=["agent_id", "step"], suffixes=("_a", "_b"))
    if rounds is not None:
        m = m[m["step"].isin(rounds)]
    return float((m["tweet_a"] == m["tweet_b"]).mean()) if len(m) else np.nan


def repetition(d):
    dup, cross = [], []
    for _, g in d.groupby("step"):
        posts = g["tweet"].tolist()
        dup += [posts.count(p) > 1 for p in posts]
        ws = [_words(p) for p in posts]
        cross += [_jaccard(x, y) for x, y in combinations(ws, 2)]
    self_copy, self_j = [], []
    for _, g in d.groupby("agent_id"):
        posts = g.sort_values("step")["tweet"].tolist()
        self_copy += [p in posts[:i] for i, p in enumerate(posts) if i > 0]
        ws = [_words(p) for p in posts]
        self_j += [_jaccard(x, y) for x, y in zip(ws[:-1], ws[1:])]
    bigrams = []
    for p in d["tweet"]:
        w = _WORD.findall(p.lower())
        bigrams += list(zip(w[:-1], w[1:]))
    return {"dup-in-round": np.mean(dup), "cross-agent J": np.mean(cross),
            "self-copy": np.mean(self_copy), "self J": np.mean(self_j),
            "distinct-2": len(set(bigrams)) / max(len(bigrams), 1)}


def main():
    rows = []
    for tag, (label, sa_dir) in MODELS.items():
        test_dir = f"data/sensitivity/seed_test/{tag}"
        for mode, (repro, indep, runs) in MODES.items():
            base = sa_dir if mode == "unseeded" else test_dir
            paths = {r: f"{base}/{r}/posts.csv" for r in runs}
            missing = [p for p in paths.values() if not os.path.isfile(p)]
            if missing:
                print(f"[skip] {label} {mode}: missing {missing[0]}")
                continue
            data = {r: _load(p) for r, p in paths.items()}
            reps = [repetition(d) for d in data.values()]
            row = {"model": label, "mode": mode,
                   "reproducible": identical(data[repro[0]], data[repro[1]]),
                   "independent": identical(data[indep[0]], data[indep[1]]),
                   "first-3-rounds": identical(data[indep[0]], data[indep[1]], rounds=[0, 1, 2])}
            row.update({k: float(np.mean([r[k] for r in reps])) for k in reps[0]})
            rows.append(row)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    os.makedirs("data/sensitivity/seed_test", exist_ok=True)
    out.to_csv("data/sensitivity/seed_test/summary.csv", index=False)
    print("\n-> data/sensitivity/seed_test/summary.csv")


if __name__ == "__main__":
    main()
